# The logging level defaults to INFO, rather than WARNING because logs are the main way
# that armory communicates to the user.
#
# console log messages are currently sent to stdout rather than stderr, mimicing the
# behavior of the previous coloredlogs package. I have kept this for now because I don't
# know what external tools are consuming the console output and I don't want to break
# them.
#
# This module creates the stdout sink at initialization, so `log.debug()` and friends
# are available at import. The files
#   1. output_dir/colored-log.txt
#   2. output_dir/armory-log.txt
#
# cannot be created until after argument parsing completes so we know where to put them.
# As soon as __main__ sets armory.paths to DOCKER or NO_DOCKER, it can call
# make_logfiles(directory) to create the sink files. After make_logfiles is called,
# all three sinks are receive the same messages.
#
# The armory-log.txt file does not include ansi color codes for easier parsing. This may
# be considered redundant, but I don't expect the near-duplication to be particularly
# costly.

import datetime
import functools
import logging
import os
import sys
from typing import List

import loguru

log = loguru.logger

log.level("PROGRESS", no=15, color="<blue>", icon="\N{downwards quadruple arrow}")
log.level("METRIC", no=24, color="<blue>", icon="\N{downwards quadruple arrow}")

default_message_filters = {
    "": "WARNING",
    "armory": "INFO",
    "art": "INFO",
    "docker": "INFO",
    "botocore": "WARNING",
    "matplotlib": "INFO",
    "s3transfer": "WARNING",
    "tensorflow": "WARNING",
    "tensorflow_cpp": "CRITICAL",
    "urllib3": "INFO",
    "absl": False,
    "h5py": False,
    "avro": False,
}


filters = default_message_filters
# need to store filter overrides because we have to pass them to scenarios.main
added_filters = {}
logfile_directory = None


def is_debug() -> bool:
    """return true if there is a filter set to DEBUG or lower"""
    global filters

    for level in filters.values():
        if level in (True, False):
            continue
        if log.level(level).no <= log.level("DEBUG").no:
            return True
    return False


def is_progress() -> bool:
    """return true if detailed progress messages are needed"""
    # this is used by the progress meters which should not be shown if false

    global filters
    if "armory" not in filters:
        return False

    return log.level(filters["armory"]).no <= log.level("PROGRESS").no


def format_log(record) -> str:
    """loads the record.extra with a friendly elapsed and return a format using it"""
    record["extra"]["duration"] = duration_string(record["elapsed"])
    message = (
        "{time:YYYY-MM-DD HH:mm:ss} {extra[duration]:>3} "
        "<level>{level:<8}</level> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "{message}\n"
    )
    if record["exception"] is not None:
        message += "\n{exception}"

    return message


def duration_string(dt: datetime.timedelta) -> str:
    seconds = int(dt.total_seconds())
    periods = [("d", 60 * 60 * 24), ("h", 60 * 60), ("m", 60), ("s", 1)]

    duration = ""
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            duration += f"{period_value}{period_name}"

    return duration if duration else "0s"


def update_filters(specs: List[str], armory_debug=None):
    """add or replace specs of the form module:level and restart the 0th sink"""
    global filters
    global logfile_directory

    log.trace(f"update_filters {specs} armory_debug={armory_debug}")

    if logfile_directory is not None:
        log.error("cannot update log filters once make_logfiles is called. ignoring.")
        return

    if specs is None:
        specs = []

    if armory_debug:
        specs.append("armory:debug")
        log.trace("added armory:debug to {specs}")

    for spec in specs:
        if ":" in spec:
            key, level = spec.split(":")
        else:
            # convenience "LEVEL" is synonym for "armory:LEVEL"
            key = "armory"
            level = spec
        level = level.upper()

        if key == "tensorflow_cpp":
            set_tensorflow_cpp_loglevel(level)
            added_filters[key] = level
            continue

        try:
            log.level(level)
            filters[key] = level
            added_filters[key] = level
        except ValueError:
            log.error(f"unknown log level {spec} ignored")
            continue

    log.remove()
    add_sink(sys.stdout, colorize=True)

    # TODO: I want to see this even if levels are set low, change to trace after debugging
    log.trace(f"log levels set to {filters}")


def add_sink(sink, colorize=True):
    global filters

    # set the base logging level to TRACE because we are using filter= to control which
    # messages are sent to the sink, so we want the level to be wide open
    new_logger = loguru.logger.add(
        sink,
        format=format_log,
        level="TRACE",
        filter=filters,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )
    return new_logger


def make_logfiles(output_dir: str) -> None:
    """now that we have a output_dir start logging to the files there"""
    global logfile_directory

    logfile_directory = output_dir
    log.trace(f"make_logfiles {output_dir}")
    add_sink(f"{output_dir}/colored-log.txt", colorize=True),
    add_sink(f"{output_dir}/armory-log.txt", colorize=False),


# adapt the logging module to use the our loguru logger
# this works by adding a new root handler which sends everything to
# our InterceptHandler which extracts the extra information that loguru
# needs and sends it to the loguru logger


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = loguru.logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru.logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def log_method(entry=True, exit=True, level="DEBUG"):
    """function decorator that logs entry and exit"""

    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = log.opt(depth=1)
            if entry:
                logger_.log(
                    level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs
                )
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper


def set_tensorflow_cpp_loglevel(level: str):
    # SEE: https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
    level_map = {
        "CRITICAL": "3",
        "ERROR": "2",
        "WARNING": "1",
        "INFO": "0",
        "PROGRESS": "0",
        "DEBUG": "0",
        "TRACE": "0",
    }
    if level:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = level_map.get(level.upper())


logging.basicConfig(handlers=[InterceptHandler()], level=0)
set_tensorflow_cpp_loglevel(filters.get("tensorflow_cpp"))

if __name__ == "__main__":
    update_filters(["armory:INFO", "art:INFO"])
    for level in "trace debug info success warning error critical".split():
        log.log(level.upper(), f"message at {level}")

    def child():
        log.info("child")
        raise Exception("child")

    def parent():
        log.info("parent")
        child()

    def library_messages():
        # this should induce a flurry of log messages through the InterceptHandler
        import boto3

        s3 = boto3.client("s3")
        s3.get_bucket_location(Bucket="armory-submission-data")

        import tensorflow as tf

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        log.success("dev test complete")

    log.info("logging library messages at INFO")
    library_messages()
    update_filters(["s3transfer:DEBUG"])
    log.info("logging s3:DEBUG")
    library_messages()
    try:
        parent()
    except Exception:
        log.exception("parent failed")


# need to instantiate the filters early, replacing the default
log.remove()
add_sink(sys.stdout)
