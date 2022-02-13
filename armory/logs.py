# there are some design decisions that this module makes which should be
# called out explicitly

# The logging level defaults to INFO, rather than WARNING because logs are
# the main way that armory communicates to the user.
#
# console log messages are currentl sent to stdout rather than stderr,
# mimicing the behavior of the previous coloredlogs package. I have kept
# this for now because I don't know what external tools are consuming the
# console output and I don't want to break them.
#
# armory constructs its output_dir slightly before evaluation, so we need
# to delay instantiation of the file sinks until after that. It currently
# creates a colored and non-colored log file for each run, the first is
# for human reading and the second for machine consumption. The only
# only difference between them is the presense of ansi color codes.
# This may be considered redundant, but I don't expect the near-duplication
# to be particularly costly.

import sys
import datetime
import loguru
import logging

#

# because the logger was used for application state storage (particularly level),
# some of those mechanisms have to be reconstituted clumsily here

# in loguru, you can't set the level of an existing logger, so level changes
# need to remove the logger and add it again. Save the id number so we can;
# the basic logger is guaranteed to be 0 at import time
_console_logger_id = 0

# there are places where the global logger is asked if the current level is debug
# this mechanism saves the lowest level so far registered and the last named
# console level so that (for example) per-run log destinations can be created at
# eval time and use the established console level
_minimum_level = loguru.logger.level("INFO").no
_last_console_level = None


def is_debug() -> bool:
    return _minimum_level <= loguru.logger.level("DEBUG").no


def format_log(record) -> str:
    """loads the record.extra with a friendly elapsed and return a format using it"""
    record["extra"]["duration"] = duration_string(record["elapsed"])
    message = (
        "{time:YYYY-MM-DD HH:mm:ss} {extra[duration]:>3} "
        "<level>{level:<8}</level> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "{message}\n"
    )
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


def add_destination(sink, colorize=True, level=None):
    """add a destination to the logger, update the minimum_level seen"""
    global _minimum_level

    if level is None and _last_console_level is not None:
        level = _last_console_level

    format = format_log

    # TODO: reasonable defaults
    # TODO: method for configuration
    level_per_module = {"": "TRACE", "botocore": "INFO", "h5py": False}

    new_logger = loguru.logger.add(
        sink, format=format, level=level, filter=level_per_module, colorize=colorize
    )
    _minimum_level = min(loguru.logger.level(level).no, _minimum_level)
    return new_logger


def set_console_level(level: str):
    global _console_logger_id
    global _last_console_level

    # check for valid level before removing the logger
    level = level.upper()
    assert loguru.logger.level(level)

    loguru.logger.remove(_console_logger_id)
    _console_logger_id = add_destination(sys.stderr, format=format_log, level=level)
    _last_console_level = level


set_console_level("DEBUG")
log = loguru.logger

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


logging.basicConfig(handlers=[InterceptHandler()], level=0)

if __name__ == "__main__":
    for level in "trace debug info success warning error critical".split():
        log.log(level.upper(), f"message at {level}")

    # this should induce a flurry of log messages through the InterceptHandler
    import boto3

    s3 = boto3.client("s3")
    s3.get_bucket_location(Bucket="armory-submission-data")

    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    log.success("dev test complete")
