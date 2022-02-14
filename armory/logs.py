# The logging level defaults to INFO, rather than WARNING because logs are
# the main way that armory communicates to the user.
#
# console log messages are currently sent to stdout rather than stderr,
# mimicing the behavior of the previous coloredlogs package. I have kept
# this for now because I don't know what external tools are consuming the
# console output and I don't want to break them.
#
# This module creates three sinks at initialization:
#   1. stdout
#   2. output_dir/colored-log.txt
#   3. output_dir/armory-log.txt
#
# which all receive the same messages, with armory-log.txt lacking the ansi color codes
# of the other two. This may be considered redundant, but I don't expect the
# near-duplication to be particularly costly.
#
# The standard logging level overlaps the per-module filter definitions, so we eschew
# the logging level entirely. For example, if a user requests `--log art:debug`, we'd
# have to adjust the level to the lowest one requested. It might be easier (and no less performant)
# to set the level to 0 and let the filters do the selection work. Note the presence
# of a "" null filter which provides the level for all sources not specified.  We still
# need to track whether any DEBUG or lower filter has be set so that is_debug() is accurate.

import sys
import datetime
import loguru
import logging
from typing import List, Dict

import armory.paths

default_message_filters = {
    "": "INFO",
    "armory": "INFO",
    "art": "INFO",
    "boto": "INFO",
}

output_dir: str = armory.paths.runtime_paths().output_dir
filters: Dict[str, str] = default_message_filters
debug_requested: bool = False
logger_ids: List[int] = []


def is_debug(filters) -> bool:
    """return true if there is a filter set to DEBUG or lower"""
    for level in filters.values():
        if level in (True, False):
            continue
        if loguru.logger.level(level).no <= loguru.logger.level("DEBUG").no:
            return True
    return False


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


def add_destination(sink, colorize=True):
    """add a destination to the logger"""
    # TODO: add configuration method to main (cf. docs/logging.md)

    # set the base logging level to TRACE because we are using filters to control which
    # messages are sent to the sink, so we want the level to be wide open
    base_level = "TRACE"
    new_logger = loguru.logger.add(
        sink, format=format_log, level=base_level, filters=filters, colorize=colorize
    )
    return new_logger


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
