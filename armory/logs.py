import sys
import datetime
import random
import loguru


class DurationFormatter:
    """General message formatter for loguru. Records the time of the last call to
    the formatter, and adds the duration since last call to the message header."""

    def __init__(self, fake_duration=False):
        self.last = datetime.timedelta(0)
        self.fake_duration = fake_duration

    def format(self, record):
        """loads the record.extra with the last elapsed and returns a delta string"""
        extra = record["extra"]
        delta = record["elapsed"] - self.last
        if self.fake_duration:
            delta = datetime.timedelta(seconds=random.randint(42, 1000))
        extra["duration"] = self.duration_string(delta)
        _message = "{time} {extra[duration]} {message}\n"
        _message = (
            # "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} {extra[duration]:>8}</green> | "
            "{time:YYYY-MM-DD HH:mm:ss} {extra[duration]:>8} | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>\n"
        )
        self.last = record["elapsed"]

        return _message

    def duration_string(self, dt: datetime.timedelta):
        seconds = int(dt.total_seconds())
        periods = [("d", 60 * 60 * 24), ("h", 60 * 60), ("m", 60), ("s", 1)]

        duration = ""
        for period_name, period_seconds in periods:
            if seconds > period_seconds:
                period_value, seconds = divmod(seconds, period_seconds)
                duration += f"{period_value}{period_name}"

        return duration if duration else "0s"


def add_destination(sink, format=None, level="DEBUG", colorize=True):
    loguru.logger.add(sink, format=format, level=level, colorize=colorize)


loguru.logger.remove(0)  # remove default logger
add_destination(
    sys.stdout, format=DurationFormatter(fake_duration=True).format, level="TRACE"
)

# TODO need to add_destination for colorized and non-colorized logs but we need armory.paths
# to be available and they may not be imported yet

log = loguru.logger

if __name__ == "__main__":
    for level in "trace debug info success warning error critical".split():
        log.log(level.upper(), f"message at {level}")
