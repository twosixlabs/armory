import datetime as dt
import logging
import time

logger = logging.getLogger(__name__)


class MyFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp

    # Defining ANSI Escape Code Colors
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32;21m"
    blue = "\x1b[34;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Setting the Formatting Structures
    FORMATS = {
        logging.DEBUG: green + "%(betterTime)s" + reset + blue + " %(name)s (%(filename)s:%(lineno)d) " + reset + grey + "[%(levelname)s] - %(message)s " + reset,
        logging.INFO: green + "%(betterTime)s" + reset + blue + " %(name)s (%(filename)s:%(lineno)d) " + reset + grey + "[%(levelname)s] - %(message)s " + reset,
        logging.WARNING: green + "%(betterTime)s" + reset + blue + " %(name)s (%(filename)s:%(lineno)d) " + reset + yellow + "[%(levelname)s] - %(message)s " + reset,
        logging.ERROR: green + "%(betterTime)s" + reset + blue + " %(name)s (%(filename)s:%(lineno)d) " + reset + red + "[%(levelname)s] - %(message)s " + reset,
        logging.CRITICAL: green + "%(betterTime)s" + reset + blue + " %(name)s (%(filename)s:%(lineno)d) " + reset + bold_red + "[%(levelname)s] - %(message)s " + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        ct = self.converter(record.created).strftime("%Y-%m-%dT%H:%M:%S")
        mmss = self.get_mmss(record.relativeCreated)
        record.betterTime = "{}(dt={})".format(ct, mmss)
        return formatter.format(record)

    def get_mmss(self, value):
        value = value / 1000.
        hh = int(float(value) / 3600.)
        value = value - 3600*hh
        mm = int(float(value) / 60.)
        ss = int(value - 60*mm)
        return "{:03d}:{:02d}:{:02d}".format(hh,mm,ss)


def setup_logger(level):
    # Getting Custom Formatter
    formatter = MyFormatter()


    # Create a filehandler object
    fh = logging.FileHandler('spam.log')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    # Create stdout handler object
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(level)
    logging.getLogger().addHandler(sh)

    # Setting Log Level for current logger
    logger.setLevel(level)
    logger.critical("Log Level set to: {}".format(level))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbose",default=logging.INFO, const=logging.DEBUG, action="store_const",help="Make Logging DEBUG")
    args = parser.parse_args()
    setup_logger(args.verbose)

    logger.info("Showing basic Log Formats")
    logger.debug("this is a debugging message")
    logger.info("this is an informational message")
    logger.warning("this is a warning message")
    logger.error("this is an error message")
    logger.critical("this is a critical message")
    print("")

    logger.info("If you want to watch, detached, open new terminal and type `tail -f spam.log`")
    for i in range(100):
        logger.info("Test: {}".format(i))
        time.sleep(1)
