import sys
import time
from typing import Callable


# Helper from armory.__main__. Duplicated due to circular import.
def _debug(parser):
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="synonym for --log-level=armory:debug",
    )
    parser.add_argument(
        "--log-level",
        action="append",
        help="set log level per-module (ex. art:debug) can be used mulitple times",
    )


def simple_progress_bar(iterable, total=None, length=40, msg=None):
    try:
        total = total if total is not None else len(iterable)
    except TypeError:
        raise ValueError("total must be specified if iterable has no length")
    bar_length = int(length)
    start_time = time.time()
    if msg is not None:
        msg = msg + " "
    else:
        msg = ""

    def update_progress(progress):
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time / progress - elapsed_time
        sys.stdout.write(
            "\r{0}[{1: <{2}}] {3:.0%} {4:.0f}s ETA ".format(
                msg,
                "=" * int(bar_length * progress),
                bar_length,
                progress,
                remaining_time,
            )
        )
        sys.stdout.flush()

    for i, item in enumerate(iterable):
        yield item
        progress = (i + 1) / total
        update_progress(progress)

    sys.stdout.write("\n")
    sys.stdout.flush()


def human_sort(x: list, key: Callable = lambda x: x) -> list:
    """Sorts a list of strings in human order.

    Args:
        x (list): The list to sort.
        key (Callable): A function that returns a string to sort by.

    Returns:
        list: The sorted list.
    """
    import re

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(item):
        return [convert(c) for c in re.split("([0-9]+)", key(item))]

    sorted_index = sorted(range(len(x)), key=lambda i: alphanum_key(x[i]))
    return [x[i] for i in sorted_index]
