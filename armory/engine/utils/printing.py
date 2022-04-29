"""
Functions for printing bold and color to the console with ANSI escape sequences.

They can be nested:
    print(bold(red("Your Text")))
"""

END = "\033[0m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"


def _end(string):
    if string.endswith(END):
        return ""
    return END


def bold(string):
    return BOLD + string + _end(string)


def italic(string):
    return ITALIC + string + _end(string)


def underline(string):
    return UNDERLINE + string + _end(string)


def red(string):
    return RED + string + _end(string)


def green(string):
    return GREEN + string + _end(string)


def yellow(string):
    return YELLOW + string + _end(string)


def blue(string):
    return BLUE + string + _end(string)
