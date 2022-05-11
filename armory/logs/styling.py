"""Armory Logs - Styling text in terminal / log messages

This module provides a set of functions that allow for printing
colors and bold/italic styles to the console with ANSI escape
sequences.

NOTE: These functions can be nested, if desired.
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
    """print `string` in bold font"""
    return BOLD + string + _end(string)


def italic(string):
    """print `string` in italic font"""
    return ITALIC + string + _end(string)


def underline(string):
    """print `string` in underline font"""
    return UNDERLINE + string + _end(string)


def red(string):
    """print `string` in red color font"""
    return RED + string + _end(string)


def green(string):
    """print `string` in green color font"""
    return GREEN + string + _end(string)


def yellow(string):
    """print `string` in yellow color font"""
    return YELLOW + string + _end(string)


def blue(string):
    """print `string` in blue color font"""
    return BLUE + string + _end(string)
