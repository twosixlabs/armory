"""
Script to format all JSON in git repo.

Output is meant to look similar to `black`
Format is the same as Python's built in json.tool
    However, it overcomes some command line errors in rewriting the same file
"""

from armory.utils import files


if __name__ == "__main__":
    files.json_tool_recursive()
