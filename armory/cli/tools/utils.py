# Helper from armory.__main__
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
