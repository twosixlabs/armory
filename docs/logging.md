# logging facilities and options in armory

Because the primary communication between armory and the user happens by way of logs,
the system and configuration options are broad, with reasonable defaults so that it does
the right thing to start.

## the armory logging api

In order to use the armory logger, all you need to do is import it:

    from armory.logs import log

The `log` object is the primary interface to the module and supports the following
standard functions:

    log.debug
    log.info
    log.warning
    log.error
    log.critical

The armory logger also adds two new levels

    log.trace - even more verbose than debug
    log.success - report that something completed ok

All these functions take one string argument

    log.success(f'uploaded {file} to {server}')

There are two additional log levels without a standard function:

    `"PROGRESS"` - for determining whether to log progress for downloads/uploads
    `"METRIC"` - for logging metric results

The explicit ordering of these log levels are:
```
TRACE = 5
DEBUG = 10
PROGRESS = 15
INFO = 20
METRIC = 24
SUCCESS = 25
WARNING = 30
ERROR = 40
CRITICAL = 50
```

The armory logger upon import is pre-initialized and requires no configuration. So as a
user of the library, that's all you need to know.

## logger destinations

The armory.logs system always logs to the console, this is initialized at import
time.  The module also writes to two files called

    colored-log.txt
    armory-log.txt

where the first is a direct duplicate of the console, and the second is identical
but without ansi color escapes for easier parsing. These files want to be
placed in the armory `output_dir` in the same timestamped directory that holds
the output file. However, that directory name isn't known until at armory start
time so the log files are created in the configured `output_dir` at the start.

When armory knows the name of that timestamped directory it calls

    armory.logs.make_logfiles(timestamp_directory)

TODO: The output directory needs to be created earlier in the armory initialization
so that the logfile can start near the top of run.

## logging level specification

As with the standard `logging` module, armory log messages are conditionally emitted
based on their level. Messages sent by armory will be logged at the INFO level
by default.

But armory.logs also handles messages sent by libraries that we call such as: art,
tensorflow, boto, etc. The armory.logs module has a filter that is applied before
emitting messages and is configured as a dictionary mapping the originating module name
to level such as:

    default_message_filters = {
        "": "TRACE"
        "armory": "INFO",
        "art": "INFO",
        "docker": "INFO",
        "botocore": "WARNING",
        "s3transfer": "INFO",
        "urllib3": "INFO",
        "absl": False,
        "h5py": False,
        "avro": False,
    }

which is how the logger is configured at the start. The `""` module is special, and
covers all cases which aren't otherwise specified. So we start with printing all
messages at TRACE and higher (which means all). For messages from armory, art, and
docker, INFO is fine. The botocore and s3transfer modules are crazy chatty so we raise
the threshold for them. I don't know that I've ever heard mention of debug tracing in
absl, h5py, or avro so I've disabled any messages from them.

I'm not sure these are good defaults, so I am hoping the other armory devs will
give opinions.

## armory --log-level option

The command

    armory run --log-level armory:debug --log-level art:debug ...

overrides the default log-levels for armory and art.  Because argparse allows
unique option substrings to be used, this command can be written more briefly
as

    armory run --log debug --log art:debug ...

as a convenience omitting the `module:` part of the level assumes `armory:`.
The `--debug` option become a deprecated alias for `--log-level armory:debug`.

The module names are done with a simple text match, so if you see too many messages
like

    2022-02-24 15:31:19  4s INFO     art.estimators.classification.pytorch:get_layers:986 ...

you can adjust that down with `--log art.estimators:warning` or the hyper-specific
`--log art.estimators.classification:warning`.
