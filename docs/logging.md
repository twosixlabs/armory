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

The armory logger upon import is pre-initialized and requires no configuration. So as a
user of the library, that's all you need to know.

## logger destinations

The armory.logs system always logs to the console, this is initialized at import
time.  The module also writes to two files called

    color-log.txt
    armory-log.txt

where the first is a direct duplicate of the console, and the second is identical
but without ansi color escapes for easier parsing. These files want to be
placed in the armory `output_dir` in the same timestamped directory that holds
the output file. However, that directory name isn't known until at armory start
time so the log files are created in the configured `output_dir` at the start.

When armory knows the name of that timestamped directory it calls

    armory.logs.move_log_files(timestamp_directory)

This operation will fail with a log message if `timestamp_directory` is not on the same
filesystem as `output_dir`.

## logging level specification

As with the standard `logging` module, armory log messages are conditionally emitted
based on their level. Messages sent by armory will be logged at the INFO level
by default, or DEBUG level if `--debug` is specified.

But armory.logs also handles messages sent by libraries that we call such as: art,
tensorflow, boto, etc. The armory.logs module has a filter that is applied before
emitting messages and is configured as a dictionary mapping the originating module name
to level such as:

    filters = {
        'art': 'DEBUG',
        'tensorflow': 'INFO',
        'boto': 'WARN',
        'h5py': False,
    }

This would pass messages from art and tensorflow as expected. In this example all
messages from h5py are dropped. The boto3 library is a notable case, its messages
originate from `botocore` not `boto3` and its debug level is almost criminally verbose.
Because the different libraries have different useful levels, armory.logs uses a simple
scheme for configuration.

There is a set of default filters set at initialization. These are overridden
by:

    user_filters = {
        'art': 'DEBUG',
        'tensorflow': False
    }
    armory.logs.change_filters(user_filters)

which will override the defaults only for the specified user_filters, leaving
the other defaults as they were. On the armory command line, this set of
user specified filters would appear as

    armory run --log-level art:debug --log-level tensorflow:none

where None, none, and false are all synonyms for False. If we get good defaults
in place, `:none` should be an uncommon command option.

### drop --debug ???

The way it currently works, `--debug` sets the lower bound on what the logger
will emit regardless of filter (--log-level) specifications. So the command

    armory run --log-level art:debug

would not emit art debug messages because the global level defaults to INFO.
This is pretty confusing.

Better might be

    armory run --log-level debug

where the no-colon argument sets the global level.

TODO: develop this idea more
