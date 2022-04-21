# using the github actions self-hosted runner

There is a GPU equipped self-hosted runner available for CI actions in
twosixlabs/armory. The machine runner2.gardproject.org is an AWS p2.xlarge
GPU instance with 2TB of storage, 64GB core, 4 VCPUs, and a Tesla K80 with 16GB.

To direct a CI job, to it add the `runs-on` directive like

    docker-build:
        name: Build Docker Image
        runs-on: gpu

in a workflow file. The runs-on directive for github hosted runners has
historically been `ubuntu-18.04` and remains that way for all but the
jobs in `end2end_cron.yml`.

The runner2 is a stateful server, unlike the GitHub hosted runners which
build everything they need from scratch.  In particular, docker images persist
from run to run, as does the `~/.armory` cache of data, weights, and outputs.
This affords some great speed-ups, for example, re-running the end-to-end tests
spends only 9 seconds confirming that nothing new needs to be built.

Unfortunately, cached data always runs the risk that it gets out of sync with
the source.  The use of SCM versioning on docker images mitigates many sync problems,
but we should be mindful that old state could pollute self-hosted runs.

The software on runner2.gardproject.org starts from a sabot-evalbox-20 base because we
know that is well tested via heavy use of `sabot-evalbox`. The GitHub Action runner
service is pure stock code installed using the [standard installation
instructions](https://github.com/twosixlabs/armory/settings/actions/runners/new).
It also needed a git version increase (2.36) so that full checkout works so that
SCM versions are correct.

# Limitations

In the current configuration, we have only one daemon on runner2.gardproject.org
which means that every job that `runs-on: gpu` will be serialized by GitHub Actions.
That is, GitHub sets up a jobs FIFO and the self-hosted runner picks the first job and
runs it to completion or failure.

This might cause unreasonable waits for CI end-to-end. If so we can fan out horizontally
because running two daemons on one box is not recommended by Github.

The GPUs have the largest per-card memory of any AWS offering, but that 16GB is
tight. We are familiar with this limitation from sabot-evalbox experience.
