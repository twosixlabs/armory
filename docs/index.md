# Welcome to Armory Testbed

## Overview

ARMORY is a test bed for running scalable evaluations of adversarial defenses. 
Configuration files are used to launch local or cloud instances of the ARMORY docker 
containers. Models, datasets, and evaluation scripts can be pulled from external 
repositories or from the baselines within this project. 

Our evaluations are created so that attacks and defenses may be 
interchanged. To do this we standardize all attacks and defenses as subclasses of 
their respective implementations in IBM's [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox)


## Usage

There are three ways to interact with the armory container system.

1) `armory run`
* `armory run <path/to/config.json>`. 
This will run a [configuration file](docs/configuration_files.md) end to end. Stdout 
and stderror logs will be displayed to the user, and the container will be removed 
gracefully upon completion. Results from the evaluation can be found in your output 
directory.

* `armory run <path/to/config.json> --interactive`. 
This will launch the framework-specific container specified in the 
configuration file, copy the configuration file into the container, and provide 
the commands to attach to the container in a separate terminal and run the 
configuration file end to end while attached to the container. Similar to 
non-interactive mode, results from the evaluation can be found in the output 
directory. To later close the interactive container simply run CTRL+C from the 
terminal where this command was ran.

2) `armory launch`
* `armory launch <tf1|tf2|pytorch> --interactive`. 
This will launch a framework specific container, with appropriate mounted volumes, for 
the user to attach to for debugging purposes. A command to attach to the container will
be returned from this call, and it can be ran in a separate terminal. To later close 
the interactive container simply run CTRL+C from the terminal where this command was 
ran.

* `armory launch <tf1|tf2|pytorch> --jupyter`. 
Similar to the interactive launch, this will spin up a container for a specific 
framework, but will instead return the web address of a jupyter lab server where 
debugging can be performed. To close the jupyter server simply run CTRL+C from the 
terminal where this command was ran.

3) `armory exec` 
* `armory exec <tf1|tf2|pytorch> -- <cmd>`. 
This will run a specific command within a framework specific container. A notable use
case for this would be to run test cases using pytest. After completion of the command 
the container will be removed.

To use custom docker images with `launch` or `exec`, replace `<tf1|tf2|pytorch>` with its
full name: `<your_image/name:your_tag>`. For use with `run`, you will need to modify the
[configuration file](docs/configuration_files.md).

Note: Since ARMORY launches Docker containers, the python package must be ran on 
system host (i.e. not inside of a docker container).

### Example usage:
```
pip install armory-testbed
armory configure
git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory run official_scenario_configs/cifar10_baseline.json
```

### What is available in the container:
All containers have a pre-installed armory package installed so that baseline models, 
datasets, and scenarios can be utilized.

Additionally, volumes (such as your current working directory) will be mounted from 
your system host so that you can modify code to be ran, and retrieve outputs. 
For more information on these mounts, please see our [Docker documentation](docs/docker.md#docker-volume-mounts) 

## Scenarios
Armory provides several baseline threat-model scenarios for various data modalities. 
When running an armory configuration file, the robustness of a defense will be 
evaluated against that given scenario. For more information please see our 
[Scenario Documentation](docs/scenarios.md).
