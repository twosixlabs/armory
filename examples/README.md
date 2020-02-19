# Examples

This directory holds example evaluation configuration files.


## Configuration File
The configuration file is designed in a way that allows evaluations to be ran locally 
or on a cloud cluster. To do this module and class/function names need to be written 
into the configuration so that modules across different repositories can interact.


### Configuration File "Schema"
```
Adhoc (JSON or null)
    Custom configurations

Attack (JSON or null)
    module: String 
    name: String
    kwargs: JSON
    knowledge: String
    budget: JSON

Dataset (JSON or null)
    module: String
    name: String

Defense (JSON or null)
    module: String
    name: String
    kwargs: JSON

Evaluation (JSON - Required)
    eval_file: String

Metric (JSON or null)
    module: String
    name: String
    kwargs: JSON

Model (JSON or null)
    name: String
    module: String
    model_kwargs: JSON
    wrapper_kwargs: JSON

SysConfig (JSON - Required)
    docker_image: String
    external_github_repo: String
    use_gpu: Boolean
```

### External Repos
Some configurations require pulling an additional external repository into the 
container. To do this simply splecify the `organization/repo` as a string within the 
configuration file. 

In order to access private external repos you'll need to set an 
environment variable `GITHUB_TOKEN` which corresponds to a user token 
with repo access.

```
export GITHUB_TOKEN="5555e8b..."
armory run <external_repo.json>
```

Tokens can be created here:
https://github.com/settings/tokens

And need the following permissions:
![github-token-permissions](https://user-images.githubusercontent.com/18154355/72368576-5aa1c180-36cc-11ea-9c2d-b7b1ca750018.png)
