# Examples

This folder contains example configurations for running ARMORY.


### JSON Structure
```
Adhoc (JSON or null)
    Custom configurations

Attack (JSON or null)
    module: String 
    name: String
    kwargs = JSON
    knowledge: String
    budget: JSON

Dataset (JSON or null)
    name: String

Defense (JSON or null)
    module: String
    name: String
    kwargs: JSON

Evaluation (JSON)
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

SysConfig (JSON)
    docker_image: String
    external_github_repo: String
    gpu
        use_gpu: Boolean
        gpu_ids: Array 
```

### External Repos
In order to access private external repos you'll need to set an 
environment variable `GITHUB_TOKEN` which corresponds to a user token 
with repo access.

```
export GITHUB_TOKEN="5555e8b..."
python run_evaluation.py examples/external_repo.json
```

Tokens can be created here:
https://github.com/settings/tokens

And need the following permissions:
![github-token-permissions](https://user-images.githubusercontent.com/18154355/72368576-5aa1c180-36cc-11ea-9c2d-b7b1ca750018.png)
