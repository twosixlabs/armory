# Examples

This folder contains example configurations for running ARMORY.


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
