# External Repos
You may want to include code from one or more external repositories that are outside of your 
current working directory project. This is supported through the `external_github_repo`
field in the configuration file. After launch, the repositories will be pulled into the 
evaluation's tmp folder and placed on the SYS PATH so that modules can be easily 
utilized.

#### Multiple Repositories
The `external_github_repo` field can be either a string for a single repo, or a JSON 
array of repositories. For example:

```
"sysconfig": {
    "external_github_repo": "hkakitani/SincNet",
}
```
```
"sysconfig": {
    "external_github_repo": ["hkakitani/SincNet", "twosixlabs/armory-example"],
}
```

#### Specifying Branches
The `external_github_repo` field in the configuration file supports specifying specific
branches to be pulled. Simply suffix the repository name with `@branch`. By default if
no branch is specified then master branch will be pulled. For example:

```
"sysconfig": {
    "external_github_repo": "hkakitani/SincNet",
}
```
```
"sysconfig": {
    "external_github_repo": "twosixlabs/armory-example@specific-branch",
}
```

#### Accessing External Modules
As mentioned, when repositories are cloned into a scenario's temporary directory, the 
cloned repository will automatically be added to the sys path, as well as the parent
directory of all cloned repositories. This enables the user 
to specify attacks, defense, scenarios from the external repo, directly in the 
evaluation config file. For example the following config snippet:
```
    "model": {
        "model_kwargs": {},
        "module": "example_models.keras.librispeech_spectrogram",
        "name": "get_art_model",
        "weights_file": "cnnspectrogram_librispeech_v1.h5",
        "wrapper_kwargs": {}
    },
    "sysconfig": {
        "external_github_repo": "twosixlabs/armory-example@master",
    }
```

Would load the keras model found within that `armory-example` external repository.

#### Custom Python Paths

While the above works for standard python module and script usage, there may be cases
where a user needs to add a different directory to the python path to enable
correct absolute imports within their repository. To do this, you will need to modify
the model module, before any module-specific imports, in the following manner:

```python
import os
import sys
module_path = globals()["__file__"]
relative_path_to_root_from_module = ".."  # this will depend on your use case
absolute_root_path = os.path.abspath(os.path.join(module_path, relative_path_to_root))
sys.path.insert(0, absolute_root_path)
```


#### Private Repos

The external repositories supports public and private GitHub repositories. If you 
would like to pull in a private repository, you'll need to set a user token as an 
environment variable before running `armory run`.

```
export ARMORY_GITHUB_TOKEN="5555e8b..."
armory run <path/to/config.json>
```

Information on creating tokens can be found here: [https://github.com/settings/tokens](https://github.com/settings/tokens)
