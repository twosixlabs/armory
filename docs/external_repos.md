# External Repos
You may want to include code from an external repository that is outside of your 
current working directory project. This is supported through the `external_github_repo`
field in the configuration file. After launch, the repository will be pulled into the 
evaluations tmp folder and placed on the SYS PATH so that modules can be easily 
utilized.

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
cloned repository will automatically be added to the sys path. This enables the user
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


#### Private Repos

The external repositories supports public and private GitHub repositories. If you 
would like to pull in a private repository, you'll need to set a user token as an 
environment variable before running `armory run`.

```
export ARMORY_GITHUB_TOKEN="5555e8b..."
armory run <path/to/config.json>
```

Information on creating tokens can be found here: [https://github.com/settings/tokens](https://github.com/settings/tokens)
