Armory No-Docker Mode
=======================
Reqs
 - Python >= 3.7 <= 3.9
Create new venv and activate
run
```bash
pip install --upgrade pip==22.0.3
pip install -r no-docker-req.txt
pip install -e . 
```

Now that this is working, to test the environment use:
```bash
pytest -s ./tests/unit/test_no_docker.py
```

This runs a series of configs in a variety of ways to ensure that 
the environment is operating as expected.  

## Run baseline CIFAR-10 config
Now to see if everything is operating correctly you can run a config.
For example the [CIFAR-10 Short](../scenario_configs/no_docker/cifar_short.json).
can be run with:
```bash
armory run ./scenario_configs/no_docker/cifar_short.json --no-docker
```
