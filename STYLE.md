# Formatting
All contributions to the repository must be formatted with [black](https://github.com/psf/black).
```
black .
```

All JSON files committed to the repository must be formatted using the following command:
```
python -m scripts.format_json
```
It is based off of Python's [json.tool](https://docs.python.org/3/library/json.html#module-json.tool)
with the `--sort-keys` argument, though overcomes an issue in 3.6 which made it unable to rewrite
the file it was reading from.
