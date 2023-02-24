"""
Validate configuration files
"""

import json
import os
import sys

import jsonschema

DEFAULT_SCHEMA = os.path.join(os.path.dirname(__file__), "config_schema.json")


def _load_schema(filepath: str = DEFAULT_SCHEMA) -> dict:
    with open(filepath, "r") as schema_file:
        schema = json.load(schema_file)
    return schema


def validate_config(config: dict) -> dict:
    """
    Validates that a config matches the default JSON Schema
    """
    schema = _load_schema()

    jsonschema.validate(instance=config, schema=schema)

    return config


def load_config(filepath: str) -> dict:
    """
    Loads and validates a config file
    """
    with open(filepath) as f:
        config = json.load(f)

    return validate_config(config)


def load_config_stdin() -> dict:
    """
    Loads and validates a config file from stdin
    """
    string = sys.stdin.read()
    config = json.loads(string)

    return validate_config(config)
