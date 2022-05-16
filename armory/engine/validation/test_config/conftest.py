def pytest_addoption(parser):
    parser.addoption(
        "--model-config",
        action="append",
        default=[],
        help="serialized, json-formatted string of model configuration",
    )


def pytest_generate_tests(metafunc):
    if "model_config" in metafunc.fixturenames:
        metafunc.parametrize("model_config", metafunc.config.getoption("model_config"))
