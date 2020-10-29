def pytest_addoption(parser):
    parser.addoption(
        "--scenario-file",
        action="append",
        default=[],
        help="list of scenario files to evaluate",
    )


def pytest_generate_tests(metafunc):
    if "scenario_file" in metafunc.fixturenames:
        metafunc.parametrize(
            "scenario_file", metafunc.config.getoption("scenario_file")
        )
