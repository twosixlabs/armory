from armory.logs import log

# test the loguru based logs module for proper pytest caplog behavior
# there is a modified caplog fixture in tests/conftest.py


def function_that_warns():
    log.warning("this is a wally warning")


def function_that_errors():
    log.error("this is a wally error")


def test_warns_function(caplog):
    function_that_warns()
    # because we known function that warns only makes one log message we know
    # that all records should be WARNING
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "wally" in caplog.text
    assert "this is a wally warning" in caplog.text


def test_error_function(caplog):
    function_that_errors()
    for record in caplog.records:
        assert record.levelname == "ERROR"
    assert "wally" in caplog.text
