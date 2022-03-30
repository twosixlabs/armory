import pytest
import armory.utils.s3 as as3

pytestmark = [pytest.mark.online]


def test_upload_download(tmp_path):
    dir = tmp_path / "crap"  # tmp_path is pytest Pathlib fixture
    dir.mkdir()
    fname = str(dir / "test.txt")
    testTXT = "Test File: 123456"
    with open(fname, "w") as f:
        f.write(testTXT)

    as3.upload("ds-noodle", "test.txt", fname, show_progress=False)

    newfname = as3.download(
        "ds-noodle",
        "test.txt",
        str(dir / "returned_test.txt"),
        show_progress=False,
        use_cache=False,
    )

    assert newfname == str(dir / "returned_test.txt")

    with open(newfname, "r") as f:
        data = f.read()

    assert data == testTXT
