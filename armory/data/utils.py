import subprocess


def curl(url: str, dirpath: str, filename: str) -> None:
    """
    Downloads a file with a specified output filename and directory
    :param url: URL to file
    :param dirpath: Output directory
    :param filename: Output filename
    """
    try:
        subprocess.check_call(["curl", "-L", url, "--output", filename], cwd=dirpath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"curl command not found. Is curl installed? {e}")
    except subprocess.CalledProcessError:
        raise subprocess.CalledProcessError
