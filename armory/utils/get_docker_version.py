# this is a hook for the use of CI
# running `python -m armory.utils.get_dotted_version`
# returns the same as `armory --version` but with the + replaced by a .
# beacuse docker tags cannot have a + in them

import armory.utils.version as version

if __name__ == "__main__":
    print(version.to_docker_tag(version.get_version()))
