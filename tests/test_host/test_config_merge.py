import argparse

from armory import arguments


def test_config_args_merge():
    config = dict(
        sysconfig={
            "output_dir": None,
            "output_filename": "file.out",
            "num_eval_batches": 2,
            "skip_misclassified": True,
        }
    )
    args = argparse.Namespace(
        num_eval_batches=5,
        skip_misclassified=False,
        output_dir="output-dir",
        check=True,
        skip_attack=False,
    )

    (config, args) = arguments.merge_config_and_args(config, args)

    sysconfig = config["sysconfig"]
    assert sysconfig["output_dir"] == "output-dir"
    assert sysconfig["output_filename"] == "file.out"
    assert sysconfig["num_eval_batches"] == 5
    assert sysconfig["skip_misclassified"]
    assert sysconfig["check"]
    assert "skip_attack" not in sysconfig

    assert args.output_dir == "output-dir"
    assert args.output_filename == "file.out"
    assert args.num_eval_batches == 5
    assert args.skip_misclassified
    assert args.check
    assert args.skip_attack is False
