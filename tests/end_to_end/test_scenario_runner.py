#!/usr/bin/env false
# -*- coding: utf-8 -*-

"""End-to-end pytest fixtures for testing scenario configs.

Example:
    $ pip install -e .[developer,engine,math,datasets,pytorch]
    $ pytest --verbose --show-capture=no tests/end_to_end/test_scenario_runner.py
"""

import re
import json
import pytest

from pathlib import Path

from armory import paths
# from armory.__main__ import run
from armory.scenarios.main import get as get_scenario


# Marks all tests in this file as `end_to_end`
pytestmark = pytest.mark.end_to_end

scenario_path = Path("scenario_configs")
host_paths    = paths.runtime_paths()
result_path   = host_paths.output_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Pre-flight checks. Future tests will  not continue if setup fails.

    Raises:
        Exception: If `scenario_path` or `result_path` are missing.
    """
    # Setup Armory paths
    paths.set_mode("host")
    if not all([Path(d).exists() for d in [scenario_path, result_path]]):
        raise Exception(f"Required application paths do not exist!")
        pytest.exit("Previous tests indicate catastrophic failure.")


def test_scenarios(capsys):
    """Scenario test.

    Args:
        scenarios (Iterator): Generator for scenario config files.
        capsys    (:pytest.CaptureFixture:`str`): PyYest builtin fixture to capture stdin/stdout.
    """

    # TODO: Handle cases where docker/torch error out; e.g. GPU not
    #       available or memory is too low. [woodall]
    skip = [
        # 'carla_video_tracking.json',
        # "so2sat_eo_masked_pgd.json",
        # "asr_librispeech_entailment.json",
        # "poisoning_gtsrb_dirty_label.json",
        # "carla_multimodal_object_detection.json",
        # "xview_robust_dpatch.json",
        # "asr_librispeech_targeted.json",
        # "poisoning_cifar10_witches_brew.json",
        # "speaker_id_librispeech.json",
        # # "cifar10_baseline.json",
        # # "mnist_baseline.json",
        # "ucf101_pretrained_masked_pgd_undefended.json",
        # "ucf101_baseline_pretrained_targeted.json",
        # "ucf101_baseline_finetune.json",
        # "ucf101_pretrained_masked_pgd_defended.json",
        # "ucf101_pretrained_frame_saliency_defended.json",
        # "ucf101_pretrained_flicker_defended.json",
        # "ucf101_pretrained_frame_saliency_undefended.json",
        # "ucf101_pretrained_flicker_undefended.json",
        # "resisc10_poison_dlbd.json",
        # "gtsrb_scenario_clbd_defended.json",
        # "gtsrb_scenario_clbd.json",
        # "gtsrb_scenario_clbd_bullethole.json",
        # "apricot_frcnn.json",
        # "apricot_frcnn_defended.json",
        # "dapricot_frcnn_masked_pgd.json",
        "mnist_baseline.json",
        # "so2sat_sar_masked_pgd_undefended.json",
        # "so2sat_sar_masked_pgd_defended.json",
        # "so2sat_eo_masked_pgd_undefended.json",
        # "so2sat_eo_masked_pgd_defended.json",
        # "librispeech_asr_snr_targeted.json",
        # "librispeech_asr_pgd_undefended.json",
        # "librispeech_asr_pgd_defended.json",
        # "librispeech_asr_pgd_multipath_channel_undefended.json",
        # "librispeech_asr_imperceptible_undefended.json",
        # "librispeech_asr_imperceptible_defended.json",
        # "librispeech_asr_kenansville_undefended.json",
        # "librispeech_asr_snr_undefended.json",
        # "librispeech_asr_kenansville_defended.json",
        # "xview_frcnn_masked_pgd_undefended.json",
        # "xview_frcnn_robust_dpatch_defended.json",
        # "xview_frcnn_targeted.json",
        # "xview_frcnn_sweep_patch_size.json",
        # "xview_frcnn_robust_dpatch_undefended.json",
        # "xview_frcnn_masked_pgd_defended.json",
        "cifar10_baseline.json",
        # "librispeech_baseline_sincnet_snr_pgd.json",
        # "librispeech_baseline_sincnet.json",
        # "librispeech_baseline_sincnet_targeted.json",
        # "resisc45_baseline_densenet121_cascade.json",
        # "resisc45_baseline_densenet121_finetune.json",
        # "resisc45_baseline_densenet121_sweep_eps.json",
        # "resisc45_baseline_densenet121_targeted.json",
        # "resisc45_baseline_densenet121.json",
        # "gtsrb_dlbd_baseline_keras.json",
        # "gtsrb_witches_brew.json",
        # "cifar10_poison_dlbd.json",
        # "cifar10_witches_brew.json",
        # "gtsrb_dlbd_baseline_pytorch.json",
        # "cifar10_dlbd_watermark_spectral_signature_defense.json",
        # "cifar10_dlbd_watermark_perfect_filter.json",
        # "cifar10_dlbd_watermark_undefended.json",
        # "cifar10_dlbd_watermark_random_filter.json",
        # "cifar10_dlbd_watermark_activation_defense.json",
        # "cifar10_witches_brew_activation_defense.json",
        # "cifar10_witches_brew_random_filter.json",
        # "cifar10_witches_brew_undefended.json",
        # "cifar10_witches_brew_perfect_filter.json",
        # "cifar10_witches_brew_spectral_signature_defense.json",
        # "gtsrb_clbd_peace_sign_random_filter.json",
        # "gtsrb_clbd_peace_sign_spectral_signature_defense.json",
        # "gtsrb_clbd_peace_sign_activation_defense.json",
        # "gtsrb_clbd_peace_sign_undefended.json",
        # "gtsrb_clbd_peace_sign_perfect_filter.json",
        # "gtsrb_clbd_bullet_holes_random_filter.json",
        # "gtsrb_clbd_bullet_holes_perfect_filter.json",
        # "gtsrb_clbd_bullet_holes_spectral_signature_defense.json",
        # "gtsrb_clbd_bullet_holes_activation_defense.json",
        # "gtsrb_clbd_bullet_holes_undefended.json",
        # "gtsrb_dlbd_peace_sign_undefended.json",
        # "gtsrb_dlbd_peace_sign_random_filter.json",
        # "gtsrb_dlbd_peace_sign_activation_defense.json",
        # "gtsrb_dlbd_peace_sign_perfect_filter.json",
        # "gtsrb_dlbd_peace_sign_spectral_signature_defense.json",
        # "gtsrb_dlbd_bullet_holes_activation_defense.json",
        # "gtsrb_dlbd_bullet_holes_spectral_signature_defense.json",
        # "gtsrb_dlbd_bullet_holes_perfect_filter.json",
        # "gtsrb_dlbd_bullet_holes_random_filter.json",
        # "gtsrb_dlbd_bullet_holes_undefended.json",
        # "gtsrb_witches_brew_random_filter.json",
        # "gtsrb_witches_brew_activation_defense.json",
        # "gtsrb_witches_brew_spectral_signature_defense.json",
        # "gtsrb_witches_brew_perfect_filter.json",
        # "gtsrb_witches_brew_undefended.json",
        # "carla_obj_det_multimodal_adversarialpatch_undefended.json",
        # "carla_obj_det_multimodal_dpatch_defended.json",
        # "carla_obj_det_multimodal_dpatch_undefended.json",
        # "carla_obj_det_dpatch_undefended.json",
        # "carla_obj_det_dpatch_defended.json",
        # "carla_obj_det_multimodal_adversarialpatch_defended.json",
        # "carla_obj_det_adversarialpatch_undefended.json",
        # "carla_video_tracking_goturn_advtextures_defended.json",
        # "carla_video_tracking_goturn_advtextures_undefended.json",
        # "defended_untargeted_snr_pgd.json",
        # "untargeted_snr_pgd.json",
        # "entailment.json",
        # "defended_targeted_snr_pgd.json",
        # "defended_entailment.json",
        # "targeted_snr_pgd.json",
        # # "cifar_short.json",
        # # "carla_short.json",
    ]


    # Skip scenarios that require large resource allocations.
    # scenarios = [ Path(f) for f in list(scenario_path.glob("**/*.json")) if f.name not in skip ]
    scenarios = [ Path(f) for f in list(scenario_path.glob("**/*.json")) if f.name in skip ]


    def scenario_runner():
        """Runs the scenario config and checks the results.

        Yields:
            str: Application stdout for the targeted scenario config file.

        Raises:
            AssertionError: If there is an error in the application stdout.
        """
        for scenario in scenarios:
            with capsys.disabled():
                print(f"\n\n{'=' * 42}")
                print(f"\tTesting: {scenario.name}")

            # scenario_path = str(scenario.absolute())
            # armory_command = [scenario_path, "--no-docker", "--check"]
            # # Run the scenario & capture the output.
            # assert run(armory_command, "armory", None) == 0, "Error occured while executing scenario."
            # out, err = capsys.readouterr()

            runner = get_scenario(scenario, check_run=True).load()
            output = runner.evaluate()

            yield output + (runner.results,)


    # Check that the results were written.
    for scenario in scenario_runner():
        # assert result[0] == 0, "Error occured while executing scenario."
        log_path, log_data, result = scenario
        with capsys.disabled():
            print(result)
            print(log_path)
    # #     result_stdout = str(result).strip()
    # #     test_results_written = "results output written to" in result_stdout.lower()
    # #     test_results_json    = result_stdout.endswith(".json")
    # #     if not all([test_results_written, test_results_json]):
    # #         assert False, f"Invalid output: {out}"
    # #         continue

    # #     # Ensure the file exists.
    # #     result_file = Path(result_stdout.split(" ")[-1])
    # #     assert result_file.exists(), f"Missing result file: {result_file}"
