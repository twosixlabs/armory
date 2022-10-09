import os
import sys
import json
import subprocess

from pathlib import Path

from libs.workflows import Workflow

from armory import paths as armory_paths

try:
  # TODO: Make the following module stateless on import:
  from armory.scenarios.main import get as get_scenario
except:
  # TODO: See above
  ...


class ScenarioWorkflow(Workflow):
  def setup(self):
    if len(self.config['args']) < 2:
      raise ValueError("Missing args!")

    if self.config['args'][0] not in ["generate-matrix", "run-scenario"]:
      raise ValueError("Invalid command!")

    self.config['app']['command'] = self.config['args'][0]
    self.config['app']['scenario_path'] = Path(self.config['args'][1]),
    self.config['app']['result_path'] = armory_paths.runtime_paths().output_dir

    self.results = matrix = {
      "prod": False,
      "platform": "ubuntu-latest",
      "scenarios": []
    }
    # matrix["platform"].append("macOS-latest")
    # if (is_prod_env := os.environ.get('ENV_PRODUCTION', "false")) == "false":
    #   matrix["prod"] = bool(is_prod_env.title())
    return True


  def run(self):
    if self.config['app']['command'] == "generate-matrix":
      self.generate_matrix()
    elif self.config['app']['command'] == "run-scenario":
      self.run_scenario()


  def generate_matrix(self):
    self.results["scenarios"] = [
      {
        "scenario_path": f.as_posix()
      }
      for f in Path(self.config['args'][1]).glob("**/*.json")
    ]

    return json.dumps(self.results["scenarios"])
    matrix_out = f'::set-output name=matrix::{json.dumps(self.results["scenarios"])}'
    subprocess.Popen(['echo', matrix_str], bufsize=1)


  def run_scenario(self):
      runner = get_scenario(scenario, check_run=True).load()
      scenario_log_path, scenario_log_data = runner.evaluate()
      print(scenario_log_data)


if __name__ == "__main__":
  workflow = ScenarioWorkflow.init(sys.argv[1:])
