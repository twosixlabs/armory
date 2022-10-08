import os
import sys
import json

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
      print(self.generate_matrix())


  def generate_matrix(self):
    self.results["scenarios"] = [f.as_posix() for f in self.config['app']['scenario_path'].glob("**/*.json")]
    # matrix_out = f'::set-output name=matrix::{json.dumps(self.results["scenarios"])}'
    return json.dumps(self.results["scenarios"])


if __name__ == "__main__":
  workflow = ScenarioWorkflow.init(sys.argv[1:])
