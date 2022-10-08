import os
import sys
from abc import ABC, abstractmethod


class Workflow:
  result = None
  config = None

  def __init__(self, args):
    args = args or sys.argv

    if (workspace := os.environ.get('GITHUB_WORKSPACE', False)):
      os.chdir(workspace)

    self.config = {
      "args": args or [ ],
      "workspace": workspace,
      "app": { }
    }

    setup = self.setup()
    if setup:
      self.exit_code = self.run()


  @classmethod
  def init(cls, args):
    return cls(args or sys.argv)


  @abstractmethod
  def setup():
    raise NotImplementedError("Method not implemented!")


  @abstractmethod
  def run():
    raise NotImplementedError("Method not implemented!")


  def Type(self):
    return "ArmoryWorkflow"


  def __str__(self):
    return str(self.Type())
