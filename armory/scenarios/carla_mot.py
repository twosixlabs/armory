"""
CARLA Multi-Object Tracking Scenario

"""


import numpy as np

from armory.scenarios.carla_video_tracking import CarlaVideoTracking
from armory.instrument.export import VideoTrackingExporter, ExportMeter


class CarlaMOT(CarlaVideoTracking):

    def run_benign(self):
        # ... TODO
        self.y_pred = np.array([0,0,0,0,0,0,0,0,0])

    def run_attack(self):
        # ... TODO
        self.x_adv, self.y_target, self.y_pred_adv = self.x, self.y, self.y