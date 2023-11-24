"""
Class that handle the loading of a CARLA overhead object detection custom dataset
using PyTorch.
"""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import CocoDetection

class CarlaOverObjtDetCustom(CocoDetection):
    def __init__(
        self, 
        root: str, 
        annFile: str, 
        modalities: List[str] = ["rgb", "foreground_mask", "patch_metadata"],
    ):
        self.root = Path(root)
        self.ann_file = Path(annFile)

        self.images = {}
        for modality in modalities:
            self.images[modality] = self.root / modality
            assert self.images[modality].exists()

        # look for RGB or Depth images to load
        self.image_path = None
        if "rgb" in self.images:
            self.image_path = self.images["rgb"]
        elif "depth" in self.images:
            self.image_path = self.images["depth"]
                
        super().__init__(root=self.image_path, annFile=self.ann_file)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, y = super().__getitem__(index)
        id = self.ids[index]
        item_path = Path(self.coco.loadImgs(id)[0]["file_name"])

        # Set depth perturbation bound based on split
        # all images that starts with 1#######.png have patch located off sidewalk/street
        if item_path.stem[0] == "1":
            max_depth_perturb_meters = 3.0
        else:
            max_depth_perturb_meters = 0.03

        mask = Image.open(self.images["foreground_mask"] / item_path.name)

        patch_metadata = {
            "gs_coords": np.load(
                self.images["patch_metadata"] / (item_path.stem + "_coords.npy")
            ),
            "avg_patch_depth": np.load(
                self.images["patch_metadata"] / (item_path.stem + "_avg_depth.npy")
            ),
            "mask": np.array(mask),
            "max_depth_perturb_meters": max_depth_perturb_meters,
        }
        y = (y, patch_metadata)

        return (x, y)
