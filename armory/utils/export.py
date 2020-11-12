import math
import os
import logging
import coloredlogs
import numpy as np
import ffmpeg
import pickle
from PIL import Image
from shutil import rmtree
from scipy.io import wavfile


logger = logging.getLogger(__name__)


class SampleExporter():
    def __init__(self, base_output_dir, domain, num_samples=math.inf):
        self.base_output_dir = base_output_dir
        self.domain = domain
        self.num_samples = num_samples
        self.saved_samples = 0
        self.output_dir = None
        self.y_dict = {}

        assert self.domain in ("image", "so2sat", "audio", "video")
        self._make_output_dir()
    
    def export(self, x, x_adv, y, y_adv):

        if self.saved_samples < self.num_samples:

            self.y_dict[self.saved_samples] = {
                "ground truth": y,
                "predicted": y_adv
            }

            if self.domain == "image":
                self._export_images(x, x_adv)
            elif self.domain == "so2sat":
                self._export_so2sat(x, x_adv)
            elif self.domain == "audio":
                self._export_audio(x, x_adv)
            elif self.domain == "video":
                self._export_video(x, x_adv)
            else:
                raise ValueError(f"Expected domain in (\"image\", \"audio\", \"video\"), found {self.domain}")
    
            if self.saved_samples == self.num_samples:
                with open(os.path.join(self.output_dir, "predictions.pkl"), "wb") as f:
                    pickle.dump(self.y_dict, f)
            
    def _make_output_dir(self):
        assert os.path.exists(self.base_output_dir) and os.path.isdir(self.base_output_dir), f"Directory {self.base_output_dir} does not exist"
        assert os.access(self.base_output_dir, os.W_OK), f"Directory {self.base_output_dir} is not writable"
        self.output_dir = os.path.join(self.base_output_dir, "saved_samples")
        if os.path.exists(self.output_dir):
            logger.warning(f"Sample output directory {self.output_dir} already exists. Removing")
            rmtree(self.output_dir)
        os.mkdir(self.output_dir)

    def _export_images(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(x_i.shape == x_adv_i.shape), f"Benign and adversarial images are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            assert x_i.min() >= 0. and x_i.max() <= 1., "Benign image out of range, should be in [0., 1.]"
            assert x_adv_i.min() >= 0. and x_adv_i.max() <= 1., "Adversarial image out of range, should be in [0., 1.]"

            benign_image = Image.fromarray(np.uint8(x_i * 255.), 'RGB')
            adversarial_image = Image.fromarray(np.uint8(x_adv_i * 255.), 'RGB')
            benign_image.save(os.path.join(self.output_dir, f"{self.saved_samples}_benign.png"))
            adversarial_image.save(os.path.join(self.output_dir, f"{self.saved_samples}_adversarial.png"))

            self.saved_samples += 1
    
    def _export_so2sat(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):
            if self.saved_samples == self.num_samples:
                break

            assert np.all(x_i.shape == x_adv_i.shape), f"Benign and adversarial images are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            assert x_i[...,:4].min() >= -1. and x_i[...,:4].max() <= 1., "Benign SAR images out of range, should be in [-1., 1.]"
            assert x_adv_i[...,:4].min() >= -1. and x_adv_i[...,:4].max() <= 1., "Adversarial SAR images out of range, should be in [-1., 1.]"
            assert x_i[...,4:].min() >= 0. and x_i[...,4:].max() <= 1., "Benign EO images out of range, should be in [0., 1.]"
            assert x_adv_i[...,4:].min() >= 0. and x_adv_i[...,4:].max() <= 1., "Adversarial EO images out of range, should be in [0., 1.]"

            folder = str(self.saved_samples)
            os.mkdir(os.path.join(self.output_dir, folder))

            x_vh = np.log10(np.abs(np.complex128(x_i[...,0] + 1j * x_i[...,1]) + (1e-9 + 1e-9j)))
            x_vv = np.log10(np.abs(np.complex128(x_i[...,2] + 1j * x_i[...,3]) + (1e-9 + 1e-9j)))
            x_adv_vh = np.log10(np.abs(np.complex128(x_adv_i[...,0] + 1j * x_adv_i[...,1]) + (1e-9 + 1e-9j)))
            x_adv_vv = np.log10(np.abs(np.complex128(x_adv_i[...,2] + 1j * x_adv_i[...,3]) + (1e-9 + 1e-9j)))
            sar_min = np.min((x_vh.min(), x_vv.min(), x_adv_vh.min(), x_adv_vv.min()))
            sar_max = np.max((x_vh.max(), x_vv.max(), x_adv_vh.max(), x_adv_vv.max()))

            benign_vh = Image.fromarray(np.uint8(255. / (sar_max - sar_min) * (x_vh - sar_min)), 'L')
            benign_vv = Image.fromarray(np.uint8(255. / (sar_max - sar_min) * (x_vv - sar_min)), 'L')
            adversarial_vh = Image.fromarray(np.uint8(255. / (sar_max - sar_min) * (x_adv_vh - sar_min)), 'L')
            adversarial_vv = Image.fromarray(np.uint8(255. / (sar_max - sar_min) * (x_adv_vv - sar_min)), 'L')
            benign_vh.save(os.path.join(self.output_dir, folder, "vh_benign.png"))
            benign_vv.save(os.path.join(self.output_dir, folder, "vv_benign.png"))
            adversarial_vh.save(os.path.join(self.output_dir, folder, "vh_adversarial.png"))
            adversarial_vv.save(os.path.join(self.output_dir, folder, "vv_adversarial.png"))
        
            eo_min = np.min((x_i[..., 4:].min(), x_adv[..., 4:].min()))
            eo_max = np.max((x_i[..., 4:].max(), x_adv[..., 4:].max()))
            for c in range(4, 14):
                benign_eo = Image.fromarray(np.uint8(255. / (eo_max - eo_min) * (x_i[..., c] - eo_min)), 'L')
                adversarial_eo = Image.fromarray(np.uint8(255. / (eo_max - eo_min) * (x_adv_i[..., c] - eo_min)), 'L')
                benign_eo.save(os.path.join(self.output_dir, folder, f"eo{c-4}_benign.png"))
                adversarial_eo.save(os.path.join(self.output_dir, folder, f"eo{c-4}_adversarial.png"))

            self.saved_samples += 1

    def _export_audio(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(x_i.shape == x_adv_i.shape), f"Benign and adversarial audio are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            assert x_i.min() >= -1. and x_i.max() <= 1., "Benign audio out of range, should be in [-1., 1.]"
            assert x_adv_i.min() >= -1. and x_adv_i.max() <= 1., "Adversarial audio out of range, should be in [-1., 1.]"

            wavfile.write(os.path.join(self.output_dir, f"{self.saved_samples}_benign.wav") , rate=16000, data=x_i)
            wavfile.write(os.path.join(self.output_dir, f"{self.saved_samples}_adversarial.wav") , rate=16000, data=x_adv_i)

            self.saved_samples += 1

    def _export_video(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(x_i.shape == x_adv_i.shape), f"Benign and adversarial videos are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            assert x_i.min() >= 0. and x_i.max() <= 1., "Benign video out of range, should be in [0., 1.]"
            assert x_adv_i.min() >= 0. and x_adv_i.max() <= 1., "Adversarial video out of range, should be in [0., 1.]"

            benign_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{x_i.shape[2]}x{x_i.shape[1]}")
                .output(os.path.join(self.output_dir, f"{self.saved_samples}_benign.mp4"), pix_fmt='yuv420p', vcodec="libx264", r=24)
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )

            for x_frame in x_i:
                benign_process.stdin.write(
                    (x_frame * 255.)
                    .astype(np.uint8)
                    .tobytes()
                )

            benign_process.stdin.close()
            benign_process.wait()
            
            adversarial_process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{x_i.shape[2]}x{x_i.shape[1]}")
                .output(os.path.join(self.output_dir, f"{self.saved_samples}_adversarial.mp4"), pix_fmt='yuv420p', vcodec="libx264", r=24)
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )

            for x_frame in x_adv_i:
                adversarial_process.stdin.write(
                    (x_frame * 255.)
                    .astype(np.uint8)
                    .tobytes()
                )

            adversarial_process.stdin.close()
            adversarial_process.wait()
            self.saved_samples += 1