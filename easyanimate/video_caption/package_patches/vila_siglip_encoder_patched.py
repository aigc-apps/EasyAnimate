# Modified from https://github.com/NVlabs/VILA/blob/1c88211/llava/model/multimodal_encoder/siglip_encoder.py
# 1. Support transformers >= 4.36.2.
import torch
import transformers
from packaging import version
from transformers import AutoConfig, AutoModel, PretrainedConfig

from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2

if version.parse(transformers.__version__) > version.parse("4.36.2"):
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
else:
    from .siglip import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            # TODO(ligeng): why pass config here leading to errors?
            model_name_or_path, torch_dtype=eval(config.model_dtype), state_dict=state_dict
        )
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )

        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size['height'] = self.image_processor.size['width'] = self.scales[-1]

        self.is_loaded = True

if version.parse(transformers.__version__) <= version.parse("4.36.2"):
    AutoConfig.register("siglip_vision_model", SiglipVisionConfig)
    AutoModel.register(SiglipVisionConfig, SiglipVisionModel)
