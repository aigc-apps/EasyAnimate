# Borrowed from https://github.com/discus0434/aesthetic-predictor-v2-5/blob/3125a9e/src/aesthetic_predictor_v2_5/siglip_v2_5.py.
import os
from collections import OrderedDict
from os import PathLike
from typing import Final

import torch
import torch.nn as nn
from transformers import (SiglipImageProcessor, SiglipVisionConfig,
                          SiglipVisionModel, logging)
from transformers.image_processing_utils import BatchFeature
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

logging.set_verbosity_error()

URL: Final[str] = (
    "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/aesthetic_predictor_v2_5.pth"
)


class AestheticPredictorV2_5Head(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.scoring_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.scoring_head(image_embeds)


class AestheticPredictorV2_5Model(SiglipVisionModel):
    PATCH_SIZE = 14

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.layers = AestheticPredictorV2_5Head(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | ImageClassifierOutputWithNoAttention:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = super().forward(
            pixel_values=pixel_values,
            return_dict=return_dict,
        )
        image_embeds = outputs.pooler_output
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        prediction = self.layers(image_embeds_norm)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct()

        if not return_dict:
            return (loss, prediction, image_embeds)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=prediction,
            hidden_states=image_embeds,
        )


class AestheticPredictorV2_5Processor(SiglipImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> BatchFeature:
        return super().__call__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str
        | PathLike = "google/siglip-so400m-patch14-384",
        *args,
        **kwargs,
    ) -> "AestheticPredictorV2_5Processor":
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


def convert_v2_5_from_siglip(
    predictor_name_or_path: str | PathLike | None = None,
    encoder_model_name: str = "google/siglip-so400m-patch14-384",
    *args,
    **kwargs,
) -> tuple[AestheticPredictorV2_5Model, AestheticPredictorV2_5Processor]:
    model = AestheticPredictorV2_5Model.from_pretrained(
        encoder_model_name, *args, **kwargs
    )

    processor = AestheticPredictorV2_5Processor.from_pretrained(
        encoder_model_name, *args, **kwargs
    )

    if predictor_name_or_path is None or not os.path.exists(predictor_name_or_path):
        state_dict = torch.hub.load_state_dict_from_url(URL, map_location="cpu")
    else:
        state_dict = torch.load(predictor_name_or_path, map_location="cpu")

    assert isinstance(state_dict, OrderedDict)

    model.layers.load_state_dict(state_dict)
    model.eval()

    return model, processor