from .autoencoder_magvit import (AutoencoderKL, AutoencoderKLCogVideoX,
                                 AutoencoderKLMagvit)
from .transformer3d import (EasyAnimateTransformer3DModel,
                            HunyuanTransformer3DModel, Transformer3DModel)

name_to_transformer3d = {
    "Transformer3DModel": Transformer3DModel,
    "HunyuanTransformer3DModel": HunyuanTransformer3DModel,
    "EasyAnimateTransformer3DModel": EasyAnimateTransformer3DModel,
}
name_to_autoencoder_magvit = {
    "AutoencoderKL": AutoencoderKL,
    "AutoencoderKLMagvit": AutoencoderKLMagvit,
    "AutoencoderKLCogVideoX": AutoencoderKLCogVideoX,
}