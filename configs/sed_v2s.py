import copy
from configs.common import common_cfg
from modules.augmentations import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
)
from audiomentations import Compose as amCompose
from audiomentations import OneOf as amOneOf
from audiomentations import AddBackgroundNoise, Gain, GainTransition, TimeStretch
import numpy as np

cfg = copy.deepcopy(common_cfg)
if cfg.WANDB_API_KEY=='your key':
    print('input your wandb api key!')
    raise NotImplementedError

cfg.model_type = "sed"
cfg.model_name = "tf_efficientnetv2_s_in21k"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5


cfg.batch_size = 24
cfg.PRECISION = 32
cfg.seed = {
    "soft_loss": 20231121,
}
cfg.DURATION_TRAIN = 10
cfg.DURATION_FINETUNE = 30
cfg.freeze = False
cfg.mixup = True
cfg.mixup2 = True
cfg.mixup_prob = 0.7
cfg.mixup_double = 0.5
cfg.mixup2_prob = 0.15
cfg.mix_beta = 5
cfg.mix_beta2 = 2
cfg.in_chans = 3
cfg.epochs = {
    "soft_loss": 70,
}
cfg.lr = {
    "soft_loss": 3e-4,
}

cfg.model_ckpt = {
    "soft_loss": None,
}

cfg.output_path = {
    "soft_loss": "outputs/sed_v2s/pytorch/soft_loss",
}

cfg.final_model_path = "outputs/sed_v2s/pytorch/soft_loss/last.ckpt"
cfg.onnx_path = "outputs/sed_v2s/onnx"
cfg.openvino_path = "outputs/sed_v2s/openvino"

cfg.loss = {
    "soft_loss": "soft_auc",
}

cfg.img_size = 384
cfg.n_mels = 128
cfg.n_fft = 2048
cfg.f_min = 0
cfg.f_max = 16000

cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)
cfg.hop_length = cfg.infer_duration * cfg.SR // (cfg.img_size - 1)

cfg.normal = 80

cfg.tta_delta = 3

cfg.am_audio_transforms = amCompose(
    [
        # sed
        AddBackgroundNoise(
            cfg.birdclef2021_nocall + cfg.birdclef2020_nocall,
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.6,
            lru_cache_size=cfg.background_noise_cache_size,
        ),
        AddBackgroundNoise(
            cfg.freefield + cfg.warblrb + cfg.birdvox,
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.3,
            lru_cache_size=cfg.background_noise_cache_size,
        ),
        AddBackgroundNoise(
            cfg.rainforest + cfg.environment,
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.4,
            lru_cache_size=cfg.background_noise_cache_size,
        ),
        amOneOf(
            [
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
            ],
        ),
    ]
)


cfg.np_audio_transforms = CustomCompose(
    [
        CustomOneOf(
            [
                NoiseInjection(p=1, max_noise_level=0.04),
                GaussianNoise(p=1, min_snr=5, max_snr=20),
                PinkNoise(p=1, min_snr=5, max_snr=20),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
            ],
            p=0.3,
        ),
    ]
)

cfg.input_shape = (120,cfg.in_chans,cfg.n_mels,768)
cfg.input_names = [ "x",'tta_delta' ]
cfg.output_names = [ "y" ]
cfg.opset_version = None

basic_cfg = cfg
