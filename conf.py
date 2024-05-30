import sys
import torch

is_half = True
if torch.cuda.is_available():
    infer_device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half = False
elif torch.backends.mps.is_available():
    infer_device = "mps"
else:
    infer_device = "cpu"
    is_half = False


class Config:
    def __init__(self):
        # your trained data
        self.sovits_path = "pth"
        self.gpt_path = "ckpt"
        self.wav = "wav"
        self.promot = "我们的像比如说我们的训练营课程一样啊，他每一节课都是非常简凑的啊，并不是那个样子的啊"
        self.language = "zh"
        self.is_half = is_half

        # self.cnhubert_path = "models/cnhubert"
        # self.bert_path = "models/bert"
        # self.pretrained_sovits_path = "models/sobits"
        # self.pretrained_gpt_path = "models/gpt"
        self.cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        self.pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

        self.exp_root = "logs"
        self.python_exec = sys.executable or "python"
        self.device = infer_device
        self.api_port = 9880


g = Config()
