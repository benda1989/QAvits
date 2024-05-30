# new
from query import query_engine
from htmlTemplate import html
from conf import g
from tool import load_audio, cut, DictToAttrRecursive
# api
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket
# model
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.transformers import AutoModelForMaskedLM, AutoTokenizer
# other
import librosa
import soundfile
import numpy as np
from time import time
from io import BytesIO
import torch
import signal
import os
import sys
import copy


tokenizer = AutoTokenizer.from_pretrained(g.bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(g.bert_path)
if g.is_half:
    bert_model = bert_model.half().to(g.device)
else:
    bert_model = bert_model.to(g.device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(g.device)  # 输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


dict_s2 = torch.load(g.sovits_path, map_location="cpu")
hps = DictToAttrRecursive(dict_s2["config"])
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(g.gpt_path, map_location="cpu")
config = dict_s1["config"]
cnhubert.cnhubert_base_path = g.cnhubert_path
ssl_model = cnhubert.get_model()
if g.is_half:
    ssl_model = ssl_model.half().to(g.device)
else:
    ssl_model = ssl_model.to(g.device)

vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
if g.is_half:
    vq_model = vq_model.half().to(g.device)
else:
    vq_model = vq_model.to(g.device)
vq_model.eval()
print(vq_model.load_state_dict(dict_s2["weight"], strict=False))

max_sec = config['data']['max_sec']
t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if g.is_half:
    t2s_model = t2s_model.half().to(g.device)
else:
    t2s_model = t2s_model.to(g.device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio_norm = torch.FloatTensor(audio).unsqueeze(0)
    return spectrogram_torch(audio_norm,
                             hps.data.filter_length,
                             hps.data.sampling_rate,
                             hps.data.hop_length,
                             hps.data.win_length,
                             center=False)


class Model():
    def __init__(self) -> None:
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if g.is_half == True else np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(g.wav, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if (g.is_half == True):
                wav16k = wav16k.half().to(g.device)
                zero_wav_torch = zero_wav_torch.half().to(g.device)
            else:
                wav16k = wav16k.to(g.device)
                zero_wav_torch = zero_wav_torch.to(g.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            self.codes = vq_model.extract_latent(ssl_content)

        phones, self.word2ph, self.norm_text = clean_text(g.promot, g.language)
        self.phones = cleaned_text_to_sequence(phones)
        self.wav = zero_wav

    def get(self):
        return copy.deepcopy(self.phones), copy.deepcopy(self.word2ph), copy.deepcopy(self.norm_text), copy.deepcopy(self.codes), copy.deepcopy(self.wav)


model = Model()


def get_tts_wav(text, text_language="zh"):
    t0 = time()
    phones1, word2ph1, norm_text1, codes, zero_wav = model.get()
    prompt_semantic = codes[0, 0]
    for text in cut(text):
        print(text)
        phones2, word2ph2, norm_text2 = clean_text("，"+text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        bert1 = get_bert_feature(norm_text1, word2ph1).to(g.device)
        if (text_language == "zh"):
            bert2 = get_bert_feature(norm_text2, word2ph2).to(g.device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(g.device).unsqueeze(0)
        bert = bert.to(g.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(g.device)
        prompt = prompt_semantic.unsqueeze(0).to(g.device)
        t2 = time()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config['inference']['top_k'],
                early_stop_num=50 * max_sec)
        t3 = time()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, g.wav)  # .to(g.device)
        if (g.is_half == True):
            refer = refer.half().to(g.device)
        else:
            refer = refer.to(g.device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = \
            vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(g.device).unsqueeze(0),
                            refer).detach().cpu().numpy()[
                0, 0]  # 试试重建不带上prompt部分
        t4 = time()
        print("%.3f\t%.3f\t%.3f" % (t2 - t0, t3 - t2, t4 - t3))
        wav = BytesIO()
        soundfile.write(wav, (np.concatenate([audio, zero_wav], 0) * 32768).astype(np.int16), hps.data.sampling_rate, format="wav")
        yield wav.getvalue()


app = FastAPI()


@app.get("/control")
async def control(command: str = None):
    if command == "restart":
        os.execl(g.python_exec, g.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


@app.get("/", response_class=HTMLResponse)
async def index():
    return html


@app.websocket("/ws")
async def handle_ws(client: WebSocket):
    await client.accept()
    while True:
        data = await client.receive_text()
        res = query_engine.query(data)
        with torch.no_grad():
            wav = next(get_tts_wav(res, g.language))
            await client.send_bytes(wav)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=g.api_port, workers=1)
