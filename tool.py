
import platform


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


def load_audio(file, sr):
    import os
    import traceback
    import ffmpeg
    import numpy as np
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str):
    if platform.system() == 'Windows':
        path_str = path_str.replace('/', '\\')
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def cut(inp):
    def split(text):
        splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
        text = text.replace("……", "。").replace("——", "，")
        if text[-1] not in splits:
            text += "。"
        start = end = 0
        len_text = len(text)
        res = []
        while 1:
            if start >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if text[start] in splits:
                start += 1
                res.append(text[end:start])
                end = start
            else:
                start += 1
        return res
    inp = inp.replace("\n", "")
    res = []
    re = ""
    i = 0
    for v in split(inp):
        ret = re+v
        i += 1
        if len(ret) > 50:
            res.append(re)
            i = 0
            re = v
        elif i % 4 == 0:
            res.append(ret)
            i = 0
            re = ""
        else:
            re = ret

    if len(res) == 0:
        res = [re]
    elif v not in res[-1]:
        res.append(v)
    return res


if __name__ == "__main__":
    ss = """重视语言表达的理论与感性兼具，平衡专业知识与公共关怀；完善逻辑推理，严谨论证过程；关注论文的六个隐性要素，从选题到修改环节都要注意这些指标。此外，重视论文形式要素，形式可以倒逼写作，遵循学术论文形式上的规律性。
重视语言表达的理论与感性兼具，平衡专业知识与公共关怀；完善逻辑推理"""
    re = cut(ss)
    for i in re:
        print(i)
