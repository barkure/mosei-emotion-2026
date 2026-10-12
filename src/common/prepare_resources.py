# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Download pinned public foundation assets, or check existing files with --check-only."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
BERT_REV = '86b5e0934494bd15c9632b12f734a8a67f723594'
WAV_REV = '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
FACE_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    metadata = json.loads((ROOT/'docs/预训练资源校验.json').read_text(encoding="utf-8"))
    if not args.check_only:
        for item in metadata['assets']:
            target = ROOT/item['path']
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
                continue
            name = target.name
            if 'bert-base-uncased/' in item['path']:
                url = f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{BERT_REV}/{name}'
            elif 'wav2vec2-base-960h/' in item['path']:
                url = f'https://huggingface.co/facebook/wav2vec2-base-960h/resolve/{WAV_REV}/{name}'
            elif name == 'face_landmarker.task':
                url = FACE_URL
            else:
                raise ValueError(f"Unknown resource: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix+'.download')
            print(f'Downloading {item["path"]}', flush=True)
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as stream:
                        shutil.copyfileobj(response, stream, length=1024*1024)
                    break
                except (OSError, TimeoutError):
                    temp.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(2)
            if hashlib.sha256(temp.read_bytes()).hexdigest() != item['sha256']:
                temp.unlink()
                raise ValueError(f'Checksum mismatch: {target}')
            temp.replace(target)
    metadata = json.loads((ROOT/'docs/预训练资源校验.json').read_text(encoding="utf-8"))
    for item in metadata['assets']:
        p = ROOT/item['path']
        assert p.is_file(), p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], p
    wav = ROOT/'models/wav2vec2-base-960h'
    for name in ['config.json','preprocessor_config.json','vocab.json']:
        assert (wav/name).is_file(), name
    assert any((wav/n).is_file() for n in ['pytorch_model.bin','model.safetensors'])
    print('Pinned public resources verified.')


if __name__ == '__main__':
    main()
