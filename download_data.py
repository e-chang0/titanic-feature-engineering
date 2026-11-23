"""타이타닉 공개 사본을 내려받고 원본과 같은 파일인지 해시로 확인한다."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
EXPECTED_SHA256 = '4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7'

def main():
    target = ROOT / 'data/train.csv'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        request = urllib.request.Request(URL, headers={'User-Agent': 'Titanic-Coursework/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != EXPECTED_SHA256:
            raise ValueError(f'다운로드한 파일이 기준 원본과 다릅니다. 예상 해시: {EXPECTED_SHA256}, 실제: {actual}')
        target.write_bytes(payload)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f'데이터 파일의 해시가 일치하지 않습니다: {actual}')
    print(f'{target.name} 원본 확인 완료: SHA256={actual}')

if __name__ == '__main__':
    main()
