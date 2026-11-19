"""Download the publicly accessible Titanic copy and verify the exact data bytes."""
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
            raise ValueError(f'Source changed: expected {EXPECTED_SHA256}, got {actual}')
        target.write_bytes(payload)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f'Data checksum mismatch: {actual}')
    print(f'Verified {target.name}: SHA256={actual}')

if __name__ == '__main__':
    main()
