"""학번·이름은 로컬 설정에서 읽고, 이 정보로 보고서 파일명을 만든다."""
import json
from pathlib import Path

def load_submission(root):
    root=Path(root)
    path=root/'submission.json'
    if not path.exists():path=root/'submission.example.json'
    return json.loads(path.read_text(encoding='utf-8'))

def report_path(root,submission):
    student_id=submission.get('student_id') or 'student'
    student_name=submission.get('student_name') or 'report'
    # 운영체제에서 파일명으로 사용할 수 없는 문자는 밑줄로 바꾼다.
    for character in '<>:"/\\|?*':
        student_id=student_id.replace(character,'_')
        student_name=student_name.replace(character,'_')
    return Path(root)/'output/pdf'/f'{student_id}_{student_name}_Titanic_특성공학_보고서.pdf'
