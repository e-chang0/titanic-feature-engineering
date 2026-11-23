"""운영체제에 설치된 한글 글꼴을 그래프에 사용한다."""
from matplotlib import font_manager, rcParams

def korean_font():
    installed={font.name for font in font_manager.fontManager.ttflist}
    for name in ['Malgun Gothic','NanumGothic','Noto Sans CJK KR','AppleGothic']:
        if name in installed:
            rcParams['axes.unicode_minus']=False
            return name
    # 한글 글꼴이 없으면 실행은 가능하지만, 글자가 깨질 수 있다.
    return 'DejaVu Sans'
