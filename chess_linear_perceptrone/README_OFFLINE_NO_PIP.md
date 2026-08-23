# 체스 AI - pip 없이 오프라인 설치 가이드

## 상황 설명

이 가이드는 **인터넷이 전혀 연결되지 않는 컴퓨터**에서 pip 명령어를 사용할 수 없을 때의 설치 방법입니다.

## 1단계: 인터넷이 연결된 컴퓨터에서

### 패키지 다운로드
```bash
# 이 스크립트 실행
python download_packages.py

# 또는 수동으로 다운로드
pip download torch --dest packages/
pip download python-chess --dest packages/
pip download numpy --dest packages/
pip download PyYAML --dest packages/
pip download matplotlib --dest packages/
pip download pandas --dest packages/
```

### packages/ 폴더 백업
```bash
# Windows
xcopy packages packages_backup /E /I /H

# Linux/Mac
cp -r packages packages_backup
```

## 2단계: 오프라인 컴퓨터에서 (pip 사용 불가)

### 방법 1: Python 설치 경로에 직접 복사 (권장)

#### Windows의 경우:
```bash
# Python 설치 경로 확인
python -c "import sys; print(sys.prefix)"

# 보통 C:\Users\[사용자명]\AppData\Local\Programs\Python\Python3x\Lib\site-packages\
# 또는 C:\Python3x\Lib\site-packages\

# packages/ 폴더의 모든 .whl 파일을 site-packages 폴더에 복사
copy packages\*.whl "C:\Users\[사용자명]\AppData\Local\Programs\Python\Python3x\Lib\site-packages\"
```

#### Linux/Mac의 경우:
```bash
# Python 설치 경로 확인
python3 -c "import sys; print(sys.prefix)"

# 보통 /usr/local/lib/python3.x/site-packages/
# 또는 /home/[사용자명]/.local/lib/python3.x/site-packages/

# packages/ 폴더의 모든 .whl 파일을 site-packages 폴더에 복사
cp packages/*.whl /usr/local/lib/python3.x/site-packages/
```

### 방법 2: 가상환경 생성 후 복사

```bash
# 가상환경 생성
python -m venv chess_env

# 가상환경 활성화
# Windows: chess_env\Scripts\activate
# Linux/Mac: source chess_env/bin/activate

# site-packages 경로 확인
python -c "import sys; print(sys.path)"

# packages/ 폴더의 .whl 파일들을 가상환경의 site-packages에 복사
```

## 3단계: 설치 확인

```bash
# Python에서 패키지 import 테스트
python -c "
import torch
import chess
import numpy
import yaml
import matplotlib
import pandas
print('모든 패키지가 정상적으로 설치되었습니다!')
"
```

## 4단계: 학습 실행

```bash
# 은닉층 4개 vs 8개 동시 학습
python multilayer/run_training.py

# 개별 학습
python multilayer/run_training.py --hidden4
python multilayer/run_training.py --hidden8
```

## 주의사항

- **파일 복사**: .whl 파일들을 정확한 site-packages 경로에 복사해야 함
- **권한 문제**: Linux/Mac에서 권한 오류 시 sudo 사용
- **경로 확인**: Python 설치 경로를 정확히 확인하고 복사
- **의존성**: 일부 패키지가 누락될 수 있으므로 import 테스트 필수

## 문제 해결

### 패키지를 찾을 수 없는 경우
```bash
# Python 경로 확인
python -c "import sys; print('\n'.join(sys.path))"

# site-packages 경로에 .whl 파일이 있는지 확인
# Windows: dir "C:\Python3x\Lib\site-packages\*.whl"
# Linux/Mac: ls /usr/local/lib/python3.x/site-packages/*.whl
```

### 버전 충돌 시
```bash
# 기존 패키지 제거 (가능한 경우)
# Windows: del "C:\Python3x\Lib\site-packages\[패키지명]*"
# Linux/Mac: rm /usr/local/lib/python3.x/site-packages/[패키지명]*
```
