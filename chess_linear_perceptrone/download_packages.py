#!/usr/bin/env python3
"""
인터넷이 연결된 컴퓨터에서 패키지를 다운로드하는 스크립트

사용법:
1. 이 스크립트를 실행하여 packages/ 폴더에 모든 패키지 다운로드
2. packages/ 폴더를 USB나 외장하드로 복사
3. 오프라인 컴퓨터로 이동
4. 오프라인 컴퓨터에서 직접 wheel 파일들을 복사하여 설치
"""

import os
import subprocess
import sys
import shutil

def download_packages():
    """패키지 다운로드"""
    print("패키지 다운로드 시작!")
    print("=" * 50)
    
    # packages 폴더 생성
    packages_dir = "packages"
    if os.path.exists(packages_dir):
        shutil.rmtree(packages_dir)
    os.makedirs(packages_dir)
    
    # requirements_offline.txt에서 패키지 목록 읽기
    packages = []
    try:
        with open("requirements_offline.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    packages.append(line)
    except FileNotFoundError:
        print("requirements_offline.txt 파일을 찾을 수 없습니다.")
        return False
    
    print(f"다운로드할 패키지: {len(packages)}개")
    print()
    
    # 각 패키지 다운로드
    for i, package in enumerate(packages, 1):
        print(f"[{i}/{len(packages)}] {package} 다운로드 중...")
        try:
            # wheel 파일과 의존성 모두 다운로드
            result = subprocess.run([
                sys.executable, "-m", "pip", "download",
                "--dest", packages_dir,
                "--only-binary=:all:",
                package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  {package} 다운로드 완료")
            else:
                print(f"  {package} 다운로드 실패 (의존성 문제일 수 있음)")
                print(f"    오류: {result.stderr}")
                
        except Exception as e:
            print(f"  {package} 다운로드 중 오류: {e}")
    
    print()
    print("=" * 50)
    print("패키지 다운로드 완료!")
    
    # 다운로드된 파일 목록 표시
    files = os.listdir(packages_dir)
    print(f"packages/ 폴더에 {len(files)}개 파일이 다운로드되었습니다.")
    
    # 폴더 크기 계산
    total_size = 0
    for root, dirs, files in os.walk(packages_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    print(f"총 용량: {total_size / (1024**3):.2f} GB")
    
    return True

def create_offline_install_script():
    """오프라인 설치 스크립트 생성"""
    print("\n오프라인 설치 스크립트 생성 중...")
    
    # Windows용 배치 파일
    with open("install_offline_no_pip.bat", "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("echo ========================================\n")
        f.write("echo 체스 AI - pip 없이 오프라인 설치\n")
        f.write("echo ========================================\n")
        f.write("echo.\n")
        f.write("echo packages/ 폴더가 있는지 확인 중...\n")
        f.write("if not exist \"packages\\\" (\n")
        f.write("    echo packages/ 폴더를 찾을 수 없습니다!\n")
        f.write("    pause\n")
        f.write("    exit /b 1\n")
        f.write(")\n")
        f.write("echo packages/ 폴더를 찾았습니다!\n")
        f.write("echo.\n")
        f.write("echo 설치 방법:\n")
        f.write("echo 1. packages/ 폴더의 wheel 파일들을 Python 설치 경로에 복사\n")
        f.write("echo 2. 또는 가상환경을 만들어서 거기에 복사\n")
        f.write("echo.\n")
        f.write("echo 자세한 내용은 README_OFFLINE_NO_PIP.md를 참고하세요.\n")
        f.write("pause\n")
    
    print("install_offline_no_pip.bat 생성 완료!")
    
    # Linux/Mac용 쉘 스크립트
    with open("install_offline_no_pip.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("echo '========================================'\n")
        f.write("echo '체스 AI - pip 없이 오프라인 설치'\n")
        f.write("echo '========================================'\n")
        f.write("echo\n")
        f.write("echo 'packages/ 폴더가 있는지 확인 중...'\n")
        f.write("if [ ! -d \"packages\" ]; then\n")
        f.write("    echo 'packages/ 폴더를 찾을 수 없습니다!'\n")
        f.write("    exit 1\n")
        f.write("fi\n")
        f.write("echo 'packages/ 폴더를 찾았습니다!'\n")
        f.write("echo\n")
        f.write("echo '설치 방법:'\n")
        f.write("echo '1. packages/ 폴더의 wheel 파일들을 Python 설치 경로에 복사'\n")
        f.write("echo '2. 또는 가상환경을 만들어서 거기에 복사'\n")
        f.write("echo\n")
        f.write("echo '자세한 내용은 README_OFFLINE_NO_PIP.md를 참고하세요.'\n")
    
    # 실행 권한 부여
    os.chmod("install_offline_no_pip.sh", 0o755)
    print("install_offline_no_pip.sh 생성 완료!")

def create_detailed_guide():
    """상세한 오프라인 설치 가이드 생성"""
    print("\n상세한 오프라인 설치 가이드 생성 중...")
    
    guide_content = """# 체스 AI - pip 없이 오프라인 설치 가이드

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

# 보통 C:\\Users\\[사용자명]\\AppData\\Local\\Programs\\Python\\Python3x\\Lib\\site-packages\\
# 또는 C:\\Python3x\\Lib\\site-packages\\

# packages/ 폴더의 모든 .whl 파일을 site-packages 폴더에 복사
copy packages\\*.whl "C:\\Users\\[사용자명]\\AppData\\Local\\Programs\\Python\\Python3x\\Lib\\site-packages\\"
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
# Windows: chess_env\\Scripts\\activate
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
python -c "import sys; print('\\n'.join(sys.path))"

# site-packages 경로에 .whl 파일이 있는지 확인
# Windows: dir "C:\\Python3x\\Lib\\site-packages\\*.whl"
# Linux/Mac: ls /usr/local/lib/python3.x/site-packages/*.whl
```

### 버전 충돌 시
```bash
# 기존 패키지 제거 (가능한 경우)
# Windows: del "C:\\Python3x\\Lib\\site-packages\\[패키지명]*"
# Linux/Mac: rm /usr/local/lib/python3.x/site-packages/[패키지명]*
```
"""
    
    with open("README_OFFLINE_NO_PIP.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print("README_OFFLINE_NO_PIP.md 생성 완료!")

def main():
    print("체스 AI 패키지 다운로드 및 오프라인 설치 가이드 생성")
    print("=" * 60)
    
    # 1. 패키지 다운로드
    if not download_packages():
        print("패키지 다운로드에 실패했습니다.")
        return
    
    # 2. 오프라인 설치 스크립트 생성
    create_offline_install_script()
    
    # 3. 상세한 가이드 생성
    create_detailed_guide()
    
    print("\n" + "=" * 60)
    print("모든 준비 완료!")
    print("\n다음 단계:")
    print("1. packages/ 폴더를 USB나 외장하드로 복사")
    print("2. 오프라인 컴퓨터로 이동")
    print("3. README_OFFLINE_NO_PIP.md를 참고하여 설치")
    print("\npip를 사용할 수 없는 경우 wheel 파일을 직접 복사하여 설치하세요.")

if __name__ == "__main__":
    main() 