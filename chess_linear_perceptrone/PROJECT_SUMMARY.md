# 체스 AI 다층 퍼셉트론 프로젝트 요약

## 🎯 프로젝트 개요
체스 게임을 학습하는 인공지능 에이전트를 다층 퍼셉트론(MLP)으로 구현한 강화학습 프로젝트입니다.

## 🏗️ 프로젝트 구조

```
chess_linear_perceptrone/
├── agents/                    # 단층 퍼셉트론 에이전트
│   └── perceptron_agent.py
├── multilayer/               # 다층 퍼셉트론 관련 파일들
│   ├── multilayer_agent.py  # 6개 층 기본 에이전트
│   ├── config_multilayer.yaml # 6개 층 설정
│   ├── main_multilayer.py   # 6개 층 학습 스크립트
│   ├── run_training.py      # 학습 실행 인터페이스
│   ├── train_both_versions.py # 3개 버전 동시 학습
│   ├── hidden4/             # 4개 층 버전
│   │   ├── multilayer_agent_hidden4.py
│   │   ├── config_hidden4.yaml
│   │   └── main_hidden4.py
│   └── hidden8/             # 8개 층 버전
│       ├── multilayer_agent_hidden8.py
│       ├── config_hidden8.yaml
│       └── main_hidden8.py
├── envs/                     # 체스 환경
│   └── chess_env.py
├── utils/                    # 유틸리티 함수들
│   ├── encoding.py          # 체스 상태 인코딩
│   ├── metrics.py           # 성능 지표 계산
│   └── visualize.py         # 그래프 시각화
├── saved_models/            # 학습된 모델 저장
│   ├── single/              # 단층 퍼셉트론 모델
│   └── multilayer/          # 다층 퍼셉트론 모델
│       ├── hidden4/         # 4개 층 모델
│       ├── hidden6/         # 6개 층 모델
│       └── hidden8/         # 8개 층 모델
└── main.py                  # 메인 실행 파일
```

## 🧠 핵심 개념

### 1. 퍼셉트론 구조
- **단층 퍼셉트론**: 입력(768차원) → 은닉층(256차원) → 출력(1차원)
- **다층 퍼셉트론**: 입력(768차원) → [은닉층(256차원)] × N → 출력(1차원)
  - 4개 층 버전: 4개의 은닉층
  - 6개 층 버전: 6개의 은닉층 (기본)
  - 8개 층 버전: 8개의 은닉층

### 2. 강화학습 알고리즘
- **Q-Learning**: 행동-가치 함수를 학습하여 최적 정책 도출
- **Epsilon-Greedy**: 탐험과 활용의 균형을 위한 정책
- **동적 Epsilon Decay**: 학습 진행에 따라 탐험률을 점진적으로 감소

### 3. 체스 상태 표현
- **입력 차원**: 768차원 (체스판의 모든 위치 × 12개 기물 종류)
- **One-Hot 인코딩**: 각 위치에 기물이 있으면 1, 없으면 0

## ⚙️ 주요 설정값

### 학습 하이퍼파라미터
```yaml
learning_rate: 0.001          # 학습률
gamma: 0.99                   # 할인율 (미래 보상 가중치)
epsilon_start: 1.0            # 초기 탐험률
epsilon_end: 0.1              # 최종 탐험률
epsilon_decay_start: 0.998    # 탐험률 감소 시작값
epsilon_decay_end: 1.0        # 탐험률 감소 종료값
num_episodes: 300000          # 총 학습 에피소드 수
max_steps_per_episode: 200    # 에피소드당 최대 스텝 수
hidden_dim: 256               # 은닉층 차원
num_hidden_layers: 6          # 은닉층 개수 (6개 층 기준)
```

## 🚀 사용 방법

### 1. 개별 학습
```bash
# 4개 층만 학습
python multilayer/run_training.py --hidden4

# 6개 층만 학습 (기본)
python multilayer/run_training.py --hidden6
# 또는
python multilayer/main_multilayer.py

# 8개 층만 학습
python multilayer/run_training.py --hidden8
```

### 2. 동시 학습 (3개 버전)
```bash
# 4개, 6개, 8개 층을 동시에 학습
python multilayer/run_training.py
# 또는
python multilayer/train_both_versions.py
```

### 3. 단층 퍼셉트론 학습
```bash
python main.py
```

## 📊 학습 과정

### 1. 에피소드 진행
- 체스 게임 시작 → 에이전트가 행동 선택 → 보상 획득 → 상태 업데이트
- 100 에피소드마다 진행상황 출력
- 10,000 에피소드마다 모델 저장

### 2. 모델 저장 구조
```
saved_models/multilayer/
├── hidden4/
│   ├── version_01/          # 1-10,000 에피소드
│   ├── version_02/          # 10,001-20,000 에피소드
│   └── ...
├── hidden6/
│   ├── version_01/
│   └── ...
└── hidden8/
    ├── version_01/
    └── ...
```

### 3. 성능 지표
- **학습 곡선**: 에피소드별 총 보상 변화
- **승률 곡선**: 에피소드별 승/무/패 비율
- **메트릭 CSV**: 상세한 학습 데이터

## 🔧 핵심 코드 구조

### MultilayerPerceptronAgent 클래스
```python
class MultilayerPerceptronAgent(nn.Module):
    def __init__(self, state_dim=768, hidden_dim=256, num_hidden_layers=6, ...):
        # 동적으로 N개 층 생성
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        # ReLU 활성화 함수 적용 (마지막 층 제외)
        x = state
        for i, layer in enumerate(self.network):
            if i < len(self.network) - 1:
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        return x.squeeze(-1)
```

### 학습 루프
```python
for episode in range(config['num_episodes']):
    state = env.reset()
    for t in range(config['max_steps_per_episode']):
        action = agent.act(state, legal_moves, env, epsilon)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    
    # 10,000 에피소드마다 모델 저장
    if (episode+1) % 10000 == 0:
        version = (episode+1) // 10000
        torch.save(agent.state_dict(), f"model_v{version}.pth")
```

## 📈 성능 분석

### 1. 학습 진행 모니터링
- 에피소드별 총 보상
- Epsilon 값 변화
- 승/무/패 비율

### 2. 모델 비교
- 4개 층 vs 6개 층 vs 8개 층 성능 비교
- 학습 속도와 최종 성능 분석
- 과적합 방지 효과

## 🎮 체스 환경

### ChessEnv 클래스
- **상태**: 768차원 벡터 (체스판 + 기물 정보)
- **행동**: UCI 표준 체스 이동 표기법
- **보상**: 승리(+1), 패배(-1), 무승부(0)
- **종료 조건**: 체크메이트, 스테일메이트, 기물 부족

## 🔄 멀티스레딩 학습

### train_both_versions.py
- 3개 버전을 동시에 학습
- 각 버전별로 별도 스레드 생성
- 학습 완료 후 자동으로 결과 파일 정리

## 📁 파일 관리

### 자동 백업 시스템
- 학습 완료 후 모든 결과 파일을 `saved_models/` 폴더로 이동
- 버전별로 체계적인 폴더 구조 생성
- 중복 파일 방지 및 정리

## 🚨 주의사항

1. **GPU 메모리**: 8개 층 버전은 GPU 메모리를 많이 사용
2. **학습 시간**: 30만 에피소드 완료까지 상당한 시간 소요
3. **저장 공간**: 모델 파일과 그래프가 많이 생성됨
4. **의존성**: PyTorch, python-chess, matplotlib 등 필요

## 🎯 다음 단계

1. **하이퍼파라미터 튜닝**: 학습률, 은닉층 크기 최적화
2. **아키텍처 개선**: 배치 정규화, 드롭아웃 등 추가
3. **성능 평가**: 실제 체스 엔진과의 대전
4. **시각화 개선**: 실시간 학습 진행률 모니터링

---

**프로젝트 실행 순서**:
1. `python multilayer/run_training.py --hidden6` (6개 층 테스트)
2. `python multilayer/run_training.py` (전체 버전 동시 학습)
3. 결과 확인: `saved_models/multilayer/` 폴더 