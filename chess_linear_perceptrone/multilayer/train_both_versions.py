import yaml
import numpy as np
import torch
import sys
import os
import shutil
import threading
import time
sys.path.append('..')  # 상위 폴더의 모듈 import를 위해
from envs.chess_env import ChessEnv
from multilayer.hidden4.multilayer_agent_hidden4 import MultilayerPerceptronAgentHidden4
from multilayer.hidden8.multilayer_agent_hidden8 import MultilayerPerceptronAgentHidden8
from multilayer.multilayer_agent import MultilayerPerceptronAgent
from utils.metrics import init_metrics, append_metrics, save_metrics_csv
from utils.visualize import plot_learning_curve, plot_winrate

# 설정 불러오기
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_result_from_reward(reward):
    if reward == 1.0:
        return 'win'
    elif reward == -1.0:
        return 'lose'
    else:
        return 'draw'

def create_version_folders():
    """10000개 단위로 버전 폴더 생성"""
    # 은닉층 4개 버전 폴더
    base_dir_4 = "saved_models/multilayer/hidden4"
    os.makedirs(base_dir_4, exist_ok=True)
    
    # 은닉층 6개 버전 폴더
    base_dir_6 = "saved_models/multilayer/hidden6"
    os.makedirs(base_dir_6, exist_ok=True)
    
    # 은닉층 8개 버전 폴더
    base_dir_8 = "saved_models/multilayer/hidden8"
    os.makedirs(base_dir_8, exist_ok=True)
    
    # 30만 에피소드를 1만 단위로 나누어 30개 폴더 생성
    for i in range(1, 31):
        version_dir_4 = os.path.join(base_dir_4, f"version_{i:02d}")
        version_dir_6 = os.path.join(base_dir_6, f"version_{i:02d}")
        version_dir_8 = os.path.join(base_dir_8, f"version_{i:02d}")
        os.makedirs(version_dir_4, exist_ok=True)
        os.makedirs(version_dir_6, exist_ok=True)
        os.makedirs(version_dir_8, exist_ok=True)
        print(f"버전 폴더 생성: {version_dir_4}, {version_dir_6}, {version_dir_8}")

def train_hidden4_agent():
    """은닉층 4개 에이전트 학습"""
    print("=== 은닉층 4개 다층 퍼셉트론 학습 시작 ===")
    
    config = load_config('multilayer/hidden4/config_hidden4.yaml')
    device = config['device']
    
    env = ChessEnv()
    agent = MultilayerPerceptronAgentHidden4(
        lr=config['learning_rate'], 
        gamma=config['gamma'], 
        hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'],
        device=device
    )

    epsilon = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay_start = config['epsilon_decay_start']
    epsilon_decay_end = config['epsilon_decay_end']

    metrics = init_metrics()

    for episode in range(config['num_episodes']):
        # 동적 epsilon_decay 계산
        progress = episode / config['num_episodes']
        epsilon_decay = epsilon_decay_start + (epsilon_decay_end - epsilon_decay_start) * progress
        
        state = env.reset()
        total_reward = 0
        done = False
        for t in range(config['max_steps_per_episode']):
            legal_moves = env.legal_moves()
            if not legal_moves:
                break
            action = agent.act(state, legal_moves, env, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        # 최종 reward로 결과 판단 (승/무/패)
        result = get_result_from_reward(reward)
        append_metrics(metrics, episode+1, total_reward, result)
        if (episode+1) % 100 == 0:
            print(f"[Hidden4] 에피소드 {episode+1}, 총 보상: {total_reward:.2f}, epsilon: {epsilon:.3f}")
        # 1만 에피소드마다 모델 저장 및 진행상황 출력
        if (episode+1) % 10000 == 0:
            version = (episode+1) // 10000
            model_path = f"hidden4_perceptron_v{version}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"[Hidden4] {episode+1} 에피소드 완료, 모델 저장: {model_path}")

    # 성능지표 저장 및 시각화
    save_metrics_csv(metrics, 'hidden4_metrics.csv')
    plot_learning_curve(metrics, save_path='hidden4_learning_curve.png')
    plot_winrate(metrics, save_path='hidden4_winrate_curve.png')
    # 최종 모델 저장
    torch.save(agent.state_dict(), 'hidden4_perceptron_final.pth')
    print("=== 은닉층 4개 다층 퍼셉트론 학습 완료! ===")

def train_hidden6_agent():
    """은닉층 6개 에이전트 학습"""
    print("=== 은닉층 6개 다층 퍼셉트론 학습 시작 ===")
    
    config = load_config('multilayer/config_multilayer.yaml')
    device = config['device']
    
    env = ChessEnv()
    agent = MultilayerPerceptronAgent(
        lr=config['learning_rate'], 
        gamma=config['gamma'], 
        hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'],
        device=device
    )

    epsilon = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay_start = config['epsilon_decay_start']
    epsilon_decay_end = config['epsilon_decay_end']

    metrics = init_metrics()

    for episode in range(config['num_episodes']):
        # 동적 epsilon_decay 계산
        progress = episode / config['num_episodes']
        epsilon_decay = epsilon_decay_start + (epsilon_decay_end - epsilon_decay_start) * progress
        
        state = env.reset()
        total_reward = 0
        done = False
        for t in range(config['max_steps_per_episode']):
            legal_moves = env.legal_moves()
            if not legal_moves:
                break
            action = agent.act(state, legal_moves, env, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        # 최종 reward로 결과 판단 (승/무/패)
        result = get_result_from_reward(reward)
        append_metrics(metrics, episode+1, total_reward, result)
        if (episode+1) % 100 == 0:
            print(f"[Hidden6] 에피소드 {episode+1}, 총 보상: {total_reward:.2f}, epsilon: {epsilon:.3f}")
        # 1만 에피소드마다 모델 저장 및 진행상황 출력
        if (episode+1) % 10000 == 0:
            version = (episode+1) // 10000
            model_path = f"hidden6_perceptron_v{version}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"[Hidden6] {episode+1} 에피소드 완료, 모델 저장: {model_path}")

    # 성능지표 저장 및 시각화
    save_metrics_csv(metrics, 'hidden6_metrics.csv')
    plot_learning_curve(metrics, save_path='hidden6_learning_curve.png')
    plot_winrate(metrics, save_path='hidden6_winrate_curve.png')
    # 최종 모델 저장
    torch.save(agent.state_dict(), 'hidden6_perceptron_final.pth')
    print("=== 은닉층 6개 다층 퍼셉트론 학습 완료! ===")

def train_hidden8_agent():
    """은닉층 8개 에이전트 학습"""
    print("=== 은닉층 8개 다층 퍼셉트론 학습 시작 ===")
    
    config = load_config('multilayer/hidden8/config_hidden8.yaml')
    device = config['device']
    
    env = ChessEnv()
    agent = MultilayerPerceptronAgentHidden8(
        lr=config['learning_rate'], 
        gamma=config['gamma'], 
        hidden_dim=config['hidden_dim'],
        num_hidden_layers=config['num_hidden_layers'],
        device=device
    )

    epsilon = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay_start = config['epsilon_decay_start']
    epsilon_decay_end = config['epsilon_decay_end']

    metrics = init_metrics()

    for episode in range(config['num_episodes']):
        # 동적 epsilon_decay 계산
        progress = episode / config['num_episodes']
        epsilon_decay = epsilon_decay_start + (epsilon_decay_end - epsilon_decay_start) * progress
        
        state = env.reset()
        total_reward = 0
        done = False
        for t in range(config['max_steps_per_episode']):
            legal_moves = env.legal_moves()
            if not legal_moves:
                break
            action = agent.act(state, legal_moves, env, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        # 최종 reward로 결과 판단 (승/무/패)
        result = get_result_from_reward(reward)
        append_metrics(metrics, episode+1, total_reward, result)
        if (episode+1) % 100 == 0:
            print(f"[Hidden8] 에피소드 {episode+1}, 총 보상: {total_reward:.2f}, epsilon: {epsilon:.3f}")
        # 1만 에피소드마다 모델 저장 및 진행상황 출력
        if (episode+1) % 10000 == 0:
            version = (episode+1) // 10000
            model_path = f"hidden8_perceptron_v{version}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"[Hidden8] {episode+1} 에피소드 완료, 모델 저장: {model_path}")

    # 성능지표 저장 및 시각화
    save_metrics_csv(metrics, 'hidden8_metrics.csv')
    plot_learning_curve(metrics, save_path='hidden8_learning_curve.png')
    plot_winrate(metrics, save_path='hidden8_winrate_curve.png')
    # 최종 모델 저장
    torch.save(agent.state_dict(), 'hidden8_perceptron_final.pth')
    print("=== 은닉층 8개 다층 퍼셉트론 학습 완료! ===")

def backup_all_files():
    """모든 파일들을 백업 폴더로 이동"""
    # 은닉층 4개 버전 백업
    backup_dir_4 = "saved_models/multilayer/hidden4"
    os.makedirs(backup_dir_4, exist_ok=True)
    
    files_to_backup_4 = [
        'hidden4_perceptron_final.pth',
        'hidden4_metrics.csv',
        'hidden4_learning_curve.png',
        'hidden4_winrate_curve.png'
    ]
    
    # 버전별 모델 파일들도 백업
    for i in range(1, 31):
        version_file = f'hidden4_perceptron_v{i}.pth'
        if os.path.exists(version_file):
            files_to_backup_4.append(version_file)
    
    for file in files_to_backup_4:
        if os.path.exists(file):
            shutil.move(file, os.path.join(backup_dir_4, file))
            print(f"은닉층 4개 백업 완료: {file} → {backup_dir_4}")
    
    # 은닉층 6개 버전 백업
    backup_dir_6 = "saved_models/multilayer/hidden6"
    os.makedirs(backup_dir_6, exist_ok=True)
    
    files_to_backup_6 = [
        'hidden6_perceptron_final.pth',
        'hidden6_metrics.csv',
        'hidden6_learning_curve.png',
        'hidden6_winrate_curve.png'
    ]
    
    # 버전별 모델 파일들도 백업
    for i in range(1, 31):
        version_file = f'hidden6_perceptron_v{i}.pth'
        if os.path.exists(version_file):
            files_to_backup_6.append(version_file)
    
    for file in files_to_backup_6:
        if os.path.exists(file):
            shutil.move(file, os.path.join(backup_dir_6, file))
            print(f"은닉층 6개 백업 완료: {file} → {backup_dir_6}")
    
    # 은닉층 8개 버전 백업
    backup_dir_8 = "saved_models/multilayer/hidden8"
    os.makedirs(backup_dir_8, exist_ok=True)
    
    files_to_backup_8 = [
        'hidden8_perceptron_final.pth',
        'hidden8_metrics.csv',
        'hidden8_learning_curve.png',
        'hidden8_winrate_curve.png'
    ]
    
    # 버전별 모델 파일들도 백업
    for i in range(1, 31):
        version_file = f'hidden8_perceptron_v{i}.pth'
        if os.path.exists(version_file):
            files_to_backup_8.append(version_file)
    
    for file in files_to_backup_8:
        if os.path.exists(file):
            shutil.move(file, os.path.join(backup_dir_8, file))
            print(f"은닉층 8개 백업 완료: {file} → {backup_dir_8}")

def main():
    print("=== 다층 퍼셉트론 은닉층 4개 vs 6개 vs 8개 동시 학습 시작 ===")
    
    # 버전 폴더 생성
    create_version_folders()
    
    # 세 에이전트를 동시에 학습 (멀티스레딩)
    thread_hidden4 = threading.Thread(target=train_hidden4_agent)
    thread_hidden6 = threading.Thread(target=train_hidden6_agent)
    thread_hidden8 = threading.Thread(target=train_hidden8_agent)
    
    start_time = time.time()
    
    thread_hidden4.start()
    thread_hidden6.start()
    thread_hidden8.start()
    
    # 세 스레드가 완료될 때까지 대기
    thread_hidden4.join()
    thread_hidden6.join()
    thread_hidden8.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 모든 학습 완료! 총 소요 시간: {total_time/3600:.2f}시간 ===")
    
    # 파일 백업
    backup_all_files()
    
    print("\n=== 결과 요약 ===")
    print("은닉층 4개 결과: saved_models/multilayer/hidden4/")
    print("은닉층 6개 결과: saved_models/multilayer/hidden6/")
    print("은닉층 8개 결과: saved_models/multilayer/hidden8/")
    print("각 버전별로 10000 에피소드 단위로 모델이 저장되었습니다.")

if __name__ == '__main__':
    main() 