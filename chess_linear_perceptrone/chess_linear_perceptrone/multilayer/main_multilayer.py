import yaml
import numpy as np
import torch
import sys
import os
import shutil
sys.path.append('..')  # 상위 폴더의 모듈 import를 위해
from envs.chess_env import ChessEnv
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

def backup_multilayer_files():
    """다층 퍼셉트론 파일들을 백업 폴더로 이동"""
    backup_dir = "saved_models/multilayer"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        'multilayer_perceptron_final.pth',
        'multilayer_metrics.csv',
        'multilayer_learning_curve.png',
        'multilayer_winrate_curve.png'
    ]
    
    # 버전별 모델 파일들도 백업
    for i in range(1, 31):  # 30만회 / 1만회 = 30개 버전
        version_file = f'multilayer_perceptron_v{i}.pth'
        if os.path.exists(version_file):
            files_to_backup.append(version_file)
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.move(file, os.path.join(backup_dir, file))
            print(f"다층 퍼셉트론 백업 완료: {file} → {backup_dir}")

def main():
    config = load_config('multilayer/config_multilayer.yaml')
    device = config['device']
    env = ChessEnv()
    agent = MultilayerPerceptronAgent(
        lr=config['learning_rate'], 
        gamma=config['gamma'], 
        hidden_dim=config['hidden_dim'],
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
            print(f"다층 퍼셉트론 - 에피소드 {episode+1}, 총 보상: {total_reward:.2f}, epsilon: {epsilon:.3f}, decay: {epsilon_decay:.4f}")
        # 1만 에피소드마다 모델 저장 및 진행상황 출력
        if (episode+1) % 10000 == 0:
            version = (episode+1) // 10000
            model_path = f"multilayer_perceptron_v{version}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"[진행상황] 다층 퍼셉트론 {episode+1} 에피소드 완료, 모델 저장: {model_path}")

    # 성능지표 저장 및 시각화
    save_metrics_csv(metrics, 'multilayer_metrics.csv')
    plot_learning_curve(metrics, save_path='multilayer_learning_curve.png')
    plot_winrate(metrics, save_path='multilayer_winrate_curve.png')
    # 최종 모델 저장
    torch.save(agent.state_dict(), 'multilayer_perceptron_final.pth')
    print("다층 퍼셉트론 모델 및 성능지표 저장 완료!")

    # 파일 백업
    backup_multilayer_files()
    
    print("\n=== 모든 학습 완료! ===")
    print("단층 퍼셉트론 결과: saved_models/single/")
    print("다층 퍼셉트론 결과: saved_models/multilayer/")

if __name__ == '__main__':
    main() 