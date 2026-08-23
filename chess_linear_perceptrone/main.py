import yaml
import numpy as np
import torch
import os
import shutil
import subprocess
from envs.chess_env import ChessEnv
from agents.perceptron_agent import PerceptronAgent
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

def backup_files():
    """기존 파일들을 백업 폴더로 이동"""
    backup_dir = "saved_models/single"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        'perceptron_agent.pth',
        'perceptron_agent_v1.pth',
        'metrics.csv',
        'learning_curve.png',
        'winrate_curve.png'
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.move(file, os.path.join(backup_dir, file))
            print(f"백업 완료: {file} → {backup_dir}")

def run_multilayer():
    """다층 퍼셉트론 학습 실행"""
    print("\n=== 다층 퍼셉트론 학습 시작 ===")
    subprocess.run(['python', 'multilayer/main_multilayer.py'])

def main():
    config = load_config('configs/config.yaml')
    device = config['device']
    env = ChessEnv()
    agent = PerceptronAgent(lr=config['learning_rate'], gamma=config['gamma'], device=device)

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
            print(f"단층 퍼셉트론 - 에피소드 {episode+1}, 총 보상: {total_reward:.2f}, epsilon: {epsilon:.3f}, decay: {epsilon_decay:.4f}")
        # 1만 에피소드마다 모델 저장 및 진행상황 출력
        if (episode+1) % 10000 == 0:
            version = (episode+1) // 10000
            model_path = f"perceptron_agent_v{version}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"[진행상황] 단층 퍼셉트론 {episode+1} 에피소드 완료, 모델 저장: {model_path}")

    # 성능지표 저장 및 시각화
    save_metrics_csv(metrics, 'metrics.csv')
    plot_learning_curve(metrics, save_path='learning_curve.png')
    plot_winrate(metrics, save_path='winrate_curve.png')
    # 최종 모델 저장
    torch.save(agent.state_dict(), 'perceptron_agent.pth')
    print("단층 퍼셉트론 모델 및 성능지표 저장 완료!")

    # 파일 백업
    backup_files()
    
    # 다층 퍼셉트론 자동 실행
    run_multilayer()

if __name__ == '__main__':
    main() 