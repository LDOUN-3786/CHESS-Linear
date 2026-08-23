import csv
import os

def init_metrics():
    return {
        'episode': [],
        'reward': [],
        'win': [],
        'draw': [],
        'lose': []
    }

def append_metrics(metrics, episode, reward, result):
    # result: 'win', 'draw', 'lose'
    metrics['episode'].append(episode)
    metrics['reward'].append(reward)
    metrics['win'].append(1 if result == 'win' else 0)
    metrics['draw'].append(1 if result == 'draw' else 0)
    metrics['lose'].append(1 if result == 'lose' else 0)

def save_metrics_csv(metrics, path):
    fieldnames = ['episode', 'reward', 'win', 'draw', 'lose']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(metrics['episode'])):
            row = {k: metrics[k][i] for k in fieldnames}
            writer.writerow(row) 