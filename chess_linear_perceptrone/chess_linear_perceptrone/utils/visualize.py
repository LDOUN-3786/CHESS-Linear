import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(metrics, save_path='learning_curve.png'):
    plt.figure(figsize=(10,5))
    plt.plot(metrics['episode'], metrics['reward'], label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve (Reward)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_winrate(metrics, save_path='winrate_curve.png', window=100):
    episodes = np.array(metrics['episode'])
    wins = np.array(metrics['win'])
    winrate = np.convolve(wins, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,5))
    plt.plot(episodes[:len(winrate)], winrate, label=f'Win rate (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rate Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(matrix, title, save_path='heatmap.png'):
    plt.figure(figsize=(6,6))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 