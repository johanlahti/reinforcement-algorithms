import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SMOOTHEN_LINES = False

def show_stats_plot(rewards, errors, avg_rewards):
  if SMOOTHEN_LINES:
    box_pts = 20
    box = np.ones(box_pts)/box_pts
    rewards = np.convolve(rewards, box, mode='same')
    errors = np.convolve(errors, box, mode='same')
    avg_rewards = np.convolve(avg_rewards, box, mode='same')

  df = pd.DataFrame({ 'errors': errors, 'rewards': rewards, 'avg_rewards': avg_rewards, 'episode': range(len(errors)) })

  alpha = 0.5

  df.plot(x='episode', y='errors', legend=True, color='r', alpha=alpha)
  df.plot(x='episode', y='rewards', legend=True, color='b', alpha=alpha)
  df.plot(x='episode', y='avg_rewards', legend=True, color='g', alpha=alpha)

  plt.title('Reward and error (y) vs episode number (x)')
  plt.show()