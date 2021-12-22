import math
from q_learn_linear_func_approx import execute

config = {
  'MAX_EPISODES': 1000,
  # 'MAX_STEPS_PER_EPISODE': 1000,
  'LEARNING_RATE': 0.001,
  # 'DISCOUNT_RATE': 0.99,
  'EXPLORATION_DECAY': 0.001,
  'MIN_EXPLORATION_RATE': 0.01,
  'DO_RENDER': False,
  # 'SHOW_STATS_PLOT': True,
}

def reward_function_mountain_car(_env_reward, state, next_state, _step):
  # Potential and kinetic energy calculation borrowed from 
  energy = (math.sin(3 * next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) - (math.sin(3 * state[0]) * 0.0025 + 0.5 * state[1] * state[1])
  reward = 100 * energy
  return reward


def run():
  rewards = execute('MountainCar-v0', reward_function_mountain_car, **config)


if __name__ == '__main__':
  run()