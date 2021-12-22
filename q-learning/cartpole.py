import numpy as np
from q_learn_linear_func_approx import execute

config = {
  'MAX_EPISODES': 2500,
  # 'MAX_STEPS_PER_EPISODE': 1000,
  'LEARNING_RATE': 0.0005,
  # 'DISCOUNT_RATE': 0.99,
  'EXPLORATION_DECAY': 0.001,
  'MIN_EXPLORATION_RATE': 0.01,
  'DO_RENDER': True,
  # 'SHOW_STATS_PLOT': True,
}

def reward_function_cartpole(env_reward, state, next_state, step):
  cart_position, cart_speed, pole_angle, pole_speed = [np.abs(s) for s in state]
  next_cart_position, next_cart_speed, next_pole_angle, next_pole_speed = [np.abs(s) for s in next_state]
  decrease_in_angle = pole_angle - next_pole_angle
  decrease_in_pole_speed = pole_speed - next_pole_speed
  reward = 1 + 100 * decrease_in_angle - next_pole_speed  # - pole_angle #  - (next_pole_speed - pole_speed) + (cart_speed - next_cart_speed)
  return reward # env_reward - next_pole_speed - next_pole_angle


def run():
  rewards = execute('CartPole-v1', reward_function_cartpole, **config)


if __name__ == '__main__':
  run()