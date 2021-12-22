import numpy as np
from numpy.core.numeric import Infinity
import gym
from collections import namedtuple
from time import sleep
import plotting


DEFAULT_CONFIG = {
  'MAX_EPISODES': 2000,
  'MAX_STEPS_PER_EPISODE': 1000,
  'LEARNING_RATE': 0.01,
  'DISCOUNT_RATE': 0.99,
  'EXPLORATION_DECAY': 0.01,
  'MIN_EXPLORATION_RATE': 0.1,
  'DO_RENDER': False,
  'SHOW_STATS_PLOT': True,
}

# Named storage of experiences
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'step'))


def setup_environment(name):
  env = gym.make(name)
  return env

class Agent():
  exploration_rate = 1.0
  experiences = []
  highest_env_reward = -Infinity

  def __init__(self, env, reward_function = None, **config):
    self.env = env
    self.config = config

    self.calculate_reward = reward_function if reward_function is not None else lambda *args: args[0]
    self.reset()

    # Since we have a discrete and small action space, we can create
    # one set of weights per action.
    nbr_of_weights, nbr_of_actions = env.observation_space.shape[0], env.action_space.n

    # Start with random weights.
    self.W = np.random.uniform(low=0.0, high=1.0, size=(nbr_of_weights, nbr_of_actions))

  def reset(self):
    ''' Get and set initial state. Decay exploration rate. '''
    self.state = self.env.reset()
    self.step = 0

    # Decrease chance of exploration according to an epsilon-greedy strategy.
    self.exploration_rate -= self.config['EXPLORATION_DECAY']
    self.exploration_rate = np.max([self.exploration_rate, self.config['MIN_EXPLORATION_RATE']])
  
  def get_best_action(self, state):
    ''' Calculate the "quality" of each action as the weighted sum (dot-product) of 
    features and weights and return the (index of the) best action. '''
    # 
    action_values = self.W.T.dot(state)
    best_action = np.argmax(action_values)
    return best_action
  
  def observation_to_state(self, observation):
    ''' Convert observations to a new state (add logic for this here if needed…).
    Compare e.g. how our senses convert detected light-waves (the observation) into "objects" (the state).'''
    state = observation.copy()
    return state
  
  def should_explore(self):
    do_explore = np.random.uniform() < self.exploration_rate
    return do_explore

  def take_action(self):
    self.step += 1
    do_explore = self.should_explore()

    if do_explore:
      action = self.env.action_space.sample()
    else:
      action = self.get_best_action(self.state)

    observation, env_reward, done, _info = self.env.step(action)    
    self.highest_env_reward = env_reward if env_reward > self.highest_env_reward else self.highest_env_reward

    next_state = self.observation_to_state(observation)

    # Allow creating home-made rewards instead of environment's reward (if any) to facilitate learning.
    our_reward = self.calculate_reward(env_reward, self.state.copy(), next_state.copy(), self.step)

    experience = Experience(self.state.copy(), action, observation, our_reward, self.step)
    self.experiences.append(experience)
    self.state = next_state.copy()
    return experience, done
  
  def learn_from(self, experiences):
    errors = []

    learning_rate = self.config['LEARNING_RATE']
    discount_rate = self.config['DISCOUNT_RATE']

    for exp in experiences:
      reward = exp.reward

      # Calculate the error (i.e. value difference) between the action we took and the actual value.
      action_value_step0 = np.dot(self.W[:, exp.action], exp.state.T)
      action_value_step1 = np.max(np.dot(self.W.T, exp.next_state))
      
      estimated_value = action_value_step0
      target = reward + discount_rate * action_value_step1

      error = np.abs(target - estimated_value)
      
      # Normalise errors as a fraction of the target value (since we don't normalise action values).
      norm_error = error / np.abs(target)
      norm_sqr_error = error ** 2 / np.abs(target)

      # Change the weights for this action by an amount proportional to how much 
      # each state component (feature) contributes to the action value. The higher the
      # value of a feature – the more will its weights adjust towards the target.
      delta = norm_error * exp.state

      # Use gradient value clipping to prevent the error from bouncing away too much.
      # delta = np.clip(delta, -1.0, 1.0)

      delta = learning_rate * delta
      self.W[:, exp.action] += delta

      errors.append(norm_sqr_error)

    return errors




def execute(env_name, reward_function = None, **config_overrides):
  config = { **DEFAULT_CONFIG, **config_overrides }
  env = setup_environment(env_name)
  agent = Agent(env, reward_function=reward_function, **config)

  errors = []
  rewards = []
  avg_rewards = []

  for episode in range(config['MAX_EPISODES']):
    agent.reset()
    episode_experiences = []
    episode_rewards = []
    episode_rewards = []
    for _step in range(config['MAX_STEPS_PER_EPISODE']):
      if config['DO_RENDER'] and episode % 200 == 0:
        env.render()
        sleep(0.002)
      experience, done = agent.take_action()
      episode_rewards.append(experience.reward)
      episode_experiences.append(experience)
      if done:
        break
    
    # Fit weights to randomly picked experiences.
    nbr_samples = np.min([len(agent.experiences), 500])
    indices = np.random.choice(nbr_samples, size=nbr_samples, replace=False)
    learning_experiences = [agent.experiences[i] for i in indices]
    episode_avg_error = np.mean(agent.learn_from(learning_experiences))
    episode_max_reward  = np.max(episode_rewards)
    episode_avg_reward  = np.mean(episode_rewards)

    # Store statistics.
    errors.append(episode_avg_error)
    rewards.append(episode_max_reward)
    avg_rewards.append(episode_avg_reward)

    if episode % 100 == 0:
      print(f'Episode {episode}:\nError: {episode_avg_error}\nMax reward: {episode_max_reward}')
  
  if config['SHOW_STATS_PLOT']:
    plotting.show_stats_plot(rewards, errors, avg_rewards)
  return rewards
  

if __name__ == '__main__':
  # optimise_hyper_params()
  _rewards = execute('CartPole-v1')