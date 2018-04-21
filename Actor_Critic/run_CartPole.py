import numpy as np
import gym
from RL_brain import Actor
from RL_brain import Critic

DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1) #repruducible
env = env.unwrapped

actor = Actor(n_features=env.observation_space.shape[0],n_actions=env.action_space.n,learning_rate=0.001)
critic = Critic(n_features=env.observation_space.shape[0],learning_rate=0.1)

for epoch in range(3000):
    observation = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()
        action = actor.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if done: reward = -20
        track_r.append(reward)
        td_error = critic.learn(observation,reward,observation_)
        actor.learn(observation,action,td_error)
        observation = observation_
        t += 1
        if done or t>= 1000:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", epoch, "  reward:", int(running_reward))
            break



