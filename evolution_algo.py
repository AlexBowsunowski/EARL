from torch.multiprocessing import set_start_method
from torch.multiprocessing import freeze_support
from typing import List

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time
import torch.multiprocessing as mp
import threading

import concurrent.futures

from agent import Agent
from agent_test import AgentTest

#Number of agents working in parallel
num_agents = 1
agents = []
for i in range(num_agents):
    env = gym.make('CartPole-v1')
    env.reset(seed=i) 
    agent = AgentTest(env, state_size=4, action_size=2, seed=i)
    agents.append(agent)

def sample_reward(current_weight, index, gamma, max_t, std):
    rng = np.random.RandomState(index)
    weights = current_weight + (std*rng.randn(agents[index].get_weights_dim())) 
    reward = agents[index].evaluate(weights, gamma, max_t)
    return {index : reward}

def update_weights(weights, indices, alpha, std, rewards):
    scaled_rewards = np.zeros(len(rewards))
    reconstructed_weights = np.zeros((len(rewards), agents[0].get_weights_dim()))
    for i in indices:
        rng = np.random.RandomState(i)
        scaled_rewards[i] = rewards[i]
        reconstructed_weights[i] = weights + std*rng.randn(agents[i].get_weights_dim())
    scaled_rewards = (scaled_rewards - np.mean(scaled_rewards)) / (np.std(scaled_rewards) + 0.1)
    n = len(rewards)
    deltas = alpha / (n * std) * np.dot(reconstructed_weights.T, scaled_rewards)
    return weights + deltas

def evolution(
        n_iterations=400, # number of episodes used to train the agent
        max_t=2000, # maximum number of timesteps per episode
        alpha = 0.01, # iteration step
        gamma=1.0, # discount rate
        std=0.1 # standard deviation of additive noise
    ) -> List:
    """Using Deep Q-learning""" 

    scores_deque = deque(maxlen=100)
    scores = []
    current_weights = []
    rewards = {}
    start_time = time.time()
    current_weights = std*np.random.randn(agents[0].get_weights_dim())
    indices = [i for i in range(num_agents)]

    for i_iteration in range(1, n_iterations+1):
        rewards.clear()

        pool = mp.Pool(num_agents)
        for i in range(num_agents):
            rewards.update(pool.apply(sample_reward, args = (current_weights, i, gamma, max_t, std,)))
        pool.close()
        pool.join()
        pool.terminate()
        current_weights = update_weights(current_weights, indices, alpha, std, rewards)

        current_rewards = []
        current_rewards.append(agents[0].evaluate(current_weights, gamma=1.0))
        current_reward = np.max(current_rewards)
        scores_deque.append(current_reward)
        scores.append(current_reward)
        
        torch.save(agent.state_dict(), 'evolution_algo_checkpoint.pth')
        
        if i_iteration % 1 == 0:
            print(f'Episode {i_iteration}\tAverage Score: {np.mean(scores_deque)}')
        if i_iteration % 100 == 0:
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)

        if np.mean(scores_deque)>=195.0:
            print(f'Environment solved in {i_iteration-100} iterations!\tAverage Score: {np.mean(scores_deque)}')
            elapsed_time = time.time() - start_time
            print("Training duration: ", elapsed_time)
            break
    return scores

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    scores = evolution()



    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'evolution_algo_with_{num_agents}_agents.png')