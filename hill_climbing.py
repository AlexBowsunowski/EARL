import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time

env = gym.make('CartPole-v1')
env.reset(seed=17) 

from agent_test import AgentTest

agent = AgentTest(env, state_size=4, action_size=2)

def evolution(
        n_iterations=1000, # number of episodes used to train the agent
        max_t=2000, # maximum number of timesteps per episode
        gamma=1.0, # discount rate
        population=20, # size of population at each iteration
        elite_frac=0.4, # proportion of the population for selection
        std=0.2 # standard deviation of additive noise
    ):
    
    """Using Deep Q-learning"""

    scores_deque = deque(maxlen=100)
    scores = []
    n_elite=int(population*elite_frac)
    current_weights = std*np.random.randn(agent.get_weights_dim())

    start_time = time.time()

    for i_iteration in range(1, n_iterations+1):
        weights_pop = [current_weights + (std*np.random.randn(agent.get_weights_dim())) for i in range(population)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        weights_pop = [weights_pop[i] for i in elite_idxs]
        rewards = [rewards[i] for i in elite_idxs]
        
        print(rewards)
        sum_rewards = np.sum(rewards)
        rewards = rewards / (sum_rewards + 1)
        print(sum_rewards)
        current_weights = np.average(weights_pop, axis=0, weights=rewards)

        reward = agent.evaluate(current_weights, gamma=1.0)

        scores_deque.append(reward)
        scores.append(reward)
        
        torch.save(agent.state_dict(), 'hill_climbing_checkpoint.pth')
        
        if i_iteration % 4 == 0:
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


if __name__ == "__main__":
    input("Press Enter to continue...")
    scores = evolution()


    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('hill_climbing.png')
