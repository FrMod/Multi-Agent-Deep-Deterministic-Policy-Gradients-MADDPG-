from unityagents import UnityEnvironment
import numpy as np
from MultiAgent import MADDPG
from collections import deque
import pickle

PATH_TO_TENNIS = "" 

config = {"memory_size":int(1e6), 
        "batch_size":512,
        "input_size":24, 
        "action_size":2,
        "seed":42,
        "actor_hidden_dim":(400,300),
        "critic_hidden_dim":(400,300),
        "critic_lr": 1e-4,
        "actor_lr":5e-4,
        "tau":0.01,
        "gamma":0.99,      
        "weight_decay":0,
        "learn_every":1,
        "update_every":1,
        "n_agents":2,
        "delta_distance":0.01,
        "alpha":1.01,  
        "intial_noise_stdev":0.0,
        }

env = UnityEnvironment(file_name=PATH_TO_TENNIS)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# # reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# # number of agents 
# num_agents = len(env_info.agents)
# print('Number of agents:', num_agents)

# # size of each action
# action_size = brain.vector_action_space_size
# print('Size of each action:', action_size)

# # examine the state space 
# states = env_info.vector_observations
# state_size = states.shape[1]
# print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
# print('The state for the first agent looks like:', states[0])

agents = MADDPG(**config)
agents.load()
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
solved = False
verbose = True

for i in range(50):                                         # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(2)                          # initialize the score (for each agent)
    while True:
        actions = agents.act(states, noisy=False) # select an action (for each agent)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (maximum over agents) this episode: {:.3f}'.format(np.max(scores)))

env.close()