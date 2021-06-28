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
        "alpha":1.05,  
        "intial_noise_stdev":1e-3,
        }


def main():
    env = UnityEnvironment(file_name=PATH_TO_TENNIS, no_graphics=False, seed=config["seed"])
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    n_episodes=15000
    max_t=3000
    agent = MADDPG(**config)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    solved = False
    verbose = True

    for i_episode in range(1, n_episodes+1):
        agent.reset(states)
        env_info = env.reset(train_mode=True)[brain_name]          # reset the environment    
        states = env_info.vector_observations                      # get the current state (for each agent)
        ags_score = np.zeros(config["n_agents"])                   # initialize the score (for each agent)

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]              # send all actions to tne environment
            next_states = env_info.vector_observations            # get next state (for each agent)
            rewards = env_info.rewards                            # get reward (for each agent)
            dones = env_info.local_done                           # see if episode finished
            ags_score += rewards                                      # update the score (for each agent)

            agent.step(states, next_states, rewards, dones, actions) # step the agent

            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        score = np.max(ags_score)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        if verbose:
            print('\rEpisode {}\tAverage Score: {:.3f}\tMax Score: {:.3f}\tLast Episode Score:  {:.3f}\tDistance [0]: {:.3e}\tStDev [0]: {:.3e}\tDistance [1]: {:.3e}\tStDev [1]: {:.3e}'.format(i_episode, 
                np.mean(scores_window), 
                score, 
                np.sum(ags_score),
                agent.agents[0].parameter_noise.distance, 
                agent.agents[0].parameter_noise.noise_std_dev, 
                agent.agents[1].parameter_noise.distance, 
                agent.agents[1].parameter_noise.noise_std_dev), end="")
        else:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)))
            # Saving        
            agent.save()
            filename = 'scores'
            outfile = open(filename,'wb')
            pickle.dump(scores,outfile)
            outfile.close()

            if solved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)))
                agent.save()
                break
        if np.mean(scores_window)>=0.5:
            solved = True

if __name__ == '__main__':
    main()