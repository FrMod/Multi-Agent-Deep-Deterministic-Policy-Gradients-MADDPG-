import torch
import torch.nn.functional as F
import os

import numpy as np

from ReplayBuffer import ReplayBuffer
from Agent import DDPG_Agent

class MADDPG(object):
    def __init__(self, **config):
        torch.autograd.set_detect_anomaly(True)
        self.agents = []
        self.action_size = config["action_size"]
        self.input_size = config["input_size"]
        self.n_agents = config["n_agents"]    #
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.agent_names = ["Paddle_%s" %i for i in range(self.n_agents)]

        for agent_id in range(self.n_agents):
            self.agents.append(DDPG_Agent(self.agent_names[agent_id], **config))

        self.memory = ReplayBuffer(self.action_size, 
                                    config["memory_size"], 
                                    config["batch_size"], 
                                    seed=config["seed"])

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        self.update_every = config["update_every"]
        self.learn_every = config["learn_every"]

    def step(self, states, next_states, rewards, dones, actions):
        # print('\rStates {0}\tNext States: {1}\tRewards: {2}\tDones: {3}\tAction: {4}'.format(np.shape(states), 
        #                                                                                                 np.shape(next_states), 
        #                                                                                                 np.shape(rewards), 
        #                                                                                                 np.shape(dones), 
        #                                                                                                 np.shape(actions)))
        states = states.reshape(1,-1)
        next_states = next_states.reshape(1,-1)
        actions = actions.reshape(1,-1)

        
        self.memory.add(states, actions, rewards, next_states, dones)
        if self.t_step % self.learn_every == 0:
            for agent_indx, agent in enumerate(self.agents):
                if len(self.memory) < self.batch_size :
                    return
                    
                states, actions, rewards, next_states, dones = self.memory.sample()
                self.learn(agent_indx,(states, actions, rewards, next_states, dones))

        if self.t_step % self.update_every == 0:
            for agent in self.agents:
                agent.soft_update(agent.actor, agent.actor_target, self.tau)
                agent.soft_update(agent.critic, agent.critic_target, self.tau) 
        
        self.t_step +=1


    def act(self, states, noisy=True):
        actions=[]
        for agent_indx, agent in enumerate(self.agents):
            actions.append(agent.act(states[agent_indx,:], noisy= noisy))
        return np.array(actions)

    def reset(self, states):
        for agent_indx, agent in enumerate(self.agents):
            agent.reset(states[agent_indx,:])

    def learn(self, indx, sampled_memory):

        states, actions, rewards, next_states, dones = sampled_memory
        # print('\rStates {0}\tNext States: {1}\tRewards: {2}\tDones: {3}\tAction: {4}'.format(np.shape(states), 
        #                                                                                 np.shape(next_states), 
        #                                                                                 np.shape(rewards), 
        #                                                                                 np.shape(dones), 
        #                                                                                 np.shape(actions)))
        
        action_targets = []
        actions_preds = []
        for agent_indx, agent in enumerate(self.agents):
            action_targets.append(agent.actor_target.forward(states.reshape(-1,self.n_agents,self.input_size)[:,agent_indx,:].detach()))
            actions_preds.append(agent.actor.forward(states.reshape(-1,self.n_agents,self.input_size)[:,agent_indx,:].detach()))
        
        action_targets = torch.cat(action_targets, dim=1).to(self.device)
        actions_preds = torch.cat(actions_preds, dim=1).to(self.device)

        self.agents[indx].learn(states, actions, rewards, next_states, dones, action_targets, actions_preds, agent_indx)

    def save(self):
        for agent in self.agents:
            agent.save()

    def load(self):
        current_path = os.getcwd() + "\BestWeights"
        for agent in self.agents:
            agent.load(current_path)