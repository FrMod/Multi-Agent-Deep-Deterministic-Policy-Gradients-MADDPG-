import torch
import torch.nn.functional as F

import numpy as np

from Critic import DDPG_Critic
from Actor import DDPG_Actor
from ParameterSpaceNoise import AdaptiveParametricNoise

class DDPG_Agent(object):
    def __init__(self, name, **config):
        self.name = name
        self.seed = config["seed"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.input_size = config["input_size"]
        self.action_size = config["action_size"]
        self.batch_size = config["batch_size"]
        self.n_agents = config["n_agents"] 
           
        self.parameter_noise = AdaptiveParametricNoise(initial_noise_stdev = config["intial_noise_stdev"],
                                                        delta_distance = config["delta_distance"], 
                                                        alpha = config["alpha"])

        self.critic = DDPG_Critic(critic_lr=config["critic_lr"],
                                  input_size=self.input_size*self.n_agents,    
                                  action_size=self.action_size*self.n_agents,
                                  name=self.name + "_critic",
                                  weight_decay=config["weight_decay"],
                                  hidden_dim=config["critic_hidden_dim"],
                                  seed=self.seed)
        
        self.actor = DDPG_Actor(actor_lr=config["actor_lr"],
                                  input_size=self.input_size,    
                                  action_size=self.action_size,
                                  name=self.name + "_actor",    
                                  hidden_dim=config["actor_hidden_dim"],
                                  seed=self.seed)

        self.actor_perturbed = DDPG_Actor(actor_lr=config["actor_lr"],
                                  input_size=self.input_size,    
                                  action_size=self.action_size,
                                  name=self.name + "_actor_perturbed",    
                                  hidden_dim=config["actor_hidden_dim"],
                                  seed=self.seed)
        
        self.critic_target = DDPG_Critic(critic_lr=config["critic_lr"],
                                  input_size=self.input_size*self.n_agents,    
                                  action_size=self.action_size*self.n_agents,
                                  name=self.name + "_critic_target",   
                                  weight_decay=config["weight_decay"],
                                  hidden_dim=config["critic_hidden_dim"],
                                  seed=self.seed)

        self.actor_target = DDPG_Actor(actor_lr=config["actor_lr"],
                                  input_size=self.input_size,    
                                  action_size=self.action_size,
                                  name=self.name + "_actor_target",    
                                  hidden_dim=config["actor_hidden_dim"],
                                  seed=self.seed)

        print("===================Actor network================================")
        print(self.actor)
        print("================================================================")
        
        print()
        print("===================Critic network===============================")
        print(self.critic)
        print("================================================================")
        
        self.soft_update(self.critic, self.critic_target, tau=1)
        self.soft_update(self.actor, self.actor_target, tau=1)
    
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        self.update_every = config["update_every"]

    def perturb_actor(self):
        self.hard_update(self.actor, self.actor_perturbed)
        state_params = self.actor_perturbed.state_dict()
        for name in state_params:
            param = state_params[name]
            param += torch.normal(mean = 0.0, std=self.parameter_noise.noise_std_dev, size=param.shape, device=self.actor.device)

    def act(self, state, noisy=True):
        self.actor.eval()
        self.actor_perturbed.eval()
        state = torch.from_numpy(state).float().to(self.actor.device)

        with torch.no_grad():
            if noisy:
                action_values = self.actor_perturbed.forward(state).cpu().data.numpy()
            else:
                action_values = self.actor.forward(state).cpu().data.numpy()
        self.actor.train()
        self.actor_perturbed.train()

        return np.clip(action_values,-1,1)  # clip output 

        
    def learn(self, states, actions, rewards, next_states, dones, action_target, actions_pred, indx):

        rewards = rewards.reshape(-1, self.n_agents, 1)[:,indx,:]
        dones = dones.reshape(-1, self.n_agents, 1)[:,indx,:]

        self.actor_target.eval()
        self.critic_target.eval()
        self.critic.eval()

        Q_targets_next = self.critic_target.forward(next_states, action_target)
        Q_expected = self.critic.forward(states, actions)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))      

        # update the critic
        self.critic.train()
        critic_loss = F.mse_loss(Q_expected,Q_targets)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic.optimizer.step()
        
        # update the actor
        self.critic.eval()
        self.actor.train()
        
        actor_loss = -self.critic(states, actions_pred)
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()   

       
    def soft_update(self, network, target_network, tau):
        for target_param, local_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

    def hard_update(self, network, target_network):
        for target_param, local_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(local_param.data)

    def reset(self, state):
        a = self.act(state, noisy=False)
        a_p = self.act(state)
        self.parameter_noise.adapt(a, a_p)
        self.perturb_actor()
        
    def save(self):
        print("...saving parameters...")
        self.critic.save()
        self.actor.save()
        self.critic_target.save()
        self.actor_target.save()
    
    def load(self, path):
        print("...loading parameters...")
        self.critic.load(path)
        self.actor.load(path)
        self.critic_target.load(path)
        self.actor_target.load(path)