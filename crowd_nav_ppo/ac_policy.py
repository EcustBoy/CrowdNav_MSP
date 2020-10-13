import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
from torch import nn
from torch.functional import F
from torch.distributions import MultivariateNormal, Categorical
import git
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState
from collections import namedtuple
from torch.autograd import Function
import math 

Human_info = namedtuple('ActionXY', ['px', 'py', 'vx', 'vy', 'radius'])

def mlp_tanh(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.Tanh())
    net = nn.Sequential(*layers)
    return net

def b(s) :
    return True if b == 'true' else False

class clipped_exp(Function):

    @staticmethod
    def forward(ctx, input, clip=15):
        ctx.save_for_backward(input)
        output = torch.exp(torch.clamp(input, min=-clip, max=clip))
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        return torch.ones_like(grad_output) * grad_output

class Feature_ext(nn.Module):
    """
    Extract the features of the crowd, Figure 3 and 4 of paper (https://arxiv.org/pdf/1809.08835.pdf)
    """
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp_tanh(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp_tanh(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp_tanh(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp_tanh(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        joint_state = torch.cat([self_state, weighted_feature], dim=1)

        return joint_state


class ActorCritic(nn.Module):
    """
    The actor crtitic model used for PPO (proximal policy approximation), the base of the model is the crowd feature extractor, from there there are two heads :
    - critic : models the value function
    - actor : modles the policy

    The possible actions are {(theta, velocitiy) : 0<=theta<2*PI, 0<=velocity<=v_pref}, we use the Beta distribution (https://en.wikipedia.org/wiki/Beta_distribution) 
    given by the parameters outputed by the actor head to model both of them. The Beta distribution outputs a number between 0 and 1 so we need to interpret this output
    as the porp.

    """
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims,  mlp_actor_dims, 
                mlp_critic_dims, with_global_state,
                cell_size, cell_num, annealing_rate, init_std):
        super(ActorCritic, self).__init__()
        self.annealing_rate = annealing_rate
        self.init_std = init_std

        #feature extractor
        self.f_ext = Feature_ext(input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims, with_global_state,
                 cell_size, cell_num)

        self.f_dim = mlp2_dims[-1] + self_state_dim
        
        self.critic_head = mlp_tanh(self.f_dim, mlp_critic_dims)
        self.actor_head = mlp_tanh(self.f_dim, mlp_actor_dims)
        
        self.joint_state = None
                
    def forward(self, x, time):
        x = self.f_ext(x)
        self.joint_state = x
        value = self.critic_head(x)
        dist_mean = self.actor_head(x)
        
        if time==0 :
            dist_std = self.init_std
        else : 
            dist_std = max(self.init_std * math.exp(-self.annealing_rate*time), 0.01)
        
        dist = MultivariateNormal(dist_mean, dist_std *torch.eye(2, 2).expand(x.shape[0], 2, 2))
        self.dist_mean = dist_mean 
        self.dist_std = dist_std 
        
        return value, dist  

class ActorCritic_simple(nn.Module):
    """
    The actor crtitic model used for PPO (proximal policy approximation), the base of the model is the crowd feature extractor, from there there are two heads :
    - critic : models the value function
    - actor : modles the policy

    The possible actions are {(theta, velocitiy) : 0<=theta<2*PI, 0<=velocity<=v_pref}, we use the Beta distribution (https://en.wikipedia.org/wiki/Beta_distribution) 
    given by the parameters outputed by the actor head to model both of them. The Beta distribution outputs a number between 0 and 1 so we need to interpret this output
    as the porp.

    """
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims,  mlp_actor_dims, 
                mlp_critic_dims, with_global_state,
                cell_size, cell_num, annealing_rate, init_std):
        super(ActorCritic, self).__init__()
        self.annealing_rate = annealing_rate
        self.init_std = init_std       
        
        self.critic_head = mlp_tanh(9, mlp_critic_dims)
        self.actor_head = mlp_tanh(9, mlp_actor_dims)
        
        self.joint_state = None
                
    def forward(self, x, time):
        value = self.critic_head(x)
        dist_mean = self.actor_head(x)
        
        if time==0 :
            dist_std = self.init_std
        else : 
            dist_std = max(self.init_std * math.exp(-self.annealing_rate*time), 0.01)
        
        dist = MultivariateNormal(dist_mean, dist_std *torch.eye(2, 2).expand(x.shape[0], 2, 2))
        self.dist_mean = dist_mean 
        self.dist_std = dist_std 
        
        return value, dist  

# class ActorCritic_discrete(nn.Module):
#     """
#     The actor crtitic model used for PPO (proximal policy approximation), the base of the model is the crowd feature extractor, from there there are two heads :
#     - critic : models the value function
#     - actor : modles the policy

#     The possible actions are {(theta, velocitiy) : 0<=theta<2*PI, 0<=velocity<=v_pref}, we use the Beta distribution (https://en.wikipedia.org/wiki/Beta_distribution) 
#     given by the parameters outputed by the actor head to model both of them. The Beta distribution outputs a number between 0 and 1 so we need to interpret this output
#     as the porp.

#     """
#     def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims,  mlp_actor_dims, 
#                 mlp_critic_dims, with_global_state,
#                 cell_size, cell_num, n_actions=30):
#         super(ActorCritic_discrete, self).__init__()
        
#         #feature extractor
#         self.f_ext = Feature_ext(input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims, with_global_state,
#                  cell_size, cell_num)

#         self.f_dim = mlp2_dims[-1] + self_state_dim
        
#         self.critic_head = mlp_tanh(self.f_dim, mlp_critic_dims)
#         self.actor_head = mlp_tanh(self.f_dim, mlp_actor_dims + [n_actions])
        
#         self.joint_state = None
                
#     def forward(self, x):
#         x = self.f_ext(x)
#         self.joint_state = x
#         value = self.critic_head(x)
#         dist_params = torch.softmax(self.actor_head(x), dim=1)
#         dist = Categorical(dist_params)
#         return value, dist

class AC_Policy(Policy) :
    """
    Class for policy using an actor critic architecture
    """
    def __init__(self) :
        self.name = 'PPO'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = ''
        self.gamma = None
        self.model = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim
        self.device=None
        self.init_std=None
        self.annealing_rate = None
        self.time = 0
    
    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def configure(self, config):
        mlp1_dims = [int(x) for x in config.get('ppo', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('ppo', 'mlp2_dims').split(', ')]
        mlp_actor_dims = [int(x) for x in config.get('ppo', 'mlp_actor_dims').split(', ')]
        mlp_critic_dims = [int(x) for x in config.get('ppo', 'mlp_critic_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('ppo', 'attention_dims').split(', ')]
        self.multiagent_training = config.get('ppo', 'multiagent_training')
        self.with_om = b(config.get('ppo', 'with_om'))
        self.with_global_state = b(config.get('ppo', 'with_global_state')) 
        self.init_std = int(config.get('ppo', 'init_std'))
        self.annealing_rate = float(config.get('ppo', 'annealing_rate'))
        device = config.get('ppo', 'device')
        self.device=torch.device(device)
        self.cell_num = int(config.get('om', 'cell_num'))
        self.cell_size = int(config.get('om', 'cell_size'))
        self.om_channel_size = int(config.get('om', 'om_channel_size'))
        
        self.model = ActorCritic(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims,
            attention_dims, mlp_actor_dims, mlp_critic_dims, self.with_global_state, self.cell_size, self.cell_num, self.annealing_rate, self.init_std).to(self.device)
        logging.info('Policy: PPO {} occupancy map, {} global state, model on {}'.format('w' if self.with_om else 'w/o', 'w' if self.with_global_state else 'w/o', device))

    
    def predict(self, state, update_time=False) :

        #construct the state to give it to policy
        n_envs, n_humans, state_sz = state.shape
        
        batch_state = torch.tensor(state).to(self.device).view(-1, state_sz).to(self.device)
        def list_humans(st) :
            return [Human_info(st[i, 9], st[i, 10], st[i, 11], st[i, 12], st[i, 13]) for i in range(st.shape[0])]
            
        humans_per_env = [list_humans(state[i]) for i in range(n_envs)]
        rotated_batch_input = self.rotate(batch_state).to(self.device).to(torch.float32)
        if self.with_om:
            occupancy_maps = torch.cat([torch.Tensor(self.build_occupancy_maps(human_states)).to(self.device) for human_states in humans_per_env], dim=0)
            rotated_batch_input = torch.cat([rotated_batch_input.to(torch.float32), occupancy_maps.to(self.device).to(torch.float32)], dim=1)
        rotated_batch_input = rotated_batch_input.view(n_envs, n_humans, -1) 
        
        output = self.model(rotated_batch_input, self.time)
        if update_time :
            self.time+=1
        
        return output

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()


    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state

class AC_Policy_discrete(AC_Policy) :
    
    def configure(self, config):
        mlp1_dims = [int(x) for x in config.get('ppo', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('ppo', 'mlp2_dims').split(', ')]
        mlp_actor_dims = [int(x) for x in config.get('ppo', 'mlp_actor_dims').split(', ')]
        mlp_critic_dims = [int(x) for x in config.get('ppo', 'mlp_critic_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('ppo', 'attention_dims').split(', ')]
        self.multiagent_training = config.get('ppo', 'multiagent_training')
        self.with_om = b(config.get('ppo', 'with_om'))
        self.with_global_state = b(config.get('ppo', 'with_global_state')) 
        device = config.get('ppo', 'device')
        self.device=torch.device(device)
        self.cell_num = int(config.get('om', 'cell_num'))
        self.cell_size = int(config.get('om', 'cell_size'))
        self.om_channel_size = int(config.get('om', 'om_channel_size'))
        
        self.model = ActorCritic_discrete(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims,
            attention_dims, mlp_actor_dims, mlp_critic_dims, self.with_global_state, self.cell_size, self.cell_num).to(self.device)
        logging.info('Policy: PPO {} occupancy map, {} global state, model on {}'.format('w' if self.with_om else 'w/o', 'w' if self.with_global_state else 'w/o', device))

class AC_Policy(Policy) :
    """
    Class for policy using an actor critic architecture
    """
    def __init__(self) :
        self.name = 'PPO'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = ''
        self.gamma = None
        self.model = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim
        self.device=None
        self.init_std=None
        self.annealing_rate = None
        self.time = 0
    
    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def configure(self, config):
        mlp1_dims = [int(x) for x in config.get('ppo', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('ppo', 'mlp2_dims').split(', ')]
        mlp_actor_dims = [int(x) for x in config.get('ppo', 'mlp_actor_dims').split(', ')]
        mlp_critic_dims = [int(x) for x in config.get('ppo', 'mlp_critic_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('ppo', 'attention_dims').split(', ')]
        self.multiagent_training = config.get('ppo', 'multiagent_training')
        self.with_om = b(config.get('ppo', 'with_om'))
        self.with_global_state = b(config.get('ppo', 'with_global_state')) 
        self.init_std = int(config.get('ppo', 'init_std'))
        self.annealing_rate = float(config.get('ppo', 'annealing_rate'))
        device = config.get('ppo', 'device')
        self.device=torch.device(device)
        self.cell_num = int(config.get('om', 'cell_num'))
        self.cell_size = int(config.get('om', 'cell_size'))
        self.om_channel_size = int(config.get('om', 'om_channel_size'))
        
        self.model = ActorCritic(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims,
            attention_dims, mlp_actor_dims, mlp_critic_dims, self.with_global_state, self.cell_size, self.cell_num, self.annealing_rate, self.init_std).to(self.device)
        logging.info('Policy: PPO {} occupancy map, {} global state, model on {}'.format('w' if self.with_om else 'w/o', 'w' if self.with_global_state else 'w/o', device))

    
    def predict(self, state, update_time=False) :

        #construct the state to give it to policy
        n_envs, n_humans, state_sz = state.shape
        
        batch_state = torch.tensor(state).to(self.device).view(-1, state_sz).to(self.device)
        def list_humans(st) :
            return [Human_info(st[i, 9], st[i, 10], st[i, 11], st[i, 12], st[i, 13]) for i in range(st.shape[0])]
            
        humans_per_env = [list_humans(state[i]) for i in range(n_envs)]
        rotated_batch_input = self.rotate(batch_state).to(self.device).to(torch.float32)
        if self.with_om:
            occupancy_maps = torch.cat([torch.Tensor(self.build_occupancy_maps(human_states)).to(self.device) for human_states in humans_per_env], dim=0)
            rotated_batch_input = torch.cat([rotated_batch_input.to(torch.float32), occupancy_maps.to(self.device).to(torch.float32)], dim=1)
        rotated_batch_input = rotated_batch_input.view(n_envs, n_humans, -1) 
        
        output = self.model(rotated_batch_input, self.time)
        if update_time :
            self.time+=1
        
        return output

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()


    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
