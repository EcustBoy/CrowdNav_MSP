import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import sys
sys.path.append("..")
import json

import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib

device = 'cpu'

# configure policy
policy = policy_factory['sarl']()
policy_config = configparser.RawConfigParser()
policy_config.read('configs/policy.config')
policy.configure(policy_config)
policy.set_device(device)

# getting trained weights
weights_path = 'data/output/rl_model.pth'
policy.model.load_state_dict(torch.load(weights_path))

# domain/env config (initially same as used in training)
env_config = configparser.RawConfigParser()
env_config.read('configs/env.config')
env = CrowdSim()
env.configure(env_config)
robot = Robot(env_config, 'robot')
robot.set_policy(policy)
env.set_robot(robot)

# explorer, memory is not used as we will only use the explorer in test mode
memory = ReplayMemory(100000)
explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

policy.set_env(env)
robot.set_policy(policy)
robot.print_info()

#this directory will contain the results of our testing 
output_path = 'data/domain_tests'
if not os.path.exists(output_path) :
    os.makedirs(output_path)

if __name__ == "__main__":
    mods = {'neighbor_dist' : 1, 'max_neighbors' : 1, 'time_horizon' : 10, 'max_speed' : 1}
    #env.modify_domain(mods)
    #env.circle_radius = 8
    env.human_num = 10
    env.test_sim = 'corr-o'
    explorer.run_k_episodes(1, 'test', output_info=True)
    env.render('video')