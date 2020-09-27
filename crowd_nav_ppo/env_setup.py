import sys
sys.path.append("..")
import configparser
import torch
import gym
import numpy as np
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from multiprocessing_env import SubprocVecEnv
from crowd_sim.envs.utils.info import *
from numpy.linalg import norm
import rvo2
from crowd_sim.envs.utils.human import Human
from ac_policy import AC_Policy, AC_Policy_discrete
from crowd_sim_ppo import CrowdSim_PPO

def gen_policy(policy_config_='configs/policy.config', policy='ac') :

    # configure policy
    policy = AC_Policy()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_)
    policy.configure(policy_config)
    #logging.info('Using device: %s', policy.device)
    return policy

def gen_policy_discrete(policy_config_='configs/policy.config', policy='ac') :

    # configure policy
    policy = AC_Policy_discrete()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_)
    policy.configure(policy_config)
    #logging.info('Using device: %s', policy.device)
    return policy

def gen_env(policy, env_config_='configs/env.config'):
    
    # # configure paths
    # make_new_dir = True
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
        
    # if make_new_dir:
    #     os.makedirs(output_dir)
    #     shutil.copy(env_config_, output_dir)
    #     shutil.copy(policy_config_, output_dir)
    # log_file = os.path.join(output_dir, 'output.log')
    # rl_weight_file = os.path.join(output_dir, 'rl_model.pth')

    # # configure logging
    # # mode = 'a' if resume else 'w'
    # # file_handler = logging.FileHandler(log_file, mode=mode)
    # # stdout_handler = logging.StreamHandler(sys.stdout)
    # # level = logging.INFO if not debug else logging.DEBUG
    # # logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
    # #                     format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_)
    env = CrowdSim_PPO()
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)


    # reinforcement learning
    robot.set_policy(policy)
    #robot.print_info()
    return env

def gen_multi_envs(n_envs, policy) :

    def make_env():
        def _thunk():
            env = gen_env(policy)
            return env

        return _thunk

    envs = [make_env() for i in range(n_envs)]
    envs = SubprocVecEnv(envs)
    return envs
