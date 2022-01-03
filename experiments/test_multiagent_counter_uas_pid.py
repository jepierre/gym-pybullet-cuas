"""Test script for multiagent problems.

This scripts runs the best model found by one of the executions of `multiagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_multiagent.py --exp ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict, Discrete
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import (
    LeaderFollowerAviary,
)
from gym_pybullet_drones.envs.multi_agent_rl.CounterUAS import CounterUAS
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

import shared_constants

OWN_OBS_VEC_SIZE = None  # Modified at runtime
ACTION_VEC_SIZE = None  # Modified at runtime

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Multi-agent reinforcement learning experiments script"
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>",
        metavar="",
    )

    ARGS = parser.parse_args()
    if ARGS.exp is None:
        # ARGS.exp = r"./results/save-leaderfollower-4-cc-kin-vel-06.17.2021_03.17.40"
        ARGS.exp = r"./results/save-leaderfollower-4-cc-kin-vel-06.18.2021_01.24.57"

    #### Parameters to recreate the environment ################
    NUM_DRONES = int(ARGS.exp.split("-")[2])
    OBS = (
        ObservationType.KIN if ARGS.exp.split("-")[4] == "kin" else ObservationType.RGB
    )
    if ARGS.exp.split("-")[5] == "rpm":
        ACT = ActionType.RPM
    elif ARGS.exp.split("-")[5] == "dyn":
        ACT = ActionType.DYN
    elif ARGS.exp.split("-")[5] == "pid":
        ACT = ActionType.PID
    elif ARGS.exp.split("-")[5] == "vel":
        ACT = ActionType.VEL
    elif ARGS.exp.split("-")[5] == "one_d_rpm":
        ACT = ActionType.ONE_D_RPM
    elif ARGS.exp.split("-")[5] == "one_d_dyn":
        ACT = ActionType.ONE_D_DYN
    elif ARGS.exp.split("-")[5] == "one_d_pid":
        ACT = ActionType.ONE_D_PID

    ACT = ActionType.VEL

    #### Constants, and errors #################################
    if OBS == ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 12
    elif OBS == ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ACT == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit()


    #### Create test environment ###############################

    initial_xyzs = np.vstack(
        [
            # np.array([0, 3, 1]),
            # np.array([0, -3, 1]),
            # np.array([1, 0, 1]),
            # np.array([-1, 0, 1]),
            np.array([4, 0, 1]),
            np.array([-4, 0, 1]),
            np.array([1, .1, 1]),
            np.array([-1, .1, 1]),
        ]
    )

    test_env = CounterUAS(
        num_drones=NUM_DRONES,
        initial_xyzs=initial_xyzs,
        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT,
        gui=True,
        record=False,
    )

    def calc_des_v(target_pos, own_pos, opponent_pos, kr=.5, ka=10):
        
        def calc_distance(obj1, obj2):
            """
            returns euclidean distance between 2 objects
            """
            return np.linalg.norm(obj1 - obj2)
        
        # kr = .01
        # ka = 20
        alpha =1 
        
        MAX_XY = 30
        own_pos = own_pos * MAX_XY
        opponent_pos = opponent_pos * MAX_XY
        distance_to_target = calc_distance(own_pos, target_pos)
        distance_to_opponent = calc_distance(own_pos, opponent_pos)

    
        
        # unit vector from attacker to defender and attacker to target
        # des_vx = -ka* (own_pos[0] - target_pos[0])
        # des_vy = -ka * (own_pos[1] - target_pos[1])
        
        # avoid_radius = 1
        # if distance_to_opponent <= avoid_radius: 
        #     des_vx += -kr * (1 - (distance_to_opponent / avoid_radius) )*(own_pos[0] - opponent_pos[0]) / distance_to_opponent**3
        #     des_vx += -kr * (1 - (distance_to_opponent / avoid_radius) )*(own_pos[1] - opponent_pos[1]) / distance_to_opponent**3
  
        des_vx = kr * ( 1/ distance_to_opponent**alpha) * ( ( own_pos[0] - opponent_pos[0]) / distance_to_opponent)
        des_vx += -ka * ( 1/ distance_to_target**alpha) * ( (own_pos[0] - target_pos[0]) / distance_to_target)

        des_vy = kr * (1 / distance_to_opponent**alpha) * ( (own_pos[1] - opponent_pos[1]) / distance_to_opponent)
        des_vy += -ka * (1/ distance_to_target ** alpha) * (( own_pos[1] - target_pos[1])  / distance_to_target)

        des_v = np.array([des_vx, des_vy, 0])
        des_v_mag = np.linalg.norm(des_v)

        des_v = des_v / des_v_mag
        
        return np.array([des_v[0], des_v[1], 0, des_v_mag])
        

    #### Show, record a video, and log the model's performance #
    obs = test_env.reset()
    logger = Logger(
        logging_freq_hz=int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS),
        num_drones=NUM_DRONES,
    )
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        action = {i: np.array([0]) for i in range(NUM_DRONES)}
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        action = {i: np.array([0, 0, 0, 0]) for i in range(NUM_DRONES)}
    elif ACT == ActionType.PID:
        action = {i: np.array([0, 0, 0]) for i in range(NUM_DRONES)}
    else:
        print("[ERROR] unknown ActionType")
        exit()
    start = time.time()
    total_reward_def = 0
    total_reward_attacker = 0
    for i in range(30 * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)):  # Up to 6''
        #### Deploy the policies ###################################
        temp = {}
        # temp[0] = np.array([0, 0, 1]) - obs[0]['own_obs'][0:3] 
        temp[0] = calc_des_v(np.array([0, 0, 1]), obs[0]['own_obs'][0:3], obs[obs[0]['closest_opponent']]['own_obs'][0:3])
        temp[1] = calc_des_v(np.array([0, 0, 1]), obs[1]['own_obs'][0:3], obs[obs[1]['closest_opponent']]['own_obs'][0:3])
        # temp[1] =   np.array([0, 0, 1]) - obs[1]['own_obs'][0:3]
        for d in range(2, NUM_DRONES):
            temp_v = (obs[obs[d]['closest_opponent']]['own_obs'][0:2] - obs[d]['own_obs'][0:2])*30*.6
            temp_v = np.array([temp_v[0], temp_v[1], 0])
            temp_v_mag = np.linalg.norm(temp_v)
            temp_v = temp_v / temp_v_mag
            temp[d] = np.array([temp_v[0], temp_v[1], 0, temp_v_mag])
        
        # temp[3] = obs[obs[3]['closest_opponent']]['own_obs'][0:2]- obs[3]['own_obs'][0:2]
        # temp[2] = np.array([0, 0, 0, 0])
        # temp[3] = np.array([0, 0, 0, 0])
       
       
        action = {i: temp[i] for i in range(NUM_DRONES)}
        obs, reward, done, info = test_env.step(action)

        total_reward_def += reward[2] + reward[3]
        total_reward_attacker += reward[0] + reward[1]

        print(
            f"rewards - attacker: {total_reward_attacker}\t defender: {total_reward_def}"
        )

        test_env.render()
        if OBS == ObservationType.KIN:
            for j in range(NUM_DRONES):
                logger.log(
                    drone=j,
                    timestamp=i / test_env.SIM_FREQ,
                    state=np.hstack(
                        [
                            obs[j]["own_obs"][0:3],
                            np.zeros(4),
                            obs[j]["own_obs"][3:15],
                            np.resize(action[j], (4)),
                        ]
                    ),
                    control=np.zeros(12),
                )
        sync(np.floor(i * test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        if done["__all__"]:
            obs = test_env.reset()  # OPTIONAL EPISODE HALT
            total_reward_def = 0
            total_reward_attacker = 0
    test_env.close()
    # logger.save_as_csv("ma")  # Optional CSV save
    logger.plot()

    #### Shut down Ray #########################################
    ray.shutdown()
