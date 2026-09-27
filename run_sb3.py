# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

"""
Run stable baselines 3 on quadruped env 
Check the documentation! https://stable-baselines3.readthedocs.io/en/master/
"""

# misc
import os
from datetime import datetime

# stable baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

# utils
from utils.utils import CheckpointCallback
from utils.file_utils import get_latest_model

# gym environment
from env.quadruped_gym_env import QuadrupedGymEnv
import torch.nn as nn

def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

LEARNING_ALG = "PPO"
LOAD_NN = True
NUM_ENVS = 128
USE_GPU = True
ADVANCED = True # activate 512-256-128 achitecture

total_timesteps = 5_000_000

env_configs = {"motor_control_mode":"CPG", # CARTESIAN_PD, CPG
               "task_env": "STAIRS_TASK", #  FWD_LOCOMOTION STAIRS_TASK
               "observation_space_mode": "STAIRS_OBS_CPG_HISTORY", #LR_COURSE_OBS_CPG LR_COURSE_OBS_PD STAIRS_OBS_CPG STAIRS_OBS_CPG_HISTORY
               "terrain": "STAIRS", # ("STAIRS", "SLOPES", "GAPS", "RANDOM", None)
               "add_noise": False,
               "test_flagrun": False}

if USE_GPU and LEARNING_ALG=="SAC":
    gpu_arg = "auto" 
else:
    gpu_arg = "cpu"

if LOAD_NN:
    interm_dir = "./logs/intermediate_models/"
    log_dir = interm_dir + '121925010848' # add path
    
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    model_name = get_latest_model(log_dir)


# # directory to save policies and normalization parameters
SAVE_PATH = './logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
os.makedirs(SAVE_PATH, exist_ok=True)

# checkpoint to save policy network periodically
checkpoint_callback = CheckpointCallback(save_freq=(int)((total_timesteps/20)/NUM_ENVS), save_path=SAVE_PATH,name_prefix='rl_model', verbose=2)

# Multi-layer perceptron (MLP) policy of two layers of size _,_ each with tanh activation function
if ADVANCED:
    policy_kwargs = dict(
        activation_fn=nn.ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
    )
else: 
    policy_kwargs = dict(net_arch=[256,256]) # act_fun=tf.nn.tanh

# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

n_steps = 216 # per env, with 216 envs -> 128*216 steps

# learning_rate = linear_schedule(3e-4)
learning_rate = lambda _: 1e-4
ppo_config = {  "gamma":0.99, 
                "n_steps": n_steps, 
                "ent_coef":0.0,
                "learning_rate":learning_rate, 
                "vf_coef":0.5,
                "max_grad_norm":0.5, 
                "gae_lambda":0.95, 
                "batch_size":256,
                "n_epochs":10, 
                "clip_range":0.2, 
                "clip_range_vf":1,
                "verbose":1, 
                "tensorboard_log":None, 
                "_init_setup_model":True, 
                "policy_kwargs":policy_kwargs,
                "device": gpu_arg}

# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
sac_config={"learning_rate":1e-4,
            "buffer_size":300000,
            "batch_size":256,
            "ent_coef":'auto', 
            "gamma":0.99, 
            "tau":0.005,
            "train_freq":1, 
            "gradient_steps":1,
            "learning_starts": 10000,
            "verbose":1, 
            "tensorboard_log":None,
            "policy_kwargs": policy_kwargs,
            "seed":None, 
            "device": gpu_arg}

if LOAD_NN:
    print(f"\n==================================================")
    print(f" LOADING MODEL for FINE-TUNING")
    print(f" Source: {model_name}")
    print(f"==================================================\n")
    
    env = lambda: QuadrupedGymEnv(**env_configs)
    env = make_vec_env(env, monitor_dir=SAVE_PATH, n_envs=NUM_ENVS)
    
    print(f"Loading VecNormalize stats from {stats_path}...")
    env = VecNormalize.load(stats_path, env)
    env.training = True
    env.norm_reward = False
    
    print("Loading Weights with FORCE-OVERRIDE")
    
    if LEARNING_ALG == "PPO":
        model = PPO.load(
            model_name, 
            env=env, 
            custom_objects={
                "learning_rate": linear_schedule(1e-5),
                # "learning_rate": lambda _: 2e-5,
                "clip_range": lambda _: 0.2,
                # "ent_coef":0.0,
            }
        )
    elif LEARNING_ALG == "SAC":
        model = SAC.load(model_name, env, custom_objects={"learning_rate": linear_schedule(5e-5)})

else:
    print(f"\n==================================================")
    print(f" CREATING NEW MODEL (Scratch Training)")
    print(f"==================================================\n")
    
    env = lambda: QuadrupedGymEnv(**env_configs)
    env = make_vec_env(env, monitor_dir=SAVE_PATH, n_envs=NUM_ENVS)
    
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)
    
    if LEARNING_ALG == "PPO":
        model = PPO('MlpPolicy', env, **ppo_config)
    elif LEARNING_ALG == "SAC":
        model = SAC('MlpPolicy', env, **sac_config)
    else:
        raise ValueError(LEARNING_ALG + ' not implemented')

info_file_path = os.path.join(SAVE_PATH, "training_info.txt")
with open(info_file_path, "w") as f:
    f.write("=== Training Quadruped ===\n")
    f.write(f"Date: {datetime.now()}\n\n")
    
    f.write(f"Learning algorithm: {LEARNING_ALG}\n")
    f.write(f"Load previous model: {LOAD_NN}\n")
    f.write(f"Number of environments: {NUM_ENVS}\n")
    f.write(f"Use GPU: {USE_GPU}\n\n")
    
    f.write("=== Environment Configs ===\n")
    for key, val in env_configs.items():
        f.write(f"{key}: {val}\n")
    f.write("\n")
    
    if LEARNING_ALG == "PPO":
        f.write("=== PPO Hyperparameters ===\n")
        for key, val in ppo_config.items():
            f.write(f"{key}: {val}\n")
        
        if LOAD_NN:
             f.write("\n[OVERRIDE] Loaded with Fine-Tuning Params:\n")
    
    f.write("\n")
    f.write(f"Policy network architecture: {policy_kwargs['net_arch']}\n")
    f.write(f"Total timesteps: {total_timesteps} steps\n")
    f.write(f"Save path: {SAVE_PATH}\n")
    
    f.write("\n=== Gym Env source code ===\n\n")
    try:
        with open('./lr-final-project/env/quadruped_gym_env.py', 'r') as env_file:
            env_code = env_file.read()
            f.write(env_code)
    except:
        f.write("Could not read env source code.")

print(f"Training info saved in: {info_file_path}")


model.learn(total_timesteps=total_timesteps, log_interval=1,callback=checkpoint_callback, progress_bar=True)

model.save( os.path.join(SAVE_PATH, "rl_model" ) ) 
env.save(os.path.join(SAVE_PATH, "vec_normalize.pkl" )) 

if LEARNING_ALG == "SAC":
    model.save_replay_buffer(os.path.join(SAVE_PATH,"off_policy_replay_buffer"))