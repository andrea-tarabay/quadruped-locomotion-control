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

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results
import time

######################################################################################
# CONFIG TO BE TESTED
######################################################################################
LEARNING_ALG = "PPO"

# interm_dir = "./lr-final-project/weights/"
interm_dir = "./weights/"

# log_dir = interm_dir + 'RL_STAIRS'
log_dir = interm_dir + 'RL_STAIRS_HISTORY'

# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CPG",
               "task_env": "STAIRS_TASK",
               "observation_space_mode": "STAIRS_OBS_CPG_HISTORY", # STAIRS_OBS_CPG STAIRS_OBS_CPG_HISTORY
               "terrain":  "STAIRS", # ("STAIRS", "SLOPES", "RANDOM")
               "add_noise": True
               }

env_config['render'] = True
env_config['record_video'] = False


last_CPG_action = np.zeros(8)
last_torques = np.zeros(12)
_current_stair_idx = 0
_stair_thresholds = []

######################################################################################
#  REWARD FUNCTION TO BE ANALYZED
######################################################################################
def reward_function(body_lin_vel,
                    body_pos     ,
                    rpy          ,
                    quaternions  ,
                    torques      ,
                    velocities   ,
                    prev_action):
    global last_CPG_action, _current_stair_idx, _stair_thresholds, env_config
    # Reward moving forward
    vel_tracking_reward = 0.3 * np.clip(body_lin_vel[0], 0.2, 0.5)

    # minimize yaw (go straight)
    orientation_penalty = -0.4 * (abs(rpy[0]) + 1.3*abs(rpy[2]))
    vertical_reward = 0.1 * max(0, np.tanh((body_pos[2] - 0.34)/0.05))

    # don't drift laterally 
    drift_penalty = -0.1 * abs(body_pos[1]) 

    # Smooth the actions
    delta_action = last_CPG_action - prev_action
    last_CPG_action = prev_action
    action_rate_penalty = -0.02 * (np.sqrt(np.mean(np.square(delta_action))))
    action_rate_penalty = 0

    # Minimize energy 
    energy_penalty = 0 
    for tau,vel in zip(torques,velocities):
        energy_penalty += np.abs(np.dot(tau,vel)) * 0.001
    energy_penalty *= -0.1

    quaternions_pen = - 0.2 *  np.linalg.norm(quaternions - np.array([0,0,0,1]))

    reward = max(0, vel_tracking_reward + drift_penalty + energy_penalty + action_rate_penalty + quaternions_pen + vertical_reward + orientation_penalty)

    rewad_and_components = [reward, vel_tracking_reward, drift_penalty, energy_penalty, action_rate_penalty, quaternions_pen, vertical_reward, orientation_penalty]
    components_names = ["velocity r", "drift p", "energy p", "action p", "quat p", "vertical r", "orientation p"]
    return rewad_and_components, components_names

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
# monitor_results = load_results(log_dir)
# print(monitor_results)
# plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
# plt.show()

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [DONE] initialize arrays to save data from simulation
track_base_positions = []
track_base_velocities = []
track_foot_forces = []
track_cpg_r = []
track_cpg_dr = []
track_cpg_theta = []
track_cpg_dtheta = []
track_orientations = []
track_ang_vel = []
track_rewards = []
track_prev_actions = []
track_feet_pos = []
track_torques = []
track_velocities = []

track_energy = 0
prev_i = 0
t1 = time.time()

for i in range(50000):
    action, _states = model.predict(obs,deterministic=False)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    # Take the logs
    body_lin_vel = env.envs[0].env.robot.GetBaseLinearVelocity()
    body_ang_vel = env.envs[0].env.robot.GetBaseAngularVelocity()
    body_pos     = env.envs[0].env.robot.GetBasePosition()
    rpy          = env.envs[0].env.robot.GetBaseOrientationRollPitchYaw()
    quaternions  = env.envs[0].env.robot.GetBaseOrientation()
    foot_forces  = env.envs[0].env.robot.GetContactInfo()[2]
    cpg_r        = env.envs[0].env._cpg.get_r()
    cpg_dr       = env.envs[0].env._cpg.get_dr()
    cpg_theta    = env.envs[0].env._cpg.get_theta()
    cpg_dtheta   = env.envs[0].env._cpg.get_dtheta()
    torques      = env.envs[0].env.robot.GetMotorTorques()
    velocities   = env.envs[0].env.robot.GetMotorVelocities()
    period       = env.envs[0].env._MAX_EP_LEN
    # dt           = env.envs[0].env._time_step
    dt = time.time() - t1
    t1 = time.time()
    prev_action  = env.envs[0].env._last_action
    energy_consumed = 0
    for tau,vel in zip(torques,velocities):
      energy_consumed += np.abs(np.dot(tau,vel)) * dt
    
    feet_pos   = []

    # _current_stair_idx = env.envs[0].env._current_stair_idx
    # _stair_thresholds = env.envs[0].env._stair_thresholds[:]
    # _stair_reward = env.envs[0].env._stair_reward

    feet_pos = []
    for k in range(4):
        _j, fp = env.envs[0].env.robot.ComputeJacobianAndPosition(k)
        feet_pos.append(fp)

    track_base_positions.append(body_pos)
    track_base_velocities.append(body_lin_vel)
    track_ang_vel.append(body_ang_vel)
    track_foot_forces.append(foot_forces)
    track_cpg_dr.append(cpg_dr)
    track_cpg_theta.append(cpg_theta)
    track_cpg_dtheta.append(cpg_dtheta)
    track_cpg_r.append(cpg_r)
    track_orientations.append(rpy)
    track_prev_actions.append(prev_action)
    track_feet_pos.append(feet_pos)
    track_torques.append(torques)
    track_velocities.append(velocities)
    track_energy += energy_consumed

    # ----- analyze reward components -----
    reward_components, components_names = reward_function(body_lin_vel,
                                                          body_ang_vel ,
                                                          body_pos     ,
                                                          rpy          ,
                                                          quaternions  ,
                                                          foot_forces  ,
                                                          torques      ,
                                                          velocities   ,
                                                          dt           ,
                                                          prev_action)
    track_rewards.append(reward_components)
    
    if dones:
        # Only plot stable episodes
        if i - prev_i > 500:
            # remove the last element of each array
            track_base_positions    = track_base_positions[:-1]
            track_base_velocities   = track_base_velocities[:-1]
            track_cpg_r             = track_cpg_r[:-1]
            track_cpg_dr            = track_cpg_dr[:-1]
            track_cpg_theta         = track_cpg_theta[:-1]
            track_cpg_dtheta        = track_cpg_dtheta[:-1]
            track_orientations      = track_orientations[:-1]
            track_ang_vel           = track_ang_vel[:-1]
            track_rewards           = track_rewards[:-1]
            track_prev_actions      = track_prev_actions[:-1]
            feet_pos                = feet_pos[:-1]
            track_torques           = track_torques[:-1]
            track_velocities        = track_velocities[:-1]

            # Plot the 4 feet positions in 2D in one
            track_feet_pos_np = np.array(track_feet_pos)
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Feet trajectories', fontsize=16)
            
            foot_names = ["Front Right", "Front Left", "Rear Right", "Rear Left"]
            
            for f_idx in range(4):
                row = f_idx // 2
                col = f_idx % 2
                ax = axs[row, col]
                
                x_pos = track_feet_pos_np[:, f_idx, 0]
                z_pos = track_feet_pos_np[:, f_idx, 2]
                
                ax.plot(x_pos, z_pos, linewidth=1.5, color='blue', alpha=0.8)
                
                ax.set_title(f'Foot: {foot_names[f_idx]}')
                ax.set_xlabel('Position X [m]')
                ax.set_ylabel('Position Z [m]')
                
                ax.grid(True, linestyle='--', alpha=0.6)

                ax.set_aspect('equal') 

            plt.tight_layout()
            # ----- AI GENERATED -----
            # Plot body velocity
            track_base_velocities = np.array(track_base_velocities)
            plt.figure()
            plt.plot(track_base_velocities[:,0], label='Vx')
            plt.plot(track_base_velocities[:,1], label='Vy')
            plt.plot(track_base_velocities[:,2], label='Vz')
            plt.title('Base Linear Velocities')
            plt.xlabel('Timestep')
            plt.ylabel('Velocity (m/s)')
            plt.legend()
            plt.grid()

            # Plot position in XY plane
            track_base_positions = np.array(track_base_positions)
            plt.figure()
            plt.plot(track_base_positions[:,0], track_base_positions[:,1])
            plt.title('Base Position in XY Plane')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.grid()

            # # Plot position in XZ plane
            plt.figure()
            plt.plot(track_base_positions[:,0], track_base_positions[:,2])
            plt.title('Base Position in XZ Plane')
            plt.xlabel('X Position')
            plt.ylabel('Z Position')
            plt.grid()

            # # Plot CPG states
            track_cpg_r = np.array(track_cpg_r)
            track_cpg_dr = np.array(track_cpg_dr)
            track_cpg_theta = np.array(track_cpg_theta)
            track_cpg_dtheta = np.array(track_cpg_dtheta)
            plt.figure()
            plt.subplot(4,1,1)
            plt.plot(track_cpg_r)
            plt.title('CPG r')
            plt.subplot(4,1,2)
            plt.plot(track_cpg_dr)
            plt.title('CPG dr')
            plt.subplot(4,1,3)
            plt.plot(track_cpg_theta)
            plt.title('CPG theta')
            plt.subplot(4,1,4)
            plt.plot(track_cpg_dtheta)
            plt.title('CPG dtheta')

            # Plot reward components
            track_rewards = np.array(track_rewards)
            plt.figure()
            plt.plot(track_rewards[:,0], 'r', label=f'Reward', linewidth=3,)

            for each_comp in range(1, track_rewards.shape[1]):
                plt.plot(track_rewards[:,each_comp], label=components_names[each_comp-1])

            plt.title('Reward Components Over Episode')
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
            plt.legend(loc='lower left', fontsize='small')
            plt.grid()
            # ----- AI GENERATED -----


        prev_i = i
        x, y, _ = info[0]['base_pos']
        total_mass = np.sum(env.envs[0].env.robot.GetTotalMassFromURDF())
        cot = track_energy / (np.linalg.norm(np.array([x, y]) * total_mass * 9.81))
        print('Total Cost of Transport, ', cot)
        print("Mean velocity: ", np.linalg.norm(np.array([x, y]))/period," m/s")
        plt.show()

        track_base_positions    = []
        track_base_velocities   = []
        track_cpg_r             = []
        track_cpg_dr            = []
        track_cpg_theta         = []
        track_cpg_dtheta        = []
        track_orientations      = []
        track_ang_vel           = []
        track_rewards           = []
        track_prev_actions      = []
        track_feet_pos = []
        track_torques = []
        track_velocities = []
        track_energy = 0

        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0