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

# stable-baselines3
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env  # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


######################################################################################
# CONFIG TO BE TESTED
######################################################################################
LEARNING_ALG = "PPO"
interm_dir = "./weights/"

# log_dir = interm_dir + "VEL_fixed"
# log_dir = interm_dir + "VEL_slow"
log_dir = interm_dir + "VEL_faster"
DESIRED_VEL_X = 1.5

# evaluation params (assignment-style: one rollout, show plots + metrics)
WARMUP_SEC = 0
CONTACT_FORCE_THRESH = 10.0

# folder for plots
plot_dir = os.path.join(log_dir, "VEL_eval_plots")
os.makedirs(plot_dir, exist_ok=True)

env_config = {
    "motor_control_mode": "CARTESIAN_PD",
    "task_env": "DESIRED_VEL_TRACKING",
    "observation_space_mode": "LR_COURSE_OBS_PD",
    "terrain": None,
    "add_noise": False,
    "test_flagrun": False,
    "desired_vel_x": float(DESIRED_VEL_X),
}
env_config["render"] = True
env_config["record_video"] = False


######################################################################################
# YOUR REWARD (POST-HOC, FOR PLOTS)
######################################################################################
def reward_function_desired_vel_tracking(
    vx, vy, roll, pitch, wz,
    des_vx,
    torques, velocities, dt_control,
    action, last_action,
    ys, y_nom
):
    v_err = vx - des_vx
    r_track = np.exp(-4.0 * (v_err ** 2))

    p_backward = max(0.0, -vx)
    p_rp = (roll ** 2 + pitch ** 2)
    p_lat = (vy ** 2)
    p_yawrate = (wz ** 2)

    # energy per CONTROL step
    energy = float(np.sum(np.abs(torques * velocities)) * dt_control)

    if last_action is None:
        p_smooth = 0.0
    else:
        da = action - last_action
        p_smooth = float(np.mean(da ** 2))

    alive = 0.05

    ys = np.array(ys).reshape(4,)
    y_nom = np.array(y_nom).reshape(4,)

    p_nom = float(np.mean((ys - y_nom) ** 2))
    front_w = float(np.mean(np.abs(ys[0:2])))
    hind_w = float(np.mean(np.abs(ys[2:4])))
    p_fh = float((front_w - hind_w) ** 2)
    p_lr = float((ys[0] + ys[1]) ** 2 + (ys[2] + ys[3]) ** 2)

    w_nom = 5.0
    w_fh = 10.0
    w_lr = 2.0
    stance_pen = w_nom * p_nom + w_fh * p_fh + w_lr * p_lr

    total = (
        1.0 * r_track
        + alive
        - 1.0 * p_backward
        - 1.0 * p_lat
        - 0.5 * p_yawrate
        - 0.5 * p_rp
        - 0.002 * energy
        - 0.05 * p_smooth
        - stance_pen
    )

    comps = [
        total,
        1.0 * r_track,
        alive,
        -1.0 * p_backward,
        -1.0 * p_lat,
        -0.5 * p_yawrate,
        -0.5 * p_rp,
        -0.002 * energy,
        -0.05 * p_smooth,
        -1.0 * stance_pen,
        front_w,
        hind_w,
    ]
    names = [
        "total_posthoc",
        "r_track",
        "alive",
        "-p_backward",
        "-p_lat",
        "-0.5*p_yawrate",
        "-0.5*p_rp",
        "-0.002*energy",
        "-0.05*p_smooth",
        "-stance_pen",
        "front_w",
        "hind_w",
    ]
    return comps, names


######################################################################################
# GAIT METRICS FROM CONTACTS
######################################################################################
def compute_gait_metrics_from_contacts(contacts, dt_control):
    out_per_leg = []
    for leg in range(4):
        c = contacts[:, leg].astype(int)
        td = np.where((c[1:] == 1) & (c[:-1] == 0))[0] + 1  # 0->1
        lo = np.where((c[1:] == 0) & (c[:-1] == 1))[0] + 1  # 1->0

        if len(td) < 2:
            out_per_leg.append({"duty": np.nan, "T_stride": np.nan, "T_stance": np.nan, "T_swing": np.nan})
            continue

        stride_times = np.diff(td) * dt_control
        stance_times = []
        swing_times = []

        for k in range(len(td) - 1):
            td_k = td[k]
            td_next = td[k + 1]

            lo_after = lo[lo > td_k]
            if len(lo_after) == 0:
                continue
            lo_k = lo_after[0]
            if lo_k >= td_next:
                continue

            T_stance = (lo_k - td_k) * dt_control
            T_stride = (td_next - td_k) * dt_control
            T_swing = T_stride - T_stance

            if T_stride > 1e-6 and T_stance >= 0 and T_swing >= 0:
                stance_times.append(T_stance)
                swing_times.append(T_swing)

        if len(stance_times) == 0:
            out_per_leg.append({"duty": np.nan, "T_stride": np.nan, "T_stance": np.nan, "T_swing": np.nan})
        else:
            T_stride = float(np.mean(stride_times))
            T_stance = float(np.mean(stance_times))
            T_swing = float(np.mean(swing_times))
            duty = float(T_stance / T_stride) if T_stride > 1e-6 else np.nan
            out_per_leg.append({"duty": duty, "T_stride": T_stride, "T_stance": T_stance, "T_swing": T_swing})

    avg = {
        "duty": float(np.nanmean([m["duty"] for m in out_per_leg])),
        "T_stride": float(np.nanmean([m["T_stride"] for m in out_per_leg])),
        "T_stance": float(np.nanmean([m["T_stance"] for m in out_per_leg])),
        "T_swing": float(np.nanmean([m["T_swing"] for m in out_per_leg])),
    }
    return out_per_leg, avg


######################################################################################
# LOAD MODEL + TRAINING CURVES
######################################################################################
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)

# learning curves
try:
    monitor_results = load_results(log_dir)
    print(monitor_results)
    plot_results([log_dir], 10e10, "timesteps", LEARNING_ALG + " ")
    plt.show()

    for i, fignum in enumerate(plt.get_fignums()):
        fig = plt.figure(fignum)
        fig.savefig(os.path.join(plot_dir, f"learning_curve_{i+1}.png"), dpi=200, bbox_inches="tight")
    plt.close("all")
except Exception as e:
    print("Could not load/plot training curves.")
    print("Reason:", e)

# env
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False
env.norm_reward = False

# model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")


######################################################################################
# RUN ONE EPISODE AND ANALYZE IT
######################################################################################
obs = env.reset()
env0 = env.envs[0].env

# IMPORTANT: env uses sim_dt * action_repeat as control dt
sim_dt = float(getattr(env0, "_time_step", 0.001))
action_repeat = int(getattr(env0, "_action_repeat", getattr(env0, "action_repeat", 10)))
dt_control = sim_dt * action_repeat

warmup_steps = int(WARMUP_SEC / dt_control)

# nominal y
try:
    y_nom = env0._robot_config.NOMINAL_FOOT_POS_LEG_FRAME.reshape(4, 3)[:, 1]
except Exception:
    y_nom = np.zeros(4)

# robot mass
robot_mass = np.nan
try:
    robot_mass = float(np.sum(env0.robot.GetTotalMassFromURDF()))
except Exception:
    try:
        robot_mass = float(env0.robot.GetTotalMass())
    except Exception:
        robot_mass = np.nan

g = 9.81

print(f"sim_dt={sim_dt}, action_repeat={action_repeat}, dt_control={dt_control}")
print(f"warmup_steps={warmup_steps}")

# buffers
track_t = []
track_base_positions = []
track_base_velocities = []
track_ang_vel = []
track_orientations = []
track_foot_forces = []
track_contacts = []
track_power = []
track_energy_cum = []
track_action_rate = []
track_ys = []
track_rewards_env = []
track_rewards_posthoc = []

episode_reward = 0.0
energy_cum = 0.0
last_action = None

terminal_base_pos = None
episode_len_steps = 0

# run until done
for step in range(20000):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)

    r_env = float(rewards[0])
    episode_reward += r_env

    # VecEnv: dones is an array([True/False])
    done = bool(dones[0])

    # If done, DON'T query env0.robot state anymore (it may already be reset)
    if done:
        episode_len_steps = step + 1
        try:
            terminal_base_pos = info[0]["base_pos"]
        except Exception:
            terminal_base_pos = None

        print("\nEpisode done.")
        print("episode_reward", episode_reward)
        print("Final base position", terminal_base_pos)
        print("Episode length (steps)", episode_len_steps)
        break

    # safe to query states here (not terminal step)
    body_lin_vel = np.array(env0.robot.GetBaseLinearVelocity(), dtype=float)
    body_ang_vel = np.array(env0.robot.GetBaseAngularVelocity(), dtype=float)
    body_pos     = np.array(env0.robot.GetBasePosition(), dtype=float)
    rpy          = np.array(env0.robot.GetBaseOrientationRollPitchYaw(), dtype=float)

    # foot forces + contacts
    try:
        foot_forces = np.array(env0.robot.GetContactInfo()[2], dtype=float).reshape(4,)
    except Exception:
        foot_forces = np.zeros(4)
    contacts = (foot_forces > CONTACT_FORCE_THRESH).astype(int)

    # torques/velocities
    try:
        torques    = np.array(env0.robot.GetMotorTorques(), dtype=float)
        velocities = np.array(env0.robot.GetMotorVelocities(), dtype=float)
    except Exception:
        torques    = np.zeros(12)
        velocities = np.zeros(12)

    power = float(np.sum(np.abs(torques * velocities)))
    energy_cum += power * dt_control

    # leg-frame y
    ys = []
    for leg in range(4):
        try:
            _, p = env0.robot.ComputeJacobianAndPosition(leg)
            ys.append(float(p[1]))
        except Exception:
            ys.append(0.0)

    a = np.array(action).reshape(-1)
    if last_action is None:
        action_rate = 0.0
    else:
        action_rate = float(np.mean((a - last_action) ** 2))

    posthoc_comps, posthoc_names = reward_function_desired_vel_tracking(
        vx=float(body_lin_vel[0]),
        vy=float(body_lin_vel[1]),
        roll=float(rpy[0]),
        pitch=float(rpy[1]),
        wz=float(body_ang_vel[2]),
        des_vx=float(DESIRED_VEL_X),
        torques=torques,
        velocities=velocities,
        dt_control=dt_control,
        action=a,
        last_action=last_action,
        ys=ys,
        y_nom=y_nom
    )

    # log
    track_t.append(step * dt_control)
    track_base_positions.append(body_pos)
    track_base_velocities.append(body_lin_vel)
    track_ang_vel.append(body_ang_vel)
    track_orientations.append(rpy)
    track_foot_forces.append(foot_forces)
    track_contacts.append(contacts)
    track_power.append(power)
    track_energy_cum.append(energy_cum)
    track_action_rate.append(action_rate)
    track_ys.append(ys)
    track_rewards_env.append(r_env)
    track_rewards_posthoc.append(posthoc_comps)

    last_action = a.copy()

# arrays
t_arr = np.array(track_t)
base_pos = np.array(track_base_positions)
base_vel = np.array(track_base_velocities)
ang_vel  = np.array(track_ang_vel)
rpy_arr  = np.array(track_orientations)
foot_forces_arr = np.array(track_foot_forces)
contacts_arr = np.array(track_contacts)
power_arr = np.array(track_power)
energy_cum_arr = np.array(track_energy_cum)
action_rate_arr = np.array(track_action_rate)
ys_arr = np.array(track_ys)
rew_env_arr = np.array(track_rewards_env)
rew_post_arr = np.array(track_rewards_posthoc)

if len(t_arr) < 2:
    print("Not enough data collected to compute metrics/plots.")
    sys.exit(0)

# metrics after warmup
w0 = min(warmup_steps, len(t_arr) - 1)

avg_vx = float(np.mean(base_vel[w0:, 0]))

# IMPORTANT FIX: use terminal base_pos from info (NOT env0 state), otherwise dist can become ~0
x_warm = float(base_pos[w0, 0])
if terminal_base_pos is not None:
    x_end = float(terminal_base_pos[0])
else:
    x_end = float(base_pos[-1, 0])

dist = max(1e-6, x_end - x_warm)

E = float(np.sum(power_arr[w0:] * dt_control))

if np.isfinite(robot_mass) and robot_mass > 0:
    cot = float(E / (robot_mass * g * dist))
else:
    cot = np.nan

per_leg_gait, avg_gait = compute_gait_metrics_from_contacts(contacts_arr[w0:], dt_control)

print("\n================ REPORT VALUES ================")
print(f"Desired vel_x: {DESIRED_VEL_X:.3f} m/s")
print(f"Avg steady-state vx (after {WARMUP_SEC:.2f}s warmup): {avg_vx:.6f} m/s")
print(f"sim_dt: {sim_dt}, action_repeat: {action_repeat}, dt_control: {dt_control}")
print(f"Robot mass: {robot_mass}")
print(f"Distance after warmup: {dist}")
print(f"Energy after warmup: {E}")
print(f"CoT: {cot}")
print("\nPer-leg gait metrics:")
for leg in range(4):
    print(f"leg{leg}: {per_leg_gait[leg]}")
print("\nAveraged gait metrics:")
print(avg_gait)
print("================================================\n")

with open(os.path.join(log_dir, "VEL_eval_metrics.txt"), "w") as f:
    f.write(f"Desired vel_x: {DESIRED_VEL_X}\n")
    f.write(f"Avg steady-state vx: {avg_vx}\n")
    f.write(f"Warmup_sec: {WARMUP_SEC}\n")
    f.write(f"sim_dt: {sim_dt}\n")
    f.write(f"action_repeat: {action_repeat}\n")
    f.write(f"dt_control: {dt_control}\n")
    f.write(f"Robot mass: {robot_mass}\n")
    f.write(f"Distance after warmup: {dist}\n")
    f.write(f"Energy after warmup: {E}\n")
    f.write(f"CoT: {cot}\n\n")
    f.write("Per-leg gait metrics:\n")
    for leg in range(4):
        f.write(f"leg{leg}: {per_leg_gait[leg]}\n")
    f.write("\nAveraged gait metrics:\n")
    f.write(str(avg_gait) + "\n")

# ---- plots (saved as PNG in one folder) ----

plt.figure()
plt.plot(t_arr, base_vel[:, 0], label="vx")
plt.plot(t_arr, np.ones_like(t_arr) * DESIRED_VEL_X, "--", label="v_des")
plt.plot(t_arr, base_vel[:, 1], label="vy")
plt.title("Velocity tracking")
plt.xlabel("time (s)")
plt.ylabel("m/s")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "VEL_vel_tracking.png"), dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(t_arr, rpy_arr[:, 0], label="roll")
plt.plot(t_arr, rpy_arr[:, 1], label="pitch")
plt.plot(t_arr, ang_vel[:, 2], label="yaw rate wz")
plt.title("Stability + yaw rate")
plt.xlabel("time (s)")
plt.ylabel("rad / rad/s")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "VEL_stability_yawrate.png"), dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(base_pos[:, 0], base_pos[:, 1])
plt.title("Base position (XY)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid()
plt.savefig(os.path.join(plot_dir, "VEL_base_xy.png"), dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(base_pos[:, 0], base_pos[:, 2])
plt.title("Base position (XZ)")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.grid()
plt.savefig(os.path.join(plot_dir, "VEL_base_xz.png"), dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(t_arr, power_arr, label="power = sum|tau*qdot|")
plt.plot(t_arr, energy_cum_arr, label="cumulative energy")
plt.title("Energy proxy")
plt.xlabel("time (s)")
plt.ylabel("W / J")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "VEL_energy.png"), dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(t_arr, action_rate_arr, label="mean (a_t - a_{t-1})^2")
plt.title("Action smoothness")
plt.xlabel("time (s)")
plt.ylabel("action-rate")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "VEL_action_smoothness.png"), dpi=200, bbox_inches="tight")

plt.figure()
for leg in range(4):
    plt.plot(t_arr, contacts_arr[:, leg] + 1.2 * leg, drawstyle="steps-post", label=f"leg{leg}")
plt.title("Foot contacts (offset per leg)")
plt.xlabel("time (s)")
plt.ylabel("contact + offset")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "VEL_contacts.png"), dpi=200, bbox_inches="tight")

plt.figure()
for leg in range(4):
    plt.plot(t_arr, foot_forces_arr[:, leg], label=f"leg{leg} force")
plt.title("Foot contact forces")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "VEL_foot_forces.png"), dpi=200, bbox_inches="tight")

front_w = np.mean(np.abs(ys_arr[:, 0:2]), axis=1)
hind_w  = np.mean(np.abs(ys_arr[:, 2:4]), axis=1)
plt.figure()
for leg in range(4):
    plt.plot(t_arr, ys_arr[:, leg], label=f"leg{leg} y")
plt.plot(t_arr, front_w, "--", label="front_w")
plt.plot(t_arr, hind_w, "--", label="hind_w")
plt.title("Foot lateral placement (leg frame y) + front/hind widths")
plt.xlabel("time (s)")
plt.ylabel("m")
plt.grid()
plt.legend()
plt.savefig(os.path.join(plot_dir, "VEL_stance_width.png"), dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(t_arr, rew_env_arr, label="env reward", linewidth=2)
plt.plot(t_arr, rew_post_arr[:, 0], label="posthoc total", linewidth=2)
for k in range(1, 11):
    plt.plot(t_arr, rew_post_arr[:, k], label=posthoc_names[k])
plt.title("Reward + components (post-hoc)")
plt.xlabel("time (s)")
plt.ylabel("reward")
plt.grid()
plt.legend(loc="best", fontsize=8)
plt.savefig(os.path.join(plot_dir, "VEL_reward_components.png"), dpi=200, bbox_inches="tight")

print("Saved all plots to:", plot_dir)
plt.show()
