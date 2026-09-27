# Quadruped Locomotion Control

**EPFL — Legged Robots**

This project investigates locomotion control for the **Unitree A1 quadruped** in PyBullet, comparing biologically inspired **Central Pattern Generators (CPGs)** with **Deep Reinforcement Learning (DRL)**.

The project includes:

- Coupled Hopf-oscillator CPGs for trot, walk, pace, and bound gaits
- Joint and Cartesian PD foot-trajectory control
- Virtual Model Control for improved stability and direction tracking
- PPO-based forward velocity tracking
- Hierarchical CPG-RL for stair climbing
- Curriculum learning and domain randomization for terrain robustness

📄 **[Full Project Report](report.pdf)**

---

## Project Overview

### CPG-Based Locomotion

Coupled **Hopf oscillators** generate coordinated rhythmic foot trajectories.  
The resulting Cartesian foot positions are tracked using joint and Cartesian PD control.

Virtual Model Control is added to improve:

- body attitude
- roll stability
- lateral motion
- heading correction

The CPG controller was evaluated across several gaits and speeds.

### Reinforcement Learning

Two PPO-based control approaches were investigated:

**Velocity Tracking:**  
A Cartesian-PD policy directly adjusts foot positions to track a desired forward velocity while maintaining stability and energy efficiency.

**Stair Climbing:**  
A hierarchical **CPG-RL** policy modulates the CPG amplitude and frequency parameters, combining structured rhythmic motion with learned terrain adaptation.

Curriculum learning and domain randomization were used to improve robustness to varying stair geometry and simulated physical parameters.

---

## Results

- RL velocity tracking at **1.5 m/s**
- CPG walking reached up to **2.6 m/s**
- CPG trotting reached up to **1.8 m/s**
- Robust traversal of randomized stair geometries
- Improved heading and roll stability using VMC
- Zero-shot qualitative generalization to slopes and irregular terrain

A history-aware policy using previous proprioceptive states was also explored to improve robustness under partial observability.

---

## Repository Structure

```text
quadruped-locomotion-control/
│
├── a1_description/
│   └── Unitree A1 robot model and simulation assets
│
├── cpg_plot_utils/
│   └── Plotting and analysis utilities for CPG experiments
│
├── env/
│   ├── configs_a1.py
│   ├── hopf_network.py
│   ├── quadruped.py
│   ├── quadruped_gym_env.py
│   └── quadruped_motor.py
│
├── utils/
│   └── Shared utilities for training and evaluation
│
├── videos/
│   └── Locomotion and terrain-navigation demonstrations
│
├── weights/
│   └── Trained PPO policies, normalization data, and evaluation results
│
├── run_cpg.py
│   └── CPG-based locomotion experiments
│
├── run_sb3.py
│   └── PPO training with Stable-Baselines3
│
├── load_vel_model.py
│   └── Evaluation of the velocity-tracking policy
│
├── load_stairs_model.py
│   └── Evaluation of the stair-climbing policy
│
├── requirements.txt
├── report.pdf
└── README.md
```

---

## Technologies

**Python · PyBullet · Stable-Baselines3 · PPO · Reinforcement Learning · CPG · Hopf Oscillators · Virtual Model Control · Quadruped Robotics**

---

## Report

For the full methodology, controller design, training procedure, experiments, and analysis:

📄 **[Read the full project report](report.pdf)**

---

## Team

This project was developed as a team project for the **EPFL Legged Robots** course.

- Andrea Tarabay
- Till Beyer
- Mattia Prandi

  
**Andrea Tarabay**  
MSc Robotics, EPFL
