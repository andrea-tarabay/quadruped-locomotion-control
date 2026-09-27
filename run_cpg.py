import time
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results


FOOT_Y = 0.0838 # this is the hip length 
SIDE_SIGN = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)
TIME_STEP = 0.001


def run_cpg(plots=None, add_cartesian_pd=False, hopf_params=None,
            time_horizon=5, k_att_vmc=None, k_dir_vmc=None, k_lat_vmc=None):
  """
  Run CPG with joint PD control and optional Cartesian PD control. 
  Args:
    plots (list): list of strings specifying which plots to generate. ["states", "foot_position", "joint_angles"]
  """
  env = QuadrupedGymEnv(
                      render=True,              # visualize
                      on_rack=False,              # useful for debugging! 
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      record_video=False
                      )

  # initialize Hopf Network, supply gait
  if hopf_params:
    cpg = HopfNetwork(**hopf_params)
  else:
    cpg = HopfNetwork() # produces a time-varying foot trajectory, can be TROT, WALK, PACE, BOUND

  TEST_STEPS = int(time_horizon / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # initialize data structures to save CPG and robot states
  joint_pos = np.zeros((12, TEST_STEPS)) # actual
  joint_pos_desired = np.zeros((12, TEST_STEPS)) # desired
  joint_vel = np.zeros((12, TEST_STEPS))
  tau_data = np.zeros((12, TEST_STEPS))
  torque_data = np.zeros((12, TEST_STEPS))
  foot_desired = np.zeros((4, 3, TEST_STEPS)) # desired foot pos per leg
  foot_act= np.zeros((4, 3, TEST_STEPS)) # actual foot pos per leg
  cpg_x = np.zeros((4, TEST_STEPS)) # desired foot x pos from CPG
  cpg_z = np.zeros((4, TEST_STEPS)) # desired foot z pos from CPG
  amplitudes = np.zeros((4, TEST_STEPS)) # CPG amplitudes
  d_amplitudes = np.zeros((4, TEST_STEPS)) # CPG amplitude derivatives
  phases = np.zeros((4, TEST_STEPS)) # CPG phases
  d_phases = np.zeros((4, TEST_STEPS)) # CPG phase derivatives

  # inititialize interesting variables
  energy_consumed = 0.0
  body_position = np.zeros((3, TEST_STEPS))
  body_orientation = np.zeros((3, TEST_STEPS)) # roll, pitch, yaw
  converged_at = -1 # j index when CPG converged (r at 99%)


  ############## Sample Gains

  # joint PD gains
  kp=np.array([100]*3)
  kd=np.array([2]*3)

  # Cartesian PD gains, 300 5??
  kpCartesian = np.diag([700]*3)
  kdCartesian = np.diag([70]*3)

  for j in range(TEST_STEPS): #on every time step 
    # initialize torque array to send to motors
    action = np.zeros(12) 

    # get desired foot positions from CPG 
    xs,zs = cpg.update() #desired foot x and z positions for all 4 legs

    # get current motor angles and velocities for joint PD
    q = env.robot.GetMotorAngles() #robot state 12 joint angles 
    dq = env.robot.GetMotorVelocities() #12 joint velocities

    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)

      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i], SIDE_SIGN[i] * FOOT_Y, zs[i]]) #desired foot position in leg frame 

      # call inverse kinematics to get corresponding joint angles
      leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) #desired joint angles computed by IK from desired foot position 
  
      # Add joint PD contribution to tau for leg i (Equation 4)
      leg_dq_desired = np.zeros(3) 
      leg_q_current = q[3*i:3*i+3] # current joint angles for leg i
      leg_dq_current = dq[3*i:3*i+3] # current joint velocities for leg i
      tau += kp * (leg_q - leg_q_current) + kd * (leg_dq_desired - leg_dq_current) # joint PD contribution

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J_current, foot_pos_current = env.robot.ComputeJacobianAndPosition(i, leg_q_current)
      R_base = env.robot.GetBaseOrientationMatrix()
      has_ground_contact = env.robot.GetContactInfo()[3][i]
      roll, _, yaw = env.robot.GetBaseOrientationRollPitchYaw()
      _, y, _ = env.robot.GetBasePosition()

      if k_att_vmc and has_ground_contact: # att additidue virtual model controll
        tau += attitude_vmc(R_base, J_current, k_att_vmc, i)  

      if k_dir_vmc and has_ground_contact: # direction virtual model controller
        desired_yaw = 0.0
        delta_angle = desired_yaw - yaw
        desired_y = 0.0
        delta_y = desired_y - y
        tau += direction_vmc(delta_angle, delta_y, J_current, k_dir_vmc, i)
      
      if k_lat_vmc: # lateral virtual model controller, no ground contact required
        hip_angle = leg_q_current[0]
        delta_roll = roll - hip_angle
        tau += lateral_vmc(delta_roll, k_lat_vmc)

      # add Cartesian PD contribution
      if add_cartesian_pd:
        # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
        foot_pos_desired = leg_xyz
  
        # Get current foot velocity in leg frame (Equation 2)
        v_current = J_current @ leg_dq_current  # foot velocity

        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        if j == 0:
          v_des = np.zeros(3)
        else:
            prev_leg_xyz = np.array([cpg_x[i, j-1], SIDE_SIGN[i]*FOOT_Y, cpg_z[i, j-1]])
            v_des = (leg_xyz - prev_leg_xyz) / TIME_STEP

        F = kpCartesian @ (foot_pos_desired - foot_pos_current) + kdCartesian @ (v_des - v_current)
        tau += J_current.T @ F



      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 

    # Save any CPG or robot states
    joint_pos[:, j] = q
    joint_vel[:, j] = dq
    tau_data[:, j]  = action
    cpg_x[:, j]     = xs
    cpg_z[:, j]     = zs
    amplitudes[:, j] = cpg.get_r()
    d_amplitudes[:, j] = cpg.get_dr()
    phases[:, j]    = cpg.get_theta()
    d_phases[:, j]  = cpg.get_dtheta()
    
    energy_consumed += np.sum(np.abs(env.robot.GetMotorTorques()) * np.abs(dq)) * TIME_STEP
    torque_data[:, j] = env.robot.GetMotorTorques()

    body_position[:, j] = env.robot.GetBasePosition()
    body_orientation[:, j] = env.robot.GetBaseOrientationRollPitchYaw()

    for i in range(4): # this loop is for convenience, all the array could be filled above but we are lazy
      # actual at current q
      _, p_cur = env.robot.ComputeJacobianAndPosition(i, q[3*i:3*i+3]) #get position of the foot from computed joint angles 
      foot_act[i, :, j] = p_cur
      # desired at leg_q_des computed above (recompute or cache if you prefer)
      leg_xyz = np.array([xs[i], SIDE_SIGN[i]*FOOT_Y, zs[i]]) #current position of leg i 
      foot_desired[i, :, j] = leg_xyz
      leg_q_des = env.robot.ComputeInverseKinematics(i, leg_xyz)
      joint_pos_desired[3*i:3*i+3, j] = leg_q_des



  ##################################################### 
  # PLOTS
  #####################################################

  if "report" in plots:
    end_position_xy = body_position[:2, -1] - body_position[:2, 0]
    total_mass = np.sum(env.robot.GetTotalMassFromURDF())
    cost_of_transport = energy_consumed / (np.linalg.norm(end_position_xy) * total_mass * 9.81)
    avg_velocity = np.linalg.norm(end_position_xy) / (TEST_STEPS * TIME_STEP)
    
    converged_at = 500 # fix this
    if converged_at != -1:
      convergence_position_xy = body_position[:2, converged_at] - body_position[:2, 0]
      avg_velocity_after_conv = np.linalg.norm(end_position_xy - convergence_position_xy) / ((TEST_STEPS - converged_at) * TIME_STEP)
    else:
      convergence_position_xy = end_position_xy
      avg_velocity_after_conv = avg_velocity

    stance_duration = 2*np.pi / cpg.get_omega_stance()
    swing_duration = 2*np.pi / cpg.get_omega_swing()
    duty_cycle = stance_duration + swing_duration # aka stride
    duty_ratio = stance_duration / duty_cycle

    print("\n----- Performance Report -----")
    print(f"Simulated Time: {TEST_STEPS * TIME_STEP:.1f} s")
    print(f"Average Velocity: {avg_velocity:.3f} m/s")
    print(f"End Position XY: {end_position_xy}")
    print(f"Cost of Transport: {cost_of_transport:.3f}\n")

    if converged_at != -1:
      print(f"CPG Converged after t = {converged_at * TIME_STEP:.3f} s")
      print(f"Average Velocity after Convergence: {avg_velocity_after_conv:.3f} m/s")

    print(f"Stance Duration: {stance_duration:.3f} s")
    print(f"Swing Duration: {swing_duration:.3f} s")
    print(f"Duty Ratio: {duty_ratio:.3f}\n")
    print("--------- Parameters ---------")
    for key, value in hopf_params.items():
      if isinstance(value, float):
        print(f"{key}: {value:.3f}")
      else:
        print(f"{key}: {value}")
    print("------------------------------\n")

    

  leg_idx = 0  # 0: FR, 1: FL, 2: RR, 3: RL
  leg_from_idx = {0: 'FR', 1: 'FL', 2: 'RR', 3: 'RL'}

  if "foot_position" in plots:
    # plot actual and desired foot position in 2D space
    fig1 = plt.figure()
    plt.plot(foot_act[leg_idx, 0, :], foot_act[leg_idx, 2, :], label='actual', color='blue')
    plt.plot(foot_desired[leg_idx, 0, :], foot_desired[leg_idx, 2, :], 'k--', label='desired')
    plt.title(f'Foot trajectory ({leg_from_idx[leg_idx]})')
    plt.xlabel('x [m]'); plt.ylabel('z [m]'); plt.legend(); plt.axis('equal'); plt.tight_layout()
    plt.grid(True)

  if "torque" in plots:
    for i in range(torque_data.shape[0]):
      plt.plot(t, torque_data[i,:])
      plt.xlabel("time"); plt.ylabel("torque (Nm)"); plt.tight_layout()
      plt.grid(True)

  if "joint_angle" in plots:
    # Create a figure with a shared title and legend
    fig2, axs = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig2.suptitle(f'Joint Angles ({leg_from_idx[leg_idx]})')
    max_hip = np.max(joint_pos[3*leg_idx, :])
    min_hip = np.min(joint_pos[3*leg_idx, :])
    max_shoulder = np.max(joint_pos_desired[3*leg_idx+1, :])
    min_shoulder = np.min(joint_pos_desired[3*leg_idx+1, :])
    max_elbow = np.max(joint_pos_desired[3*leg_idx+2, :])
    min_elbow = np.min(joint_pos_desired[3*leg_idx+2, :])
    angle_range = max(max_hip - min_hip, max_shoulder - min_shoulder, max_elbow - min_elbow)

    # Plot HIP angles
    axs[0].plot(t, joint_pos[3*leg_idx, :], color='blue', label='Actual')
    axs[0].plot(t, joint_pos_desired[3*leg_idx, :], 'k--', label='Desired')
    axs[0].grid(True)
    axs[0].set_ylabel('Hip Angle (rad)')
    axs[0].set_xlim(0, max(t))
    axs[0].set_ylim(0.5*(max_hip + min_hip - 1.1*angle_range), 0.5*(max_hip + min_hip + 1.1*angle_range))

    # Plot SHOULDER angles
    axs[1].plot(t, joint_pos[3*leg_idx+1, :], color='blue', label="Actual")
    axs[1].plot(t, joint_pos_desired[3*leg_idx+1, :], 'k--', label="Desired")
    axs[1].grid(True)
    axs[1].set_ylabel('Shoulder Angle (rad)')
    axs[1].set_ylim(0.5*(max_shoulder + min_shoulder - 1.1*angle_range), 0.5*(max_shoulder + min_shoulder + 1.1*angle_range))

    # Plot ELBOW angles
    axs[2].plot(t, joint_pos[3*leg_idx+2, :], color='blue', label="Actual")
    axs[2].plot(t, joint_pos_desired[3*leg_idx+2, :], 'k--', label="Desired")
    axs[2].grid(True)
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Elbow Angle (rad)')
    axs[2].set_ylim(0.5*(max_elbow + min_elbow - 1.1*angle_range), 0.5*(max_elbow + min_elbow + 1.1*angle_range))

    # Add a single legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.8, 0.95), ncol=2)

    plt.tight_layout()

  if "states" in plots:
    fig, axs = plt.subplots(4, 2, figsize=(13, 12))
    fig.suptitle('CPG States')

    for i in range(4):
      leg_name = leg_from_idx[i]

      # -------- Amplitude subplot --------
      ax = axs[i, 0]
      ax2 = ax.twinx()

      # Value (blue)
      ax.plot(t, amplitudes[i], linewidth=2, color='blue')
      ax.set_ylabel("r", color='blue')
      ax.tick_params(axis='y', labelcolor='blue')
      ax.grid(True, alpha=0.3)
      ax.set_title(f"{leg_name} Amplitude")

      # Derivative (green)
      ax2.plot(t, d_amplitudes[i], linestyle='--', linewidth=1.5, color='green')
      ax2.set_ylabel("dr/dt", color='green')
      ax2.tick_params(axis='y', labelcolor='green')

      if i == 3:
        ax.set_xlabel("Time [s]")
      else:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)

      # -------- Phase subplot --------
      ax = axs[i, 1]
      ax2 = ax.twinx()

      # Value (blue)
      ax.plot(t, phases[i], linewidth=2, color='blue')
      ax.set_ylabel("Theta [rad]", color='blue')
      ax.tick_params(axis='y', labelcolor='blue')
      ax.grid(True, alpha=0.3)
      ax.set_title(f"{leg_name} Phase")

      # Derivative (green)
      ax2.plot(t, d_phases[i], linestyle='--', linewidth=1.5, color='green')
      ax2.set_ylabel("dTheta/dt [rad/s]", color='green')
      ax2.tick_params(axis='y', labelcolor='green')

      if i == 3:
        ax.set_xlabel("Time [s]")
      else:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)

    plt.tight_layout()
  
  if "orientation" in plots:
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Body Orientation')

    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs[i].plot(t, body_orientation[i, :], color='blue')
        axs[i].set_ylabel(f'{labels[i]} (rad)')
        axs[i].grid(True)
        if i == 2:
            axs[i].set_xlabel('Time (s)')
        axs[i].set_xlim(0, max(t))

    plt.tight_layout()

    np.save("orientation", body_orientation)
  
  if "trajectory" in plots:
    # Plot xy trajectory
    fig = plt.figure()
    plt.plot(body_position[0, :], body_position[1, :], color='blue')
    plt.title('Body XY Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()

    np.save("xy_traj", body_position[:2, :])

  if plots:
    plt.show()

def attitude_vmc( # virtual model controller 
    base_orientation_matrix: np.ndarray,
    J: np.ndarray, # Jacobian of the leg
    k_att_vmc,
    leg_id,
    ) -> np.ndarray:

    # All motor torques are in a single array
    tau = np.zeros(12)
    BP = np.array([[1, 1, -1, -1],
                  [-1, 1, -1, 1],
                  [0, 0, 0, 0]])
    R = base_orientation_matrix
    P = R @ BP
    F = k_att_vmc * (np.array([0, 0, 1]) @ P)
    F_vmc = np.vstack((np.zeros((2, 4)), F))
    F_i = F_vmc[:, leg_id]

    tau_i = J.T @ F_i

    return tau_i

def direction_vmc( # virtual model controller
    delta_yaw: float,
    delta_y: float,
    J: np.ndarray, # Jacobian of the leg
    k_dir_vmc: tuple,
    leg_id: int
    ) -> np.ndarray:

    k_rot, k_lin = k_dir_vmc

    # Angular correction
    F_i = - np.array([0, 1, 0]) * k_rot * delta_yaw
    if leg_id in (2, 3): # rear legs -> invert y force
      F_i[1] *= -1
    
    # Linear correction
    F_i += - np.array([0, 1, 0]) * k_lin * delta_y

    tau_i = J.T @ F_i

    return tau_i

def lateral_vmc(
    delta_roll: float,
    k_lat_vmc
    ) -> np.ndarray:

    # Apply to Hip joints only
    tau_i = np.array([1, 0, 0]) * k_lat_vmc * delta_roll

    return tau_i


if __name__ == "__main__":
  # For the configurations below use:
  # # Joint PD gains
  # kp=np.array([100,100,100])
  # kd=np.array([2,2,2])

  # # Cartesian PD gains, 300 5??
  # kpCartesian = np.diag([700]*3)
  # kdCartesian = np.diag([70]*3)

  hopf_trot_high = {'gait': 'TROT', # TROT_HIGH_1_8ms
        'omega_swing': 14.85*2*np.pi,
        'omega_stance': 5.4*2*np.pi,
        'alpha': 5.0,
        "coupling_strength": 1,     
        "couple": True,                
        "ground_clearance": 0.01, 
        "ground_penetration": 0.01, 
        "robot_height": 0.28,  
        "des_step_len": 0.095}

  hopf_trot_low = {'gait': 'TROT', # TROT_LOW_0_18ms
        'omega_swing': 2.4*2*np.pi,
        'omega_stance': 1.2*2*np.pi,
        'alpha': 10.0,              
        "ground_clearance": 0.02, 
        "ground_penetration": 0.01}

  hopf_walk_high = {'gait': 'WALK', # WALK_HIGH_2_5ms
        'omega_swing': 12*2*np.pi,
        'omega_stance': 6*2*np.pi,
        'alpha': 10.0,
        "coupling_strength": 1,     
        "couple": True,                
        "ground_clearance": 0.01, 
        "ground_penetration": 0.01, 
        "robot_height": 0.3,  
        "des_step_len": 0.12}

  hopf_walk_low = {'gait': 'WALK', # WALK_LOW_0_18ms
        'omega_swing': 4.7*2*np.pi,
        'omega_stance': 1.7*2*np.pi,
        'alpha': 5.0,              
        "ground_clearance": 0.03, 
        "ground_penetration": 0.01}

  hopf_bound_high = {'gait': 'BOUND', # BOUND_HIGH_1ms
        'omega_swing': 3.6*2*np.pi,
        'omega_stance': 1.8*2*np.pi,
        'alpha': 15.0,
        "coupling_strength": 1,
        "couple": True,
        "ground_clearance": 0.04,
        "ground_penetration": 0.01,
        "robot_height": 0.23,
        "des_step_len": 0.13}
  
  hopf_bound_low = {'gait': 'BOUND', # BOUND_LOW_0_18ms
        'omega_swing': 3.6*2*np.pi,
        'omega_stance': 0.8*2*np.pi,
        'alpha': 15.0,
        "coupling_strength": 1,
        "couple": True,
        "ground_clearance": 0.03,
        "ground_penetration": 0.01,
        "robot_height": 0.23,
        "des_step_len": 0.09}

  run_cpg(plots=["report"], hopf_params=hopf_trot_low, add_cartesian_pd=True, time_horizon=7,
          # k_att_vmc=300,
          # k_dir_vmc=(100, 100),
          # k_lat_vmc=250
          )

