from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

class G1WalkPhaseCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        num_envs = 4096
        num_actions = 21
        n_priv = 0
        n_proprio = 2 + 3 + 3 + 2 + 3*num_actions
        n_priv_latent = 4 + 1 + 2*num_actions + 3
        history_len = 10
        
        num_observations = n_proprio + n_priv_latent + history_len*n_proprio + n_priv

        num_privileged_obs = None

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = True
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
    
    class terrain(HumanoidCfg.terrain):
        mesh_type = 'trimesh'
        height = [0, 0.04]
        horizontal_scale = 0.1
    
    class init_state(HumanoidCfg.init_state):
        pos = [0, 0, 0.8]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.4,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.8,
            'left_ankle_pitch_joint': -0.35,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.8,
            'right_ankle_pitch_joint': -0.35,
            'right_ankle_roll_joint': 0.0,
            
            'torso_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_pitch_joint': 0.0,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_pitch_joint': 0.0,
        }
    
    class control(HumanoidCfg.control):
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 200,
                     'knee': 200,
                     'ankle': 20,
                     'torso': 200,
                     'shoulder': 40,
                     'elbow': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 5,
                     'ankle': 4,
                     'torso': 5,
                     'shoulder': 10,
                     'elbow': 10,
                     }  # [N*m/rad]  # [N*m*s/rad]
        
        action_scale = 0.5
        decimation = 20
    
    class sim(HumanoidCfg.sim):
        dt = 0.001
        
    class normalization(HumanoidCfg.normalization):
        clip_actions = 5
    
    class asset(HumanoidCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1-nohand.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof.urdf'
        # for both joint and link name
        torso_name: str = 'torso_link'  # humanoid pelvis part
        chest_name: str = 'torso_link'  # humanoid chest part
        forehead_name: str = 'head_link'  # humanoid head part

        waist_name: str = 'torso_joint'

        # for link name
        thigh_name: str = 'hip_roll_link'
        shank_name: str = 'knee_link'
        foot_name: str = 'ankle_roll_link'  # foot_pitch is not used
        upper_arm_name: str = 'shoulder_roll_link'
        lower_arm_name: str = 'elbow_pitch_link'
        hand_name: str = 'hand'

        # for joint name
        hip_name: str = 'hip'
        hip_roll_name: str = 'hip_roll_joint'
        hip_yaw_name: str = 'hip_yaw_joint'
        hip_pitch_name: str = 'hip_pitch_joint'
        knee_name: str = 'knee_joint'
        ankle_name: str = 'ankle'
        ankle_pitch_name: str = 'ankle_pitch_joint'
        shoulder_name: str = 'shoulder'
        shoulder_pitch_name: str = 'shoulder_pitch_joint'
        shoulder_roll_name: str = 'shoulder_roll_joint'
        shoulder_yaw_name: str = 'shoulder_yaw_joint'
        elbow_name: str = 'elbow_pitch_joint'

        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip"]
        terminate_after_contacts_on = ['torso_link']
    
    class rewards(HumanoidCfg.rewards):
        regularization_names = [
                        "dof_error", 
                        "dof_error_upper",
                        "feet_stumble",
                        "feet_contact_forces",
                        "lin_vel_z",
                        "ang_vel_xy",
                        "orientation",
                        "dof_pos_limits",
                        "dof_torque_limits",
                        "collision",
                        "torque_penalty",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = True
        regularization_scale_gamma = 0.0001
        class scales:
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2

            feet_air_time = 1.
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2

            tracking_lin_vel_exp = 1.875
            tracking_ang_vel = 2.0

            alive = 2.0
            dof_error = -0.15
            dof_error_upper = -0.2
            feet_stumble = -1.25
            feet_contact_forces = -5e-4
            
            lin_vel_z = -1.0
            ang_vel_xy = -0.1
            orientation = -1.0
            
            collision = -10.0
            
            dof_pos_limits = -10
            dof_torque_limits = -1.0
            torque_penalty = -6e-7

        min_dist = 0.2
        max_dist = 0.5
        max_knee_dist = 0.5
        target_joint_pos_scale = 0.17
        target_feet_height = 0.1
        cycle_time = 0.64
        double_support_threshold = 0.1
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 350
        termination_height = 0.3
    
    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.6, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidCfg.noise):
        add_noise = True
        noise_increasing_steps = 5000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            imu = 0.05
        
    class commands:
        curriculum = False
        num_commands = 3
        resampling_time = 3. # time before command are changed[s]

        ang_vel_clip = 0.1
        lin_vel_clip = 0.1

        class ranges:
            lin_vel_x = [0., 0.8] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]

class G1WalkPhaseCfgPPO(HumanoidCfgPPO):
    seed = 1
    class runner(HumanoidCfgPPO.runner):
        policy_class_name = 'ActorCriticRMA'
        algorithm_class_name = 'PPORMA'
        runner_class_name = 'OnPolicyRunner'
        max_iterations = 20001 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    class policy(HumanoidCfgPPO.policy):
        action_std = [0.3, 0.3, 0.3, 0.4, 0.2, 0.2] * 2 + [0.1] + [0.2] * 8
        
    class algorithm(HumanoidCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]
        