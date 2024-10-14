from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

class GR1WalkPhaseCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        num_envs = 4096
        num_actions = 19
        num_dofs = 21
        n_priv = 0
        n_proprio = 2 + 3 + 3 + 2 + 2*num_dofs + num_actions
        n_priv_latent = 4 + 1 + 2*num_dofs + 3
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
        
        use_motor_model = True
        normalize_obs = True
    
    class terrain(HumanoidCfg.terrain):
        mesh_type = 'trimesh'
        height = [0, 0.04]
        horizontal_scale = 0.1
    
    class init_state(HumanoidCfg.init_state):
        pos = [0, 0, 1.0]
        default_joint_angles = {
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.,
            'l_hip_pitch': -0.4,
            'l_knee_pitch': 0.8,
            'l_ankle_pitch': -0.4,
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': -0.,
            'r_hip_yaw': 0.,
            'r_hip_pitch': -0.4,
            'r_knee_pitch': 0.8,
            'r_ankle_pitch': -0.4,
            'r_ankle_roll': 0.0,

            # waist
            'waist_yaw': 0.0,
            'waist_pitch': 0.0,
            'waist_roll': 0.0,

            # head
            'head_yaw': 0.0,
            'head_pitch': 0.0,
            'head_roll': 0.0,

            # left arm
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.2,
            'l_shoulder_yaw': 0.0,
            'l_elbow_pitch': -0.3,
            'l_wrist_yaw': 0.0,
            'l_wrist_roll': 0.0,
            'l_wrist_pitch': 0.0,

            # right arm
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': -0.2,
            'r_shoulder_yaw': 0.0,
            'r_elbow_pitch': -0.3,
            'r_wrist_yaw': 0.0,
            'r_wrist_roll': 0.0,
            'r_wrist_pitch': 0.0
        }
    
    class control(HumanoidCfg.control):
        stiffness = {
        'hip_roll': 251.625, 'hip_yaw': 362.52, 'hip_pitch': 200,
        'knee_pitch': 200,
        'ankle_pitch': 10.98, 'ankle_roll': 0.0,
        'waist_yaw': 362.52, 'waist_pitch': 362.52, 'waist_roll': 362.52,
        'head_yaw': 10.0, 'head_pitch': 10.0, 'head_roll': 10.0,
        'shoulder_pitch': 40, 'shoulder_roll': 40, 'shoulder_yaw': 40,
        'elbow_pitch': 40,
        'wrist_yaw': 10.0, 'wrist_roll': 10.0, 'wrist_pitch': 10.0
        }
        damping = {
            'hip_roll': 14.72, 'hip_yaw': 10.08, 'hip_pitch': 11,
            'knee_pitch': 11,
            'ankle_pitch': 0.60, 'ankle_roll': 0.1,
            'waist_yaw': 10.08, 'waist_pitch': 10.08, 'waist_roll': 10.08,
            'head_yaw': 1.0, 'head_pitch': 1.0, 'head_roll': 1.0,
            'shoulder_pitch': 2, 'shoulder_roll': 2, 'shoulder_yaw': 2,
            'elbow_pitch': 2,
            'wrist_yaw': 1.0, 'wrist_roll': 1.0, 'wrist_pitch': 1.0
        }
        
        action_scale = 0.5
        decimation = 20
    
    class sim(HumanoidCfg.sim):
        dt = 0.001
        
    class normalization(HumanoidCfg.normalization):
        clip_actions = 5
    
    class asset(HumanoidCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1.urdf'

        torso_name: str = 'base'
        chest_name: str = 'waist_roll'
        forehead_name: str = 'head_pitch'

        waist_name: str = 'waist'
        waist_roll_name: str = 'waist_roll'
        waist_pitch_name: str = 'waist_pitch'
        head_name: str = 'head'
        head_roll_name: str = 'head_roll'
        head_pitch_name: str = 'head_pitch'

        # for link name
        thigh_name: str = 'thigh'
        shank_name: str = 'shank'
        foot_name: str = 'foot_roll'
        upper_arm_name: str = 'upper_arm'
        lower_arm_name: str = 'lower_arm'
        hand_name: str = 'hand'

        # for joint name
        hip_name: str = 'hip'
        hip_roll_name: str = 'hip_roll'
        hip_yaw_name: str = 'hip_yaw'
        hip_pitch_name: str = 'hip_pitch'
        knee_name: str = 'knee'
        ankle_name: str = 'ankle'
        ankle_pitch_name: str = 'ankle_pitch'
        shoulder_name: str = 'shoulder'
        shoulder_pitch_name: str = 'shoulder_pitch'
        shoulder_roll_name: str = 'shoulder_roll'
        shoulder_yaw_name: str = 'shoulder_yaw'
        elbow_name: str = 'elbow'
        wrist_name: str = 'wrist'
        wrist_yaw_name: str = 'wrist_yaw'
        wrist_roll_name: str = 'wrist_roll'
        wrist_pitch_name: str = 'wrist_pitch'

        feet_bodies = ['l_foot_roll', 'r_foot_roll']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "thigh"]
        terminate_after_contacts_on = ['waist']
    
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
            feet_contact_forces = -2e-3
            
            lin_vel_z = -1.0
            ang_vel_xy = -0.2
            orientation = -1.0
            
            collision = -10.0

            dof_pos_limits = -10.0
            dof_torque_limits = -1.0
            torque_penalty = -6e-7


        min_dist = 0.2
        max_dist = 0.5
        max_knee_dist = 0.25
        target_joint_pos_scale = 0.17
        target_feet_height = 0.1
        cycle_time = 0.8
        double_support_threshold = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500  # Forces above this value are penalized
        soft_torque_limit = 0.9

    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
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
            lin_vel_x = [0., 0.6] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]

class GR1WalkPhaseCfgPPO(HumanoidCfgPPO):
    seed = 1
    class runner(HumanoidCfgPPO.runner):
        policy_class_name = 'ActorCriticRMA'
        algorithm_class_name = 'PPORMA'
        runner_class_name = 'OnPolicyRunner'
        max_iterations = 12002 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    class algorithm(HumanoidCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]
    
    class policy(HumanoidCfgPPO.policy):
        action_std = [0.3, 0.3, 0.3, 0.4, 0.2] * 2 + [0.1] + [0.2] * 8
        