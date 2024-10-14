from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

class BerkeleyWalkPhaseCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        num_envs = 4096
        num_actions = 12
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
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.0,
                        "parkour_gap": 0.0,
                        "demo": 0.,}
        terrain_proportions = list(terrain_dict.values())
        mesh_type = 'trimesh'
        height = [0, 0.04]
        horizontal_scale = 0.1
    
    class init_state(HumanoidCfg.init_state):
        pos = [0, 0, 0.55]
        default_joint_angles = {
            'LL_HR': -0.071,
            'LL_HAA': 0.103,
            'LL_HFE': -0.463,
            'LL_KFE': 0.983,
            'LL_FFE': -0.350,
            'LL_FAA': 0.126,
            'LR_HR': 0.071,
            'LR_HAA': -0.103,
            'LR_HFE': -0.463,
            'LR_KFE': 0.983,
            'LR_FFE': -0.350,
            'LR_FAA': -0.126,
        }
    
    class control(HumanoidCfg.control):
        stiffness = {
        'HR': 10, 'HAA': 10, 'HFE': 15,
        'KFE': 15, 'FFE': 1, 'FAA': 1,
        }
        damping = {
            'HR': 1.5, 'HAA': 1.5, 'HFE': 1.5,
            'KFE': 1.5, 'FFE': 0.1, 'FAA': 0.1,
        }
        
        action_scale = 0.5
        decimation = 4
        
    class normalization(HumanoidCfg.normalization):
        clip_actions = 5
    
    class asset(HumanoidCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/berkeley_humanoid/urdf/robot.urdf'
        # for both joint and link name
        torso_name: str = 'torso'  # humanoid pelvis part
        chest_name: str = 'torso'  # humanoid chest part

        # for link name
        thigh_name: str = 'hfe'
        shank_name: str = 'kfe'
        foot_name: str = 'faa'  # foot_pitch is not used

        # for joint name
        hip_name: str = 'hip'
        hip_roll_name: str = 'HR'
        hip_yaw_name: str = 'HAA'
        hip_pitch_name: str = 'HFE'
        ankle_pitch_name: str = 'FFE'


        feet_bodies = ['ll_faa', 'lr_faa']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["hfe", "kfe"]
        terminate_after_contacts_on = ['torso']
    
    class rewards(HumanoidCfg.rewards):
        regularization_names = [
                        "dof_error", 
                        "feet_stumble",
                        "feet_contact_forces",
                        "lin_vel_z",
                        "ang_vel_xy",
                        "orientation",
                        "chest_orientation",
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
            
            feet_air_time = 1.5
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            
            tracking_lin_vel_exp = 1.875
            tracking_ang_vel = 2.0
            
            alive = 2.0
            dof_error = -0.15
            feet_stumble = -1.25
            feet_contact_forces = -5e-4
            
            lin_vel_z = -1.0
            ang_vel_xy = -0.1
            orientation = -1.0
            
            collision = -10.0

            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            torque_penalty = -6e-7

        min_dist = 0.2
        max_dist = 0.45
        max_dist_knee = 0.45
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.05      # m
        cycle_time = 0.64                # sec
        double_support_threshold = 0.1
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 160  # Forces above this value are penalized
        termination_height = 0.2
    
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
            lin_vel_x = [0., 0.8] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]

class BerkeleyWalkPhaseCfgPPO(HumanoidCfgPPO):
    seed = 1
    class runner(HumanoidCfgPPO.runner):
        policy_class_name = 'ActorCriticRMA'
        algorithm_class_name = 'PPORMA'
        runner_class_name = 'OnPolicyRunner'
        max_iterations = 30002 # number of policy updates

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
        action_std = [0.3, 0.3, 0.3, 0.4, 0.2, 0.2] * 2
        