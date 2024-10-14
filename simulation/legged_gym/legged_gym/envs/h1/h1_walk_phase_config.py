from legged_gym.envs.base.humanoid_config import HumanoidCfg, HumanoidCfgPPO

class H1WalkPhaseCfg(HumanoidCfg):
    class env(HumanoidCfg.env):
        num_envs = 4096
        num_actions = 19
        n_priv = 0
        n_proprio = 2 + 3 + 3 + 2 + 3*num_actions
        n_priv_latent = 4 + 1 + 2*num_actions
        history_len = 10
        
        num_observations = n_proprio + n_priv_latent + history_len*n_proprio + n_priv + 3

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
        pos = [0, 0, 1.0]
        default_joint_angles =  {# = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.6,         
           'left_knee_joint' : 1.2,       
           'left_ankle_joint' : -0.6,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.6,                                       
           'right_knee_joint' : 1.2,                                             
           'right_ankle_joint' : -0.6,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
    }
    
    class control(HumanoidCfg.control):
        stiffness = {
            'hip_yaw': 200,
            'hip_roll': 200,
            'hip_pitch': 200,
            'knee': 200,
            'ankle': 40,
            'torso': 300,
            'shoulder': 40,
            'elbow': 40,
        }
        damping = {
            'hip_yaw': 5,
            'hip_roll': 5,
            'hip_pitch': 5,
            'knee': 5,
            'ankle': 2,
            'torso': 6,
            'shoulder': 2,
            'elbow': 2,
        }
        
        action_scale = 0.5
        decimation = 20
    
    class sim(HumanoidCfg.sim):
        dt = 0.001
        
    class normalization(HumanoidCfg.normalization):
        clip_actions = 5
    
    class asset(HumanoidCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision.urdf'
        # for both joint and link name
        torso_name: str = 'torso_link'  # humanoid pelvis part

        chest_name: str = 'imu_link'  # humanoid chest part
        forehead_name: str = 'head_pitch'  # humanoid head part

        waist_name: str = 'torso_joint'

        # for link name
        thigh_name: str = 'hip_pitch_link'
        shank_name: str = 'knee_link'
        foot_name: str = 'ankle_link'  # foot_pitch is not used
        upper_arm_name: str = 'shoulder_yaw_link'
        lower_arm_name: str = 'elbow_link'
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

        feet_bodies = ['left_ankle_link', 'right_ankle_link']
        n_lower_body_dofs: int = 11

        penalize_contacts_on = ["shoulder", "hip", "elbow"]
        terminate_after_contacts_on = ["pelvis"]
    
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
            ang_vel_xy = -0.4
            orientation = -1.0

            collision = -10.0

            dof_pos_limits = -10
            dof_torque_limits = -0.1
            torque_penalty = -6e-7


        min_dist = 0.25
        max_dist = 0.5
        max_dist_knee = 0.5
        target_joint_pos_scale = 0.17
        target_feet_height = 0.2       # m
        cycle_time = 0.80                # sec
        double_support_threshold = 0.1
        only_positive_rewards = False
        tracking_sigma = 0.25
        tracking_sigma_ang = 0.125
        max_contact_force = 500
    
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
            lin_vel_x = [-0.3, 1.5] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
    

class H1WalkPhaseCfgPPO(HumanoidCfgPPO):
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
    
    class algorithm(HumanoidCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]
        
    class policy(HumanoidCfgPPO.policy):
        action_std = [0.3, 0.3, 0.3, 0.4, 0.2] * 2 + [0.1] + [0.5] * 8
        