import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR

import torch

gym = gymapi.acquire_gym()

args = gymutil.parse_arguments(
    description="Script for testing the urdf or mjcf asset",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create"},
        ])

sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity.x = 0
sim_params.gravity.y = 0
sim_params.gravity.z = -9.81
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()
    
# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

asset_root = f'{LEGGED_GYM_ROOT_DIR}/resources/robots'
# asset_file = "gr1/gr1.urdf"
# asset_file = "h1/h1.urdf"
# asset_file = "gr1t2/urdf/GR1T2.urdf"
# asset_file = "g1/g1-nohand.urdf"
asset_file = "gr1t1/urdf/GR1T1.urdf"
# asset_file = "GR2T4A/URDF/GR2T4A/urdf/GR2T4A_new.urdf"
# asset_file = "berkeley_humanoid/urdf/robot.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.use_mesh_materials = True
asset_options.disable_gravity = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

num_envs = args.num_envs
num_per_row = int(np.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # create ball pyramid
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.5)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    humanoid_handle = gym.create_actor(env, asset, pose, "humanoid", i, 1)
    
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(5, 5, 5), gymapi.Vec3(0, 0, 0))


initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)