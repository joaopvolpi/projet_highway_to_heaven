import gymnasium as gym

"""

For including the new merge (continuous and 3 lanes):
1. Copy the file merge_custom_env.py in the directory highway_env.envs 

2. Add the following line in the file __init__.py in the directory highway_env.envs:
    from highway_env.envs.merge_custom_env import MergeEnv

3. Add the following line in the file __init__.py in the directory highway_env:
    register(
        id='merge-custom-v0',
        entry_point='highway_env.envs:MergeCustomEnv',
    )
    
4. Run the following code: 
    
    import highway_env

    highway_env.register_highway_envs()



"""



env = gym.make("merge-custom-v0", render_mode="rgb_array")

config = {
    "observation": {
        "type": "Kinematics",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
    },
    "action": {
        "type": "ContinuousAction",
    },
    "vehicules_count": 1,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": False,
}

env.unwrapped.configure(config_merge)
env.reset()


env.unwrapped.configure(config)
print(env.reset())