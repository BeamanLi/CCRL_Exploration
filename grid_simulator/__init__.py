from gym.envs.registration import register

register(
    id="ExploreEnv-v1",
    entry_point="grid_simulator.envs.explore_env_v1:ExploreEnv_v1",
)

