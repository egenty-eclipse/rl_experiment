from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id="gymnasium_env/Bess-v0",
    entry_point="gymnasium_env.envs:BessEnv",
    max_episode_steps=100,
)
