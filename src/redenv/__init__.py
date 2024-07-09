from gym.envs.registration import register

register(
    id='redenv-v0',
    entry_point='redenv.env:RedmountainEnv'
)