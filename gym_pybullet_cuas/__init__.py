from gym.envs.registration import register

register(
    id='cuas-v0',
    entry_point='gym_pybullet_cuas.envs:CounterUAS',
)