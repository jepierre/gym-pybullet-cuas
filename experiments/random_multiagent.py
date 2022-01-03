import numpy as np

from gym_pybullet_cuas.envs.counter_uas import CounterUAS
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
import shared_constants


OBS = ObservationType.KIN
ACT = ActionType.ONE_D_RPM

init_xyzs = np.vstack(
    [
        np.array([1, 0, 1]),
        np.array([2, 0, 1]),
        np.array([-1, 0, 1]),
        np.array([-2, 0, 1]),
    ]
)


env = CounterUAS(
    num_drones=4,
    aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
    #  initial_xyzs = init_xyzs,
    obs=OBS,
    act=ACT,
    gui=True,
    record=False,
)

print(env.reset())


for i in range(100 * int(env.SIM_FREQ / env.AGGR_PHY_STEPS)):
    random_action = env.action_space.sample()
    obs, reward, done, info = env.step(random_action)
    env.render()

    # if done['__all__']:
    #     break


env.close()