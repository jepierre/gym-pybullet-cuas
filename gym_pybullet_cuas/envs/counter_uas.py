import math
from gym.spaces.discrete import Discrete
import numpy as np
from gym import spaces
import pybullet as p
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import (
    BaseMultiagentAviary,
)

from enum import IntEnum


class AgentType(IntEnum):
    A = 0
    D = 1


class Agent:
    def __init__(self, id, type) -> None:
        self.id = id
        self.type = type
        self.done = False


def cartesian2polar(point1=(0, 0), point2=(0, 0)):
    """ Retuns conversion of cartesian to polar coordinates """
    r = distance(point1, point2)
    alpha = angle(point1, point2)

    return r, alpha


def distance(point_1=(0, 0), point_2=(0, 0)):
    """Returns the distance between two points"""
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def angle(point_1=(0, 0), point_2=(0, 0)):
    """Returns the angle between two points"""
    return math.atan2(point_2[1] - point_1[1], point_2[0] - point_1[0])


class CounterUAS(BaseMultiagentAviary):
    """Multi-agent RL problem: Counter Uncrewed Aerial Systems."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 4,
        num_attackers: int = 2,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
    ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        if num_drones - num_attackers < 1:
            raise ValueError(
                "number of drones must be greater than number of attackers"
            )

        if initial_xyzs is None:
            initial_xyzs = np.vstack(
                [
                    np.array([4, -0.5, 1]),
                    np.array([-4, 2.5, 1]),
                    np.array([-1.5, 0, 1]),
                    np.array([1.5, 1, 1]),
                ]
            )

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        self.EPISODE_LEN_SEC = 10
        self.target_reached = False
        self.num_defenders = num_drones - num_attackers
        self.num_attackers = num_attackers
        self.target_pos = np.array([0, 0, 1])

        # create agents
        self.agents = self._create_agents()

    ################################################################################
    def _create_agents(self):
        i = 0
        agents = []
        for j in range(self.num_attackers):
            agent = Agent(i, AgentType.A)
            agents.append(agent)
            i += 1

        for j in range(self.num_defenders):
            agent = Agent(i, AgentType.D)
            agents.append(agent)
            i += 1

        return agents

    def _addObstacles(self):
        """"""
        pass
        # p.loadURDF(
        #     "block.urdf",
        #     [0, 1, 0.1],
        #     p.getQuaternionFromEuler([0, 0, 0]),
        #     physicsClientId=self.CLIENT,
        # )

        # p.loadURDF(
        #     "duck_vhacd.urdf",
        #     [-1, 0, 0.1],
        #     p.getQuaternionFromEuler([0, 0, 0]),
        #     physicsClientId=self.CLIENT,
        # )

    def calc_distance(self, obj1, obj2):
        """
        returns euclidean distance between 2 objects
        """
        return np.linalg.norm(obj1 - obj2)

    def calc_bearing(self, obj1, obj2):
        """Returns bearing between 2 objects

        Args:
            obj1 ([type]): [description]
            obj2 ([type]): [description]
        """
        pass

    def closest_opponent(self):
        """
        Returns index of closest oponent

        Args:
            obj ([type]): [description]
            opponents ([type]): [description]

        Returns:
            [type]: [description]
        """        
        states = np.array(
            [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        )
        
        agent_other_dist = []
        for i, agent in enumerate(self.agents):
            temp_dist = []
            for j, other_agent in enumerate(self.agents):
                if agent.type == other_agent.type or agent.id == other_agent.id:
                    temp_dist.append(np.inf)
                else:
                    temp_dist.append(self.calc_distance(states[i, 0:2], states[j, 0:2]))
                    
            agent_other_dist.append(np.argmin(np.array(temp_dist)))
            
            
        return agent_other_dist
        # attacker_states = states[:self.num_attackers]
        # defender_states = states[self.num_attackers:]

        
        # for 
        # opponents_distances = np.array(
        #     [self.calc_distance(obj, opponent) for opponent in opponents]
        # )
        # return np.argmin(opponents_distances)

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Dict(
                {
                    i: spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                        dtype=np.uint8,
                    )
                    for i in range(self.NUM_DRONES)
                }
            )
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 12
            # observation vector x, y, z, r, p, y, vx, vy, vz, wx, wy, wz
            return spaces.Dict(
                {
                    i: spaces.Dict(
                        {
                            "own_obs": spaces.Box(
                                low=np.array(
                                    [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                                ),
                                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                dtype=np.float32,
                            ),
                            "closest_opponent": spaces.Discrete(self.NUM_DRONES),
                        }
                    )
                    for i in range(self.NUM_DRONES)
                }
            )
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(
                        i, segmentation=False
                    )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(
                            img_type=ImageType.RGB,
                            img_input=self.rgb[i],
                            path=self.ONBOARD_IMG_PATH + "drone_" + str(i),
                            frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ),
                        )
            return {i: self.rgb[i] for i in range(self.NUM_DRONES)}
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
            ############################################################
            #### OBS SPACE OF SIZE 12
            opponent_array = self.closest_opponent()
            
            obs_12 = np.zeros((self.NUM_DRONES, 12))
            for i in range(self.NUM_DRONES):
                obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs_12[i, :] = np.hstack(
                    [obs[0:3], obs[7:10], obs[10:13], obs[13:16]]
                ).reshape(
                    12,
                )
            return {i: {'own_obs': obs_12[i, :], 'closest_opponent':opponent_array[i]} for i in range(self.NUM_DRONES)}
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._computeObs()")

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        target_safe_radius = 5
        # MAX_XY = np.array([2, 2])
        # MAX_Z = 2
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range
        
        rewards = {i: 0 for i in range(self.NUM_DRONES)}
        self.target_reached = False
        states = np.array(
            [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        )

        array_closest_opponents = self.closest_opponent()
        
        for i, agent in enumerate(self.agents):
            reward = 0
            opponent_idx = array_closest_opponents[i]

            if agent.type == AgentType.A:
                distance_to_target = self.calc_distance(self.target_pos[0:2], states[i, 0:2])
                
                if distance_to_target < 0.2:
                    self.target_reached = True
                    reward += 1

                    # rewards[opponent_idx] += -.5
                    
                # else:
                    # reward += -1 * distance_to_target / target_safe_radius
                elif target_safe_radius >= distance_to_target >= 0.2:
                    safe_radius = 0.05 * (1 - (distance_to_target / target_safe_radius))
                    # rewards[opponent_idx] += -safe_radius
                    reward += safe_radius

            
            distance_to_opponent = self.calc_distance(states[i, 0:2], states[opponent_idx, 0:2])
            
            if distance_to_opponent < 0.1:
                if agent.type == AgentType.A:
                    reward += -1
                    # agent.done = True
                elif agent.type == AgentType.D:
                    reward += 1
            elif target_safe_radius >= distance_to_opponent >= 0.1:
                if agent.type == AgentType.D:
                    reward += 0.01 * (1 - (distance_to_opponent / target_safe_radius))
                    
                    
                    
                    

                
                    
            #     opponent_idx = array_closest_opponents[i]
                
            
            #     reward += self.calc_distance(states[i, 0:2], states[opponent_idx, 0:2])
                    
            # if agent.type == AgentType.D:
            #     opponent_idx = array_closest_opponents[i]
            #     reward = -1 / self.num_defenders * self.calc_distance(states[i, 0:2], states[opponent_idx, 0:2])
                
            rewards[i] += reward
           
            
        # attacker_states = states[:2]
        # defender_states = states[2:]

        # reward_idx = 0
        # reward = 0
        # for i in range(self.num_attackers):
        #     distance_to_target = self.calc_distance(
        #         self.target_pos, attacker_states[i, 0:3]
        #     )

        #     if distance_to_target < 0.5:
        #         self.target_reached = True
        #         reward = 20
        #     else:
        #         reward = -1 * distance_to_target
        #     # reward = -1 * np.linalg.norm(np.array([.5, .5, 1]) - states[i, 0:3]) ** 2

        #     # for j in range(self.num_defenders):
        #     #     reward += 1/self.num_defenders * self.calc_distance(np.array([defender_states[j, 0], defender_states[j, 1], defender_states[j, 2]]), attacker_states[i, 0:3])
        #     # reward += 1/self.num_defenders * np.linalg.norm(np.array([defender_states[j, 0], defender_states[j, 1], defender_states[j, 2]]) - attacker_states[i, 0:3]) ** 2

        #     rewards[reward_idx] = reward
        #     reward_idx += 1

        # reward = 0
        # for j in range(self.num_defenders):
        #     for i in range(self.num_attackers):
        #         reward += (
        #             -1
        #             / self.num_defenders
        #             * self.calc_distance(
        #                 np.array(
        #                     [
        #                         attacker_states[i, 0],
        #                         attacker_states[i, 1],
        #                         attacker_states[i, 2],
        #                     ]
        #                 ),
        #                 defender_states[j, 0:3],
        #             )
        #         )

        #     rewards[reward_idx] = reward
        #     reward_idx += 1

        # rewards[0] = -1 * np.linalg.norm(np.array([0, 0, 0.5]) - states[0, 0:3]) ** 2
        # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD
        # for i in range(1, self.NUM_DRONES):
        #     rewards[i] = (
        #         -(1 / self.NUM_DRONES)
        #         * np.linalg.norm(
        #             np.array([states[i, 0], states[i, 1], states[0, 2]])
        #             - states[i, 0:3]
        #         )
        #         ** 2
        #     )
        return rewards

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and
            one additional boolean value for key "__all__".

        """
        bool_val = (
            True if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        )
        # done = {i: bool_val for i in range(self.NUM_DRONES)}
        done = {i: bool_val or agent.done for i, agent in enumerate(self.agents)}
        done["__all__"] = (
            # bool_val
            bool_val or self.target_reached
        )  # True if True in done.values() else False
        return done

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        # MAX_XY = np.array([2, 2])
        # MAX_Z = 2
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16])
            if np.linalg.norm(state[13:16]) != 0
            else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(
            20,
        )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )
