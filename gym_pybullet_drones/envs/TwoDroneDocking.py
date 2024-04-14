import numpy as np

# from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary

import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseHetero import BaseHetero, DroneEntity
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class BaseRLHetero(BaseHetero):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_cfg: list=[DroneEntity(drone_model=DroneModel.CF2X), DroneEntity(drone_model=DroneModel.CF2X)],
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB_GND_DRAG_DW,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 500,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 obs_normlization: bool=True,
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        # Create a buffer for the last 0.5 sec of actions
        self.NUM_DRONES = len(drone_cfg)
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2) 
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        vision_attributes = False # We don't take RGB as the input
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.OBS_NORM = obs_normlization
        
        # Create integrated controllers
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(self.NUM_DRONES)]
            
        super().__init__(drone_cfg=drone_cfg,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=False,
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        # Set a limit on the maximum target speed
        if act == ActionType.VEL:
            for drone_idx, drone_entity in enumerate(self.DRONE_CFG):   
                drone_entity.SPEED_LIMIT = 0.03 * drone_entity.MAX_SPEED_KMH * (1000/3600)

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros(size))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)



    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            drone_entity = self.DRONE_CFG[k]
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(drone_entity.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(drone_entity.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        assert self.OBS_TYPE == ObservationType.KIN
        # Observation vector: [X, Y, Z, R, P, Y, VX, VY, VZ, WX, WY, WZ, AX, AY, AZ, ax, ay, az]
        lo = -100.0
        hi = +100.0
        obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,0, lo,lo,lo] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
        '''
        # Add action buffer to observation space
        act_lo = -1
        act_hi = +1
        for i in range(self.ACTION_BUFFER_SIZE):
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo,act_lo,act_lo,act_lo])])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi,act_hi,act_hi,act_hi])])
        '''
        
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    


    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A list with self.NUM_DRONE if obs.shape=(12) array

        """
        assert self.OBS_TYPE == ObservationType.KIN
        
        ret = np.zeros((self.NUM_DRONES, 24))
        obs_1 = self._getDroneStateVector(nth_drone=0)
        obs_2 = self._getDroneStateVector(nth_drone=1)
        # obs = [X, Y, Z, R, P, Y, VX, VY, VZ, WX, WY, WZ, AX, AY, AZ, ax, ay, az] + [X2, Y2, Z2, R2, P2, Y2]
        true_obs_1 = np.hstack([obs_1[0:3], obs_1[7:16], obs_1[19:25], obs_2[0:3], obs_2[7:10]]).reshape(24)
        true_obs_2 = np.hstack([obs_2[0:3], obs_2[7:16], obs_2[19:25], obs_1[0:3], obs_1[7:10]]).reshape(24)
        ret[0, :] = true_obs_1
        ret[1, :] = true_obs_2
        if self.OBS_NORM:
            ret = ret * 0.01
        return ret

# ==============================================================================================


class MultiDocking(BaseRLHetero):
    """Multi-agent RL problem: Flying-Battery Docking."""

    def __init__(self,
                 drone_cfg: list=[DroneEntity(drone_model=DroneModel.CF2X), DroneEntity(drone_model=DroneModel.CF2X)],
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs = None,
                 initial_rpys = None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 200,
                 ctrl_freq: int = 200,
                 sim_time: float = 10.0, # Simulation time in second.
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
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
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = sim_time
        super().__init__(drone_cfg=drone_cfg,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        self.NUM_DRONES = len(drone_cfg)
        
        self.TARGET_POS = np.array([
            [0, 0, 1.5, 0, 0, 0],
            [0, 0, 1.7, 0, 0, 0]
        ])
        self.TARGET_VEL = np.zeros_like(self.TARGET_POS)
        self.TARGET_ACC = np.zeros_like(self.TARGET_POS)
        # self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1/(i+1)] for i in range(self.NUM_DRONES)])

    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        self.alive_reward = 3
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            ret += self.alive_reward - np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:6], ord=2) - 0.1*np.linalg.norm(states[i][6:10], ord=2)
            
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        '''
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist = 0
        
        for i in range(self.NUM_DRONES):
            dist += np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
        if dist < .0001:
            return True
        else:
            return False
        '''
        return False

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
             or abs(states[i][7]) > 0.4 or abs(states[i][8]) > 0.4 # Truncate when a drone is too tilted
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years