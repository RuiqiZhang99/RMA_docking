"""
Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.
Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.
"""

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
# from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType
from gym_pybullet_drones.envs.BaseHetero import DroneEntity
from gym_pybullet_drones.envs.MultiDocking import TwoDroneDock
from gym_pybullet_drones.rma.ppo import PPO
from gym_pybullet_drones.rma.rma_phase1 import RMA_phase1

import wandb

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
# DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DRONE_CFG = [DroneEntity(drone_model=DroneModel.CF2X), DroneEntity(drone_model=DroneModel.CF2X)]
# DEFAULT_MA = True

parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
# parser.add_argument('--trial', default='1', type=str, help='The Number of Trials', metavar='')
parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO,  type=str2bool, help='Whether to record a video (default: False)', metavar='')
parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
args = parser.parse_args()


def run(output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        plot=True, 
        record_video=DEFAULT_RECORD_VIDEO, 
        local=True):

    filename = os.path.join(output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    
    train_env = make_vec_env(TwoDroneDock,
                            env_kwargs=dict(drone_cfg=DRONE_CFG, obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=100, pyb_freq=200),
                            n_envs=1,
                            seed=10086,
                            )
    eval_env = TwoDroneDock(drone_cfg=DRONE_CFG, obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=100, pyb_freq=200)

    # =========================  Check the Env Space ===============================
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # ======================  Define the Train function ============================
    wandb.init(project="real_world_learning", name=f"run_2", entity="hiperlab")
    
    model = PPO(policy = 'MlpPolicy',
                env = train_env,
                # policy_kwargs = dict(activation_fn= torch.nn.Tanh, #torch.nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[512, 512])], log_std_init=-0.5,),
                use_sde = False,
                n_steps = 1000,
                batch_size = 128,
                seed = 1,
                # tensorboard_log=filename+'/tb/',
                ent_coef = 0.01,
                verbose=1)

    # =====================  Define the Reward Threshold ===========================
    '''
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 949.5
    else:
        target_reward = 920.
        
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    '''
    eval_callback = EvalCallback(eval_env,
                                 # callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=1000,
                                 deterministic=True,
                                 render=False)
    
    
    model.learn(total_timesteps=int(2e6),
                callback=eval_callback,
                log_interval=1000)
    
    wandb.finish()

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    if local:
        input("Press Enter to continue...")

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    
    test_env = TwoDroneDock(gui=gui,
                            drone_cfg=DRONE_CFG,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            ctrl_freq=100, pyb_freq=200,
                            record=record_video)
    test_env_nogui = TwoDroneDock(drone_cfg=DRONE_CFG, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                drone_cfg=DRONE_CFG,
                output_folder=output_folder,
                num_drones=len(DRONE_CFG)
                )

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=64, options={})
    start = time.time()
    for i in range(int((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            for d in range(len(DRONE_CFG)):
                logger.log(drone=d, timestamp=i/test_env.CTRL_FREQ, state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]), control=np.zeros(12))
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=1, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':

    run(**vars(args))
