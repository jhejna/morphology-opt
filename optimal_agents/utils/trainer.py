import os
import imageio
import copy
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import logger
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import optimal_agents
from optimal_agents.envs import GraphDummyVecEnv
from optimal_agents.utils.loader import get_paths, get_env, get_alg, get_policy, get_morphology
from optimal_agents.utils.loader import Parameters
from optimal_agents.utils.tester import eval_policy

class TrainCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, data_dir, tb_dir=None, verbose=1, eval_freq=10000):
        super(TrainCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.data_dir = data_dir
        self.tb_dir = tb_dir
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.data_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
            else:
                mean_reward = -np.inf
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                    print("Saving new best model.")
                self.model.save(self.data_dir + '/best_model')

        return True

def train_rl(params, model=None, env=None, morphology=None, path=None, verbose=1):
    if verbose > 0:
        print("Training Parameters: ", params)

    data_dir, tb_path = get_paths(params, path=path)
    os.makedirs(data_dir, exist_ok=True)

    # Currently saving params immediatly
    # TODO: Figure out where to save params later for the purpose of
    params.save(data_dir)

    if morphology is None:
        morphology = get_morphology(params)
        morphology = [morphology for _ in range(params['num_envs'])]
    if not isinstance(morphology, list):
        morphology = [morphology for _ in range(params['num_envs'])]
    
    # Create the environment if not given
    if env is None:  
        def make_env(i):
            env = get_env(params, morphology=morphology[i]) # Might be issues with same morphology in vec env but not sure.
            env = Monitor(env, data_dir + '/' + str(i))
            return env

        if params['alg'] in ("GPPO",): # Wrapper in graph obs wrapper if we are a graph environment.
            env = GraphDummyVecEnv([(lambda n: lambda: make_env(n))(i) for i in range(params['num_envs'])])
        else:
            env = DummyVecEnv([(lambda n: lambda: make_env(n))(i) for i in range(params['num_envs'])])

    for i, m in enumerate(morphology):
        if not m is None:
            m.save(data_dir + '/' + str(i) + '.morphology.pkl')
    
    # Set the seeds
    if params['seed']:
        seed = params['seed']
        set_random_seed(seed)
        params['alg_args']['seed'] = seed
    
    if model is None:
        alg = get_alg(params)
        policy = get_policy(params)
        model = alg(policy,  env, verbose=verbose, tensorboard_log=tb_path, policy_kwargs=params['policy_kwargs'], **params['alg_kwargs'])
    else:
        model.set_env(env)

    if verbose > 0:
        print("\n===============================\n")
        print("TENSORBOARD PATH:", tb_path)
        print("\n===============================\n")

    callback = TrainCallback(data_dir, tb_path, verbose=verbose)

    model.learn(**params['learn_kwargs'], 
                callback=callback)
    
    model.save(data_dir +'/final_model')
    
    env.close()
    del env

    # Return the model and storage path
    return model, data_dir

def train_ea(params, path=None):
    if path is None:
        path = get_paths(params, path=path, evo=True)[0] # throw away the tensorboard path.
    evo_alg = vars(optimal_agents.algs)[params['evo_alg']]
    evo_model = evo_alg(params, **params['evo_alg_kwargs'])
    evo_model.learn(path, **params['evo_learn_kwargs'])
