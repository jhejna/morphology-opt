import os
import optimal_agents
import yaml
import pprint
from datetime import date
import stable_baselines3

BASE = os.path.dirname(os.path.dirname(optimal_agents.__file__)) + '/data'
LOGS = os.path.dirname(os.path.dirname(optimal_agents.__file__)) + '/tb_logs'

class Parameters(object):

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.config = dict()

        # Algorithm Args
        self.config['alg'] = None
        self.config['alg_kwargs'] = {}
        self.config['learn_kwargs'] = {}
        self.config['callback_kwargs'] = {}

        # Policy Args
        self.config['policy'] = None
        self.config['policy_kwargs'] = {}

        # Environment Args
        self.config['env'] = None
        self.config['env_kwargs'] = {}
        self.config['wrapper'] = None
        self.config['wrapper_kwargs'] = {}
        self.config['num_envs'] = 1

        # Morphology Arguments
        self.config['morphology'] = None
        self.config['morphology_kwargs'] = {}

        self.config['mutation_kwargs'] = {}

        # Arena Arguments
        self.config['arena'] = None
        self.config['arena_kwargs'] = {}

        # Evolution Argumenats
        self.config['evo_alg'] = None
        self.config['evo_alg_kwargs'] = {}
        self.config['evo_learn_kwargs'] = {}

        # Logging Args
        self.config['seed'] = None
        self.config['monitor_kwargs'] = {}
        
        self.config['tensorboard'] = False
        
    def update(self, d):
        self.config.update(d)
    
    def save(self, path):
        if not '.' in path:
            path = os.path.join(path, "params.yaml")
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path):
        if not '.' in path:
            path = os.path.join(path, "params.yaml")
        with open(path, 'r') as f:
            data = yaml.load(f)
        config = cls()
        config.update(data)
        return config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, item):
        self.config[key] = item

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __str__(self):
        return pprint.pformat(self.config, indent=4)

    def get_save_name(self, evo=False):
        if 'name' in self.config and self.config['name']:
            name =  self['name']
        elif evo:
            name = self['env'] + ('_' + self['wrapper'] if self['wrapper'] else "") + '_' + self['evo_alg']
        else:
            name = self['env'] + ('_' + self['wrapper'] if self['wrapper'] else "") + '_' + self['alg']
        if not self['seed'] is None:
            name += '_s' + str(self['seed'])
        return name

def get_alg(params: Parameters):
    alg_name = params['alg']
    try:
        alg = vars(optimal_agents.algs)[alg_name]
    except:
        alg = vars(stable_baselines3)[alg_name]
    return alg

def get_env(params: Parameters, morphology=None):
    env_name = params['env']
    try:
        env_cls = vars(optimal_agents.envs)[params['env']]
        arena = get_arena(params)
        env = env_cls(morphology, arena=arena, **params['env_kwargs'])
        if params['wrapper']:
            env = vars(optimal_agents.envs)[params['wrapper']](env, **params['wrapper_kwargs'])
    except:
        # If we don't get the env, then we assume its a gym environment
        import gym
        env = gym.make(params['env'])
        if params['wrapper']:
            env = vars(gym.wrappers)[params['wrapper']](env, **params['wrapper_kwargs'])
    return env    

def get_morphology(params: Parameters):
    morphology_name = params['morphology']
    if morphology_name is None:
        return None
    try:
        morphology = optimal_agents.morphology.Morphology.load(morphology_name)
    except:
        morphology = vars(optimal_agents.morphology)[morphology_name](**params['morphology_kwargs'], mutation_kwargs=params['mutation_kwargs'])
    return morphology

def get_arena(params: Parameters):
    arena_name = params['arena']
    if arena_name is None:
        return None
    arena = vars(optimal_agents.morphology.arenas)[arena_name](**params['arena_kwargs'])
    return arena

def get_policy(params: Parameters):
    policy_name = params['policy']
    if policy_name is None:
        policy_name = 'MlpPolicy' 
    try:
        policy = vars(optimal_agents.policies)[policy_name]
        return policy
    except:
        alg_name = params['alg']
        if 'SAC' in alg_name:
            search_location = stable_baselines3.sac.policies
        elif 'DDPG' in alg_name:
            search_location = stable_baselines3.ddpg.policies
        elif'DQN' in alg_name:
            search_location = stable_baselines3.deepq.policies
        elif 'TD3' in alg_name:
            search_location = stable_baselines3.td3.policies
        elif 'PPO' in alg_name:
            search_location = stable_baselines3.ppo.policies
        else:
            search_location = stable_baselines3.common.policies
        policy = vars(search_location)[policy_name]
        return policy

def get_paths(params: Parameters, path=None, evo=False):
    if path is None:
        date_dir = BASE
    else:
        date_dir = path

    save_name = params.get_save_name(evo=evo)
    if os.path.isdir(date_dir):
        candidates = [f_name for f_name in os.listdir(date_dir) if '_'.join(f_name.split('_')[:-1]) == save_name]
        if len(candidates) == 0:
            save_name += '_0'
        else:
            num = max([int(dirname[-1]) for dirname in candidates]) + 1
            save_name += '_' + str(num)
    else:
        save_name += '_0'

    save_path = os.path.join(date_dir, save_name)
    tb_path = os.path.join(LOGS, save_name) if params['tensorboard'] else None
    return save_path, tb_path

def load_from_name(path, best=False, load_env=True, ret_params=False, alg_args=None, morphology_index=0):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    params = Parameters.load(path)
    if ret_params:
        return load(path, params, best=best, load_env=load_env, alg_args=alg_args, morphology_index=morphology_index) + (params,)
    return load(path, params, best=best, load_env=load_env, alg_args=alg_args, morphology_index=morphology_index)

def load(path: str, params : Parameters, best=False, load_env=True, alg_args=None, morphology_index=0):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    files = os.listdir(path)
    if not 'final_model.zip' in files and 'best_model.zip' in files:
        model_path = path + '/best_model.zip'
    elif 'best_model.zip' in files and best:
        model_path = path + '/best_model.zip'
    elif 'final_model.zip' in files:
        model_path = path + '/final_model.zip'
    else:
        raise ValueError("Cannot find a model for name: " + path)
    # get model
    alg = get_alg(params)
    if alg_args is None:
        alg_args = params['alg_kwargs']
    model = alg.load(model_path, **alg_args)
    if load_env:
        morphology_file_name = str(morphology_index) + ".morphology.pkl"
        if morphology_file_name in files:
            from optimal_agents.morphology import Morphology
            morphology = Morphology.load(os.path.join(path, morphology_file_name))
            print("Loaded", morphology_file_name)
        else:
            print("WARNING: Could not find morphology.")
            morphology = None
        env = get_env(params, morphology=morphology)
    else:
        env = None
    return model, env
