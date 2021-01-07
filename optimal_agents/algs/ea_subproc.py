import argparse
import os
from optimal_agents.morphology import Morphology
from optimal_agents.utils.loader import get_env, get_morphology, get_paths, load, Parameters
from optimal_agents.utils.trainer import train_rl
from optimal_agents.utils.tester import eval_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
import torch_geometric
import torch as th
import numpy as np

from optimal_agents.algs.graph_rollout_buffer import obs_to_graph

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--base-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--morphology", type=str, default=None)
    parser.add_argument("--eval-data", type=int, default=0)
    parser.add_argument("--eval-ep", type=int, default=10)
    parser.add_argument("--segment-embeddings", action='store_true', default=False)

    args = parser.parse_args()
    
    base_path, model_path = args.base_path, args.model_path
    params = Parameters.load(args.params)
    morphology = Morphology.load(args.morphology)
    
    if model_path:
        model, _ = load(model_path, params, best=False, load_env=False)
        print("Finetuning from existing model", model_path)
    else:
        model = None # Initialize if no path given.
    
    # try:
    model, model_path = train_rl(params, model=model, env=None, morphology=morphology, path=base_path, verbose=1)
    eval_env = get_env(params, morphology=morphology)

    if args.eval_ep > 0:
        fitness = eval_policy(model, eval_env, num_ep=args.eval_ep, deterministic=True, gif=False, render=False, 
                            verbose=0)[0]
    else:
        # Determine fitness from training history
        x, y = ts2xy(load_results(model_path), 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last -eval_ep episodes
            ys = y[args.eval_ep:]
            fitness = np.mean(ys)
        else:
            fitness = -np.inf
    # except Exception as e:
    #     print("Encountered Error", e)
    #     print("Assigning zero fitness")
    #     model_path, tb_path = get_paths(params, path=base_path)
    #     os.makedirs(model_path, exist_ok=True)
    #     with open(os.path.join(model_path, 'fitness.tmp'), 'w+') as f:
    #         f.write(str(-1*float('inf')))
    #     exit() # Exit if we fail to train the model.

    # TODO: Since methods are now static, this can be moved to the pruning inner loop, but then its not parallel.
    # IO overhead with files is probably not worth it.
    # We have succeeded in training the model. Evaluate it and collect relevant data as required.
    if args.eval_data > 0:
        eval_obs = eval_env.get_morphology_obs(morphology, include_segments=args.segment_embeddings)
        eval_data = []
        eval_data.append((obs_to_graph([eval_obs])[0], fitness))
        th.save(eval_data, os.path.join(model_path, 'pruning_data.pkl'))
        
    # Log fitness to a file.
    with open(os.path.join(model_path, 'fitness.tmp'), 'w+') as f:
        f.write(str(fitness))

    del eval_env
    del model
