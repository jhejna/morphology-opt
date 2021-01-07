import os
import copy
import numpy as np
import optimal_agents
from .loader import load_from_name, load
from .loader import BASE, Parameters

RENDERS = os.path.dirname(os.path.dirname(optimal_agents.__file__)) + '/renders'

def eval_policy(model, env, num_ep=10, deterministic=True, verbose=1, gif=False, render=False, log_smoothing=False):
    if model.__class__.__name__ in ("ModelBased", "ModelBasedEdge"):
        provide_env = True
    else:
        provide_env = False

    ep_rewards, ep_lens, ep_infos = list(), list(), list()
    mode = 'rgb_array' if gif else 'human'
    frames = list()
    for ep_index in range(num_ep):
        obs = env.reset()
        done = False
        ep_rew, ep_len = 0.0, 0
        while not done:
            if provide_env:
                action, _ = model.predict(obs, deterministic=deterministic, envs=env)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)            
            ep_len += 1
            ep_rew += reward
            if render:
                frames.append(env.render(mode=mode))
        ep_rewards.append(ep_rew)
        ep_lens.append(ep_len)
        ep_infos.append(info)
        if verbose:
            print("Finished Episode", ep_index + 1, "Reward:", ep_rew, "Length:", ep_len)
    
    print("Completed Eval of", num_ep, "Episodes")
    print("Avg. Reward:", np.mean(ep_rewards), "Avg. Length", np.mean(ep_lens))
    if log_smoothing:
        avg = np.mean(np.log(np.clip(ep_rewards, 1, None)))
    else:
        avg = np.mean(ep_rewards)
    return avg, frames

def test(name, num_ep=10, deterministic=True, verbose=1, gif=False, best=False, morphology_index=0):
    model, env, params = load_from_name(name, load_env=True, ret_params=True, morphology_index=morphology_index, best=best)
    print("Params:", params)
    _, frames = eval_policy(model, env, num_ep=num_ep, deterministic=deterministic, verbose=verbose, gif=gif, render=True)
    if gif:
        import imageio
        if name.endswith('/'):
            name = name[:-1]
        if name.startswith(BASE):
            # Remove the base
            name = name[len(BASE):]
            if name.startswith('/'):
                name = name[1:]
        render_path = os.path.join(RENDERS, name + '.gif')
        print("Saving gif to", render_path)
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        imageio.mimsave(render_path, frames[::5], subrectangles=True, duration=0.05)
    env.close()
    del model