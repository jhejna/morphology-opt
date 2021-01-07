import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from stable_baselines3.common.results_plotter import window_func, load_results
from optimal_agents.utils.loader import BASE, Parameters

'''
This file contains code used to generate plots as seen in the paper.
In order to generate plots like in the paper, the file structure must be as follows for evolution:

Experiment
    - Seed 1
        - Eval Env 1 
        - Eval Env 2
        ...
        - Eval Env n
    - Seed 2
        - Eval Env 1 
        - Eval Env 2
        ...
        - Eval Env n
    ...
    - Seed n
        ...

For regular episode plotting, the structure should be:
Experiment:
- Seed 1
- Seed 2
...
- Seed n

If the files are not structured like this, the plotting is not going to work as intended.
'''


EPISODES_WINDOW = 25

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]

def ts2xy(timesteps, xaxis, yaxis='r'):
    """
    Modified to let you get keyword values.
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
    y_var = timesteps[yaxis].values
    return x_var, y_var

def get_subdirs(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def generate_plots(dirs, xaxis=X_TIMESTEPS, yaxis='r', title=None, labels=None, num_timesteps=None, subsample=None, individual=False):
    for i in range(len(dirs)):
        if not dirs[i].startswith('/'):
            dirs[i] = os.path.join(BASE, dirs[i])
    
    # If pointing to a single folder and that folder has many results, use that as dir
    if len(dirs) == 1 and len(get_subdirs(dirs[0])) > 1:
        dirs = [os.path.join(dirs[0], subdir) for subdir in get_subdirs(dirs[0])]
        dirs = sorted(dirs)
    
    # Make everything reproducible by sorting. Can comment out later for organization.
    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in dirs]

    dirs, labels = zip(*sorted(zip(dirs, labels), key=lambda x: x[0]))
    plt.gcf().dpi = 100.0
    print("Dirs", dirs)
    for i, folder in enumerate(dirs):
        if not 'params.json' in os.listdir(folder):
            # If directory contains 1 folder, and none of those folders have params.json, move down.
            while True:
                contents = get_subdirs(folder)
                if any(['params.json' in os.listdir(os.path.join(folder, c)) for c in contents]):
                    break
                folder = os.path.join(folder, contents[0])
        
        if not 'params.json' in os.listdir(folder):
            runs = sorted([os.path.join(folder, r) for r in get_subdirs(folder)])
        else:
            runs = [folder]

        print("Different seeds for folder", folder, ":")
        print(runs)
        print("----")

        sns.set_context(context="paper", font_scale=1.5)
        sns.set_style("darkgrid", {'font.family': 'serif'})
        xlist, ylist = [], []
        for run in runs:
            timesteps = load_results(run)
            if num_timesteps is not None:
                timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
            x, y = ts2xy(timesteps, xaxis, yaxis=yaxis)
            if x.shape[0] >= EPISODES_WINDOW:
                x, y = window_func(x, y, EPISODES_WINDOW, np.mean)
            xlist.append(x)
            ylist.append(y)
        if individual:
            for i, (xs, ys, run) in enumerate(zip(xlist, ylist, runs)):
                g = sns.lineplot(x=xs, y=ys, label=os.path.basename(run))
        else:
            # Zero-order hold to align the data for plotting
            joint_x = sorted(list(set(np.concatenate(xlist))))
            combined_x, combined_y = [], []
            for xs, ys in zip(xlist, ylist):
                cur_ind = 0
                zoh_y = []
                for x in joint_x:
                    # The next value matters
                    if cur_ind < len(ys) - 1 and x >= xs[cur_ind + 1]:
                        cur_ind += 1
                    zoh_y.append(ys[cur_ind])
                if subsample:
                    combined_x.extend(joint_x[::subsample])
                    combined_y.extend(zoh_y[::subsample])
                else:
                    combined_x.extend(joint_x)
                    combined_y.extend(zoh_y)
            data = pd.DataFrame({xaxis : combined_x, yaxis: combined_y})
            g = sns.lineplot(x=xaxis, y=yaxis, data=data, ci=None, sort=True, label=labels[i])

        print("Completed folder", folder)

    if title:
        plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout(pad=0)

def ea_plot(dirs, title=None, labels=None, window=1, individual=False):
    for i in range(len(dirs)):
        if not dirs[i].startswith('/'):
            dirs[i] = os.path.join(BASE, dirs[i])

    # For legacy plotting behavior, remove this if statement
    if len(dirs) == 1 and len(get_subdirs(dirs[0])) > 1:
        dirs = [os.path.join(dirs[0], subdir) for subdir in get_subdirs(dirs[0])]
        dirs = sorted(dirs)
        # assert labels is None, "Single Directory Plot with runs can't have labels."

    # Make everything reproducible by sorting. Can comment out later for organization.
    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in dirs]

    dirs, labels = zip(*sorted(zip(dirs, labels), key=lambda x: x[0]))
    # print(dirs, labels)
    plt.gcf().dpi = 100.0
    for i, folder in enumerate(dirs):
        sns.set_context(context="paper", font_scale=1.5)
        sns.set_style("darkgrid", {'font.family': 'serif'}) 
        if 'gen0.txt' not in os.listdir(folder):
            # We have multiple runs
            runs = sorted([os.path.join(folder, r) for r in get_subdirs(folder)])
        else:
            runs = [folder]

        xs, ys = [], []
        for run in runs:
            assert 'gen0.txt' in os.listdir(run), "{run} did not contain gen0.txt"
            gen_records = [f for f in os.listdir(run) if f.endswith('.txt') and f.startswith('gen')]
            gen_records.sort(key=lambda x: int(x[3:-4]))
            x, y = [], []
            for j, gen_record in enumerate(gen_records):
                with open(os.path.join(run, gen_record)) as f:
                    fitnesses = []
                    for _ in range(window):
                        fitnesses.append(float(f.readline().split(' ')[0]))
                    x.append(j+1)
                    y.append(np.mean(fitnesses))
            if individual:
                sns.lineplot(x=x, y=y, label=os.path.basename(run))
            else:
                xs.extend(x)
                ys.extend(y)

        if not individual:
            data = pd.DataFrame({'gen' : xs, 'reward': ys})
            sns.lineplot(x='gen', y='reward', data=data, ci="sd", sort=True, label=labels[i])

        print("Completed folder", folder)
    
    if title:
        plt.title(title)
    plt.xlabel('generation')
    plt.ylabel('Fitness')
    plt.tight_layout(pad=0)

def percentile_plot(dirs, title=None, labels=None, curve=False):
    for i in range(len(dirs)):
        if not dirs[i].startswith('/'):
            dirs[i] = os.path.join(BASE, dirs[i])

    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in dirs]

    categorical_x = []
    categorical_y = []
    for i, directory in enumerate(dirs):
        all_runs = []
        folders = [os.path.join(directory, subdir) for subdir in get_subdirs(directory)]
        rewards = []
        for folder in folders:
            runs = []
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isdir(item_path) and '0.monitor.csv' in os.listdir(item_path):
                    runs.append(item_path)
            returns = []
            print("Different evals for folder", folder, "Please ensure this has ALL intended environments or results may skew:")
            print(runs)
            print("----")
            for run in runs:
                timesteps = load_results(run)
                x, y = ts2xy(timesteps, X_TIMESTEPS, yaxis='r')
                if x.shape[0] >= EPISODES_WINDOW:
                    x, y = window_func(x, y, EPISODES_WINDOW, np.mean)
                if len(y) > 0:
                    returns.append(max(y))
                else:
                    print("Was Error on run", run)
            avg_reward = np.mean(returns)
            print(folder, avg_reward)
            rewards.append(avg_reward)
        print("Seeds for", directory, ":", len(rewards), "of", len(folders))
        print("Stats:")
        rounded_mean = round(np.mean(rewards),1)
        rounded_confidence = round(np.std(rewards)/np.sqrt(len(rewards)),1)
        print("$" + str(rounded_mean) + " \pm " + str(rounded_confidence) + "$")

        if curve:
            x = np.sort(rewards)
            y = 1 - (np.arange(0, len(x)) / len(x))
            sns.set_context(context="paper", font_scale=1.5)
            sns.set_style("darkgrid", {'font.family': 'serif'})
            sns.scatterplot(x=x, y=y, label=labels[i])
        else:
            categorical_x.extend([labels[i]]*len(rewards))
            categorical_y.extend(rewards)

    if curve:
        plt.xlabel('reward')
        plt.ylabel('% Above threshold')
    else:
        sns.set(rc={'figure.figsize':(6.4,4.8)})
        sns.set_context(context="paper", font_scale=1.4)
        sns.set_style("darkgrid", {'font.family': 'serif'})
        sns.boxplot(x=categorical_x, y=categorical_y, whis=1.5, showcaps=False, showfliers=True, saturation=0.7, width=0.9)
        sns.swarmplot(x=categorical_x, y=categorical_y, color="0.25")
        plt.ylabel('Reward')
    if title:
        plt.title(title)
    plt.tight_layout(pad=0)


