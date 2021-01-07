import argparse
import matplotlib.pyplot as plt
from optimal_agents.utils import plotter

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str, nargs='+')
parser.add_argument("--legend", "-l", type=str, default=None, nargs='+')
parser.add_argument("--n", "-n", type=int, default=None)
parser.add_argument("--title", "-t", type=str, default=None)
parser.add_argument("--use-wall-time", "-w", action='store_true', default=False)
parser.add_argument("--use-episodes", "-e", action='store_true', default=False)
parser.add_argument("--yaxis", "-y", type=str, default='r')
parser.add_argument("--subsample", type=int, default=None)
parser.add_argument("--individual", "-i", action='store_true', default=False)
args = parser.parse_args()

if args.title:
    title = args.title
else:
    title = args.path

if args.use_wall_time:
    print("Using time")
    xaxis = plotter.X_WALLTIME
elif args.use_episodes:
    print("Using episodes")
    xaxis = plotter.X_EPISODES
else:
    print("Using timesteps")
    xaxis = plotter.X_TIMESTEPS

plotter.generate_plots(args.path, xaxis=xaxis, yaxis=args.yaxis, title=args.title, num_timesteps=args.n, subsample=args.subsample, labels=args.legend, individual=args.individual)
plt.show()

