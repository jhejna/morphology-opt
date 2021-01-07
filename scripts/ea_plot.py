import argparse
import matplotlib.pyplot as plt
from optimal_agents.utils import plotter

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str, nargs='+')
parser.add_argument("--legend", "-l", type=str, default=None, nargs='+')
parser.add_argument("--title", "-t", type=str, default=None)
parser.add_argument("--window", "-w", type=int, default=1)
parser.add_argument("--individual", "-i", action='store_true', default=False)
parser.add_argument("--timesteps", "-s", action='store_true', default=False)
args = parser.parse_args()

plotter.ea_plot(args.path, title=args.title, labels=args.legend, window=args.window, individual=args.individual, use_timesteps=args.timesteps)
plt.show()
