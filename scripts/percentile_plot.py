import argparse
import matplotlib.pyplot as plt
from optimal_agents.utils import plotter

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str, nargs='+')
parser.add_argument("--title", "-t", type=str, default=None)
parser.add_argument("--legend", "-l", type=str, default=None, nargs='+')
parser.add_argument("--curve", "-c", action='store_true', default=False)
args = parser.parse_args()

if args.title:
    title = args.title
else:
    title = args.path

plotter.percentile_plot(args.path, title=args.title, labels=args.legend, curve=args.curve)
plt.show()

