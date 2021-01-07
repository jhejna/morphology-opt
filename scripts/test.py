import argparse
import os

from optimal_agents.utils.tester import test
from optimal_agents.utils.loader import BASE

parser = argparse.ArgumentParser()
parser.add_argument('--name', "-n", help='name of checkpoint model', required=True)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', "-p", help='name of checkpoint model', required=True)
    parser.add_argument('--episodes', "-e", type=int, default=10)
    parser.add_argument('--gif', "-g", action='store_true', default=False)
    parser.add_argument('--undeterministic', "-u", action='store_false', default=True)
    parser.add_argument('--morphology', "-m", type=int, default=0, help="Morphology index")
    parser.add_argument('--best', "-b", action='store_true', default=False, help="Load best policy by eval.")
    args = parser.parse_args()

    # Auto forward the path until we reach a model.
    folder = args.path
    if not folder.startswith('/'):
        folder = os.path.join(BASE, folder)
    while 'params.json' not in os.listdir(folder):
        contents = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        assert len(contents) == 1, "Traversing down directory with multiple paths:" + " ".join(contents)
        folder = os.path.join(folder, contents[0])
    
    test(folder, num_ep=args.episodes, gif=args.gif, deterministic=args.undeterministic, morphology_index=args.morphology)

if __name__ == '__main__':
    main()