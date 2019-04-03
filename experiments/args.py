import argparse

parser = argparse.ArgumentParser(description='Run an experiment.')
parser.add_argument('name', metavar='EXPERIMENT_NAME', type=str,
                    help='A name for the experiment')
parser.add_argument('--epoch', metavar='E', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--prune', metavar='P', type=int, default=10,
                    help='Number of pruning iterations')


args = parser.parse_args()
print(args.name)
print(args.epoch)
print(args.prune)