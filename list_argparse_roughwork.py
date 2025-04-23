import argparse

parser = argparse.ArgumentParser(description="Train feed forward networks.")
parser.add_argument('--width', action='append', help="Width to add to the list of widths (required).", required=True)

args = parser.parse_args()

print('args.width = ', args.width)
print('type(args.width) = ', type(args.width))
print('type(args.width[0]) = ', type(args.width[0]))