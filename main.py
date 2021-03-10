"""Main script to run all components."""
import argparse
from algorithms import *
from environment import Room

parser = argparse.ArgumentParser()

parser.add_argument('--room_l', type=float, default=10)
parser.add_argument('--room_b', type=float, default=20)
parser.add_argument('--room_h', type=float, default=15)
...


def optimise(args):
    pass

if __name__=='__main__':
    args = parser.parse_args()

    optimise(args)
