"""This module parses all command line arguments to main.py"""
import argparse
import numpy as np

parser = argparse.ArgumentParser('Runs a neural net classifier')
parser.add_argument('--datadir', type=str, required=True, metavar='DIR',
        help='data storage directory')
parser.add_argument('--logdir', type=str, default="results/",metavar='DIR',
        help='directory for outputting results.')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
        help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
        help='Initial step size. (default: 0.01)')
parser.add_argument('--all', action='store_true',
        help='Create one model for all outputs if activated.')
