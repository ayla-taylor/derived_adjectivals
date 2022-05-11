import argparse


parser = argparse.ArgumentParser(description="Preprocess data for various models")

parser.add_argument('model', type=str, default='baseline', help='which model is this preprocessing for')

