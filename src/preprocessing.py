import argparse
from preprocessing_utils import read_sense_dict


parser = argparse.ArgumentParser(description="Preprocess data for various models")

parser.add_argument('model', type=str, default='baseline', help='which model is this preprocessing for')
args = parser.parse_args()

SENSE_DICT_FILE = "../data/sense_dict.json"


def process_datafiles(datafiles: list[str]) -> None:
    filepath = '../data/'
    for data in datafiles:


def main():
    sense_dict = read_sense_dict(SENSE_DICT_FILE)
    if args.model == "baseline":
        datafiles = []
    elif args.model == "derived_embed":
        datafiles = []
    elif args.model == "full_model":
        datafiles = []
    process_datafiles(datafiles)

if __name__ == '__main__':
    main()
