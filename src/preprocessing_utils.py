import json


def read_sense_dict(filename: str) -> dict:
    with open(filename, emcoding='utf8') as f:
        sense_dict = json.load(f)
    return sense_dict


