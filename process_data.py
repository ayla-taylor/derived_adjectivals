import json
import random
from typing import Any

import pandas as pd


SENSE_DICT = {'cool': {'Sense 1': 'cool_1', 'Sense 2': 'cool_2'},
              'hard': {'Sense 1': 'hard_1', 'Sense 2': 'hard_2', 'Sense 3': 'hard_3'}}
FILES = ['cool-annotation_annotations.json', 'hard-annotation_annotations.json']


def read_file(filename: str) -> dict:
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
    return data


def main():
    sentence_dicts = []
    filepath = '../data/Annotated/'
    # datafile = 'cool-annotation_annotations.json'
    for datafile in FILES:
        data = read_file(filepath + datafile)
        target_word = data['dataset']['name']
        sentences = data['examples']
        for sent in sentences:
            if sent['classifications']:
                data_sentence = sent['content']
                words = []
                for word in data_sentence.split():
                    # if word == "[" + target_word + "]":
                    #     words.append(word[1:-1])
                    # else:
                    #     words.append(word)
                    words.append(word)
                data_sentence = ' '.join(words)
                # tag = sent['classifications'][0]['classname']
                for classification in sent['classifications']:
                    if classification['correct']:
                        tag = classification['classname']
                # print(sent['classifications'])
                        sense = SENSE_DICT[target_word][tag]
                        sent_list = [target_word, data_sentence, sense]
                # print(sent_list)
                        sentence_dicts.append(sent_list)
    random.shuffle(sentence_dicts)

    train = pd.DataFrame(sentence_dicts[:int(len(sentence_dicts)*.8)], columns=['target', 'sentence', 'label'])
    dev = pd.DataFrame(sentence_dicts[int(len(sentence_dicts)*.8):int(len(sentence_dicts)*.9)], columns=['target', 'sentence', 'label'])
    test = pd.DataFrame(sentence_dicts[-int(len(sentence_dicts)*.1):], columns=['target', 'sentence', 'label'])
    splits = {'train': train, 'dev': dev, 'test': test}
    for split_name, split_df in splits.items():
        split_df.to_csv('../data/data/' + split_name + '.csv', index=False)


if __name__ == '__main__':
    main()
