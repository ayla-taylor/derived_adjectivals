import json
import random
from typing import Any

import pandas as pd

DATAFILES = ['cool-annotation_annotations.json', 'hard-annotation_annotations.json']
SENSE_DICT = {'cool': {'Sense 1': 'cool: trendy, fashonable, interesting',
                       'Sense 2': 'cool: of or at a relatively low temperature'},
              'hard': {'Sense 1': 'hard: difficult',
                       'Sense 2': 'hard: Forceful, potent, vigorous',
                       'Sense 3': 'hard: Firm, solid, resistant to pressure, tangible'}}


def read_file(filename: str) -> dict:
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
    return data


def main():
    filepath = '../data/Annotated/'
    sentence_dicts = []
    for datafile in DATAFILES:
        data = read_file(filepath + datafile)
        target_word = data['dataset']['name']
        sentences = data['examples']
        for sent in sentences:
            if sent['classifications']:
                data_sentence = sent['content']
                # words = []
                # for word in data_sentence.split():
                #     # if word == "[" + target_word + "]":
                #     #     words.append(word[1:-1])
                #     # else:
                #     #     words.append(word)
                #     words.append(word)
                # data_sentence = ' '.join(words)
                # tag = sent['classifications'][0]['classname']
                for classification in sent['classifications']:
                    if classification['correct']:
                        tag = classification['classname']
                        for sense_tag, sense_def in SENSE_DICT[target_word].items():
                            if sense_tag == tag:
                                sentence_dicts.append({'text1': data_sentence, 'text2': sense_def, 'label': 1})
                            else:
                                sentence_dicts.append({'text1': data_sentence, 'text2': sense_def, 'label': 0})
    random.shuffle(sentence_dicts)

    train = pd.DataFrame(sentence_dicts[:int(len(sentence_dicts) * .8)], columns=['text1', 'text2', 'label'])
    dev = pd.DataFrame(sentence_dicts[int(len(sentence_dicts) * .8):int(len(sentence_dicts) * .9)], columns=['text1', 'text2', 'label'])
    test = pd.DataFrame(sentence_dicts[-int(len(sentence_dicts) * .1):], columns=['text1', 'text2', 'label'])
    splits = {'train': train, 'dev': dev, 'test': test}
    # splits_fixed = dict()
    # for split_name, split_list in splits.items():
    #     split_dict = {'text1': [], 'text2': []}
    #     for sent_dict in split_list:
    #         split_dict['text1'].append(sent_dict['text1'])
    #         split_dict['text2'].append(sent_dict['text2'])
    #     split_df = pd.DataFrame
    #     splits_fixed[split_name] = split_dict

    # train = pd.DataFrame(sentence_dicts[:int(len(sentence_dicts) * .8)], columns=['text', 'text2'])
    # dev = pd.DataFrame(sentence_dicts[int(len(sentence_dicts) * .8):int(len(sentence_dicts) * .9)], columns=['text', 'text2'])
    # test = pd.DataFrame(sentence_dicts[-int(len(sentence_dicts) * .1):], columns=['text', 'text2'])
    # print(test)

    for split_name, split_df in splits.items():
        # with open ('../data/data/glossbert/'+ split_name + '.json', 'w', encoding='utf8') as f:
        split_df.to_csv('../data/data/glossbert/' + split_name + '.csv', index=False)
            # json.dump(split_d, f)


if __name__ == '__main__':
    main()
