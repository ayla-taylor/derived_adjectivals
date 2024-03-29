import argparse
import random

import pandas as pd

from preprocessing_utils import read_sense_dict, read_file, root_to_derived, get_head

# parser = argparse.ArgumentParser(description="Preprocess data for various models")
#
# parser.add_argument('model', type=str, default='baseline', help='which model is this preprocessing for')
# args = parser.parse_args()

SENSE_DICT_FILE = "../../data/sense_dict.json"


def process_model_files(datafiles: list[str], sense_dict: dict) -> None:
    filepath = '../../data/Annotated/'
    sentence_dicts = []
    for datafile in datafiles:
        data = read_file(filepath + datafile)
        target_word = data['dataset']['name']
        sentences = data['examples']
        for sent in sentences:

            data_sentence = sent['content']
            if sent['classifications']:
                for classification in sent['classifications']:
                    if classification['correct']:
                        sentence_dict = {}
                        tag = classification['classname']
                        for sense_tag, sense_def in sense_dict[target_word].items():
                            sentence_dict['text1'], sentence_dict['text2'] = data_sentence, sense_def
                            if sense_tag == tag:
                                sentence_dict['label'] = 1
                            else:
                                sentence_dict['label'] = 0
                        if sent['annotations']:
                            for annotation in sent['annotations']:
                                if annotation['correct']:
                                    sentence_dict['derived_pair'] = root_to_derived[target_word] + ' ' + annotation['value']
                        else:
                            sentence_dict['derived_pair'] = get_head(data_sentence, target_word)
                        sentence_dicts.append(sentence_dict)
    random.shuffle(sentence_dicts)
    baseline_sents, full_sents = sentence_dicts[:int(len(sentence_dicts)/2)], sentence_dicts[int(len(sentence_dicts)/2):]
#    for sent in models_sentences[1]:
#        sent['text1'] = sent['derived_pair'] + ' | ' + sent['text1']
    split_for_each_model = []

    train = pd.DataFrame(baseline_sents[:int(len(baseline_sents) * .8)], columns=['text1', 'text2', 'label'])
    dev = pd.DataFrame(baseline_sents[int(len(baseline_sents) * .8):int(len(sentence_dicts) * .9)],
                       columns=['text1', 'text2', 'label'])
    test = pd.DataFrame(baseline_sents[-int(len(baseline_sents) * .1):], columns=['text1', 'text2', 'label'])
    split_for_each_model.append({'train': train, 'dev': dev, 'test': test})

    train = pd.DataFrame(full_sents[:int(len(full_sents) * .8)], columns=['text1', 'text2', 'derived_pair', 'label'])
    dev = pd.DataFrame(full_sents[int(len(full_sents) * .8):int(len(sentence_dicts) * .9)],
                       columns=['text1', 'text2', 'derived_pair', 'label'])
    test = pd.DataFrame(full_sents[-int(len(full_sents) * .1):], columns=['text1', 'text2', 'derived_pair', 'label'])
    split_for_each_model.append({'train': train, 'dev': dev, 'test': test})

    baseline_split, fullmodel_split = split_for_each_model
    for split_name, split_df in baseline_split.items():
        split_df.to_csv('../data/baseline_model/' + split_name + '.csv', index=False)
    for split_name, split_df in fullmodel_split.items():
        split_df.to_csv('../data/full_model/' + split_name + '.csv', index=False)


def main():
    sense_dict = read_sense_dict(SENSE_DICT_FILE)

    # if args.embed_data:
    #     datafiles = [['cooled-annotation-2_annotations.json',
    #                   'blackened-annotation_annotations.json',
    #                   'softened-annotation_annotations.json'],
    #                  ['cooled-sense-annotation-only_annotations.json',
    #                   'hardened-sense-annotation_annotations.json']]
    #     process_embed_data(datafiles)
    datafiles = ['cool-annotation_annotations.json',
                 'hard-annotation_annotations.json',
                 'black-annotations_annotations.json']
    process_model_files(datafiles, sense_dict)


if __name__ == '__main__':
    main()
