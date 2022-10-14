import os
import json
import spacy
import tqdm


WORD_LISTS = ['root_adj_list.txt', 'root_verb_list.txt', 'double_derived_adj.txt']
SOURCE_FILES = ['c4-train.00001-of-01024_adj_ayla_2_filtered.json',
               'c4-train.00001-of-01024_adj_filtered.json',
               'c4-train.00001-of-01024_ayla_v3_filtered.json']

spacy_nlp = spacy.load("en_core_web_sm", exclude=['ner', 'lemmatizer', 'textcat'])


def match_with_words(parsed_doc, word_dicts: list[dict[str:str]]) -> None:
    for word in parsed_doc.doc:
        for word_dict in word_dicts:
            if word.text in set(word_dict.keys()):
                per_word_dict = {'target_word': word.text,
                                 'words': [i.text for i in parsed_doc],
                                 'pos': [i.pos_ for i in parsed_doc],
                                 'tags': [i.tag_ for i in parsed_doc],
                                 'heads': [i.head.text for i in parsed_doc],
                                 'deps': [i.dep_ for i in parsed_doc],
                                 'sent_start': [i.is_sent_start for i in parsed_doc],
                                 'text': parsed_doc.text}
                word_dict[word.text] += json.dumps(per_word_dict) + '\n'


def get_c4_text(jsonl_file: str, word_dicts: list[dict]) -> None:
    for line in tqdm.tqdm(open(jsonl_file, "r", encoding="utf-8")):  # encoding='ISO-8859-1'?
        line_dict = json.loads(line)
        doc = line_dict["passage"]
        parsed_doc = spacy_nlp(doc)
        match_with_words(parsed_doc, word_dicts)


def main():
    dirs = []
    word_dicts = []
    for filename in WORD_LISTS:
        print(f'Extracting data from {filename}...')
        file_split = filename.split('_')
        dir_name = '_'.join([file_split[0], file_split[1]])
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        dirs.append(dir_name+'/')
        words = dict()
        assert os.path.isfile(filename)
        with open(filename, 'r', encoding='utf-8') as wf:
            for line in wf:
                words[line.strip()] = ''
        word_dicts.append(words)
    for source in tqdm.tqdm(SOURCE_FILES):
        get_c4_text(source, word_dicts)
    print("writing to files....")
    for i, dir_name in enumerate(dirs):
        for word in word_dicts[i].keys():
            with open(dir_name+word+'.json', 'w', encoding='utf-8') as out_f:
                out_f.write(word_dicts[i][word])


if __name__ == '__main__':
    main()
