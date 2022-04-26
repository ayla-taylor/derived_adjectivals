import sys
import nltk
from nltk.corpus import wordnet as wn


nltk.download('omw-1.4')


def get_synset(words: set[str]) -> dict:
    synset_dict = dict()
    for word in words:
        synsets = wn.synsets(word, pos=wn.ADJ)
        # print(word, synsets)
        synset_dict[word] = synsets
    return synset_dict


def read_to_set(filename: str) -> set[str]:
    word_set = set()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            word_set.add(line.strip())
    # print(word_set)
    return word_set


def main():
    csi_inventory = read_to_set('csi_data/inventory.txt')
    target_words = read_to_set(sys.argv[1])
    word_dict = get_synset(target_words)
    csi_match_dict = dict()
    with open('csi_data/wn_synset2csi.txt', encoding='utf8') as f:
        for line in f:
            look_up = line.strip().split()[0].split(':')[-1]
            cluster = line.strip().split()[1:]
            offset, pos = int(look_up[:-1]), look_up[-1]
            synset = wn.synset_from_pos_and_offset(pos, offset)
            csi_match_dict[synset] = cluster
    unable_to_find = []
    with open('clustered_senses.tsv', 'w', encoding='utf8') as out_f:
        for word in target_words:
            for synset in word_dict[word]:
                if synset in set(csi_match_dict.keys()):
                    print('\t'.join([word, str(csi_match_dict[synset]), synset.definition(), str(synset.examples())]), file=out_f)
                    print(synset.examples())
                    print(synset.definition())
                else:
                    print('\t'.join([word, "[None]", synset.definition(), str(synset.examples())]), file=out_f)

                    # print("skipped", word, synset)
                    unable_to_find.append((word, synset))
            # print(csi_match_dict[synset])
        # print(word, word_dict[word])


if __name__ == '__main__':
    main()
