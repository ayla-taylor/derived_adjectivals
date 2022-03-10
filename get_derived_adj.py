import stanza
import json
import tqdm
import click
import attrs

ADVERBS = {'recently', 'perfectly', 'completely', 'badly', 'quickly'}
NOT_AFTER = {'out', 'up'}
NOT_BEFORE = {}
# TARGETS = {'blackened', 'cleaned', 'cleared', 'cooled', 'deepen', 'dirtied', 'dried', 'hardened', 'toughened',
#            'enlarged', 'lengthened', 'reddened', 'sharpened', 'shortened', 'smoothed', 'softened', 'straightened',
#            'strengthened', 'sweetened', 'tighten', 'warmed', 'weakened', 'whitened', 'widened'}
# TARGET_ROOT = {'black', 'clean', 'clear', 'cool', 'deep', 'dirty', 'dry', 'hard', 'tough', 'large', 'long', 'red',
#                'sharp','short', 'smooth', 'soft', 'straight', 'strength', 'sweet', 'tight', 'warm', 'weak', 'white',
#                'wide'}
TARGET_FILES = ['root_adj_list.txt', 'root_verb_list.txt', 'double_derived_adj.txt']
TARGETS = set()


@attrs.define
class Word:
    id: int
    text: str
    upos: str
    xpos: str
    head: int


def is_derived(sent: list[dict]):
    words = dict()
    for word in sent:
        words[int(word['id'])] = Word(word['id'], word['text'], word['upos'], word['xpos'], word['head'])
    for i, word in words.items():
        if word.text in TARGETS:
            if word.head != 0 and words[word.head].upos == 'NOUN':
                if i + 1 < len(words) and words[i+1].text in NOT_AFTER:
                    return False
                else:
                    # print(words[i-1].text, word.text, words[i+1].text)
                    # print(words[word.head].text)
                    return True
            else:
                return False


def make_targetset(order: int) -> None:
    filename = TARGET_FILES[order]
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            TARGETS.add(line.strip())
    # return targetset


@click.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.argument("out_jsonl_file", type=click.Path())
@click.argument("order", type=int)
def main(json_file: str, out_jsonl_file: str, order: int):
    selected = []
    make_targetset(order)
    with open(json_file, encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            line_dict = json.loads(line)
            for sent in line_dict['parsed']:
                if is_derived(sent):
                    # print(sent)
                    selected.append(sent)
    with open(out_jsonl_file, "w", encoding="utf-8") as out_f:
        for x in tqdm.tqdm(selected):
            out_f.write(json.dumps(x) + '\n')


if __name__ == "__main__":
    main()
