import stanza
import json
import tqdm
import click
import attrs

ADVERBS = {'recently', 'perfectly', 'completely', 'badly', 'quickly'}
NOT_AFTER = {'dried': {'out', 'up'}}
NOT_BEFORE = {}

TARGET_DERIVED = {'dried'}
TARGET_ROOT_ADJ = {'dry'}



@attrs.define
class Word:
    id: int
    text: str
    upos: str
    xpos: str
    head: int


def is_derived(sent: list[dict]):
    target = 'dried'
    words = dict()
    for word in sent:
        words[int(word['id'])] = Word(word['id'], word['text'], word['upos'], word['xpos'], word['head'])
    for i, word in words.items():
        if word.text == target and word.upos == 'VERB':
            if i + 1 < len(words) and words[i+1].text in NOT_AFTER[target]:
                return False
            if i > 1 and words[i-1].upos == 'DET':
                return True
            if i > 2 and words[i-1].upos == 'ADV' and words[i-2].upos == 'DET':
                # if i + 1 < len(words) and words[i - 1].upos == 'NOUN':
                return True
            # if i + 1 < len(words) and words[i-1].upos == 'NOUN':
            #     if


# @click.command()
# @click.argument("json_file", type=click.Path(exists=True))
# @click.argument("out_jsonl_file", type=click.Path())
def main(json_file: str, out_jsonl_file: str):
    selected = []
    with open(json_file, encoding='utf8') as f:
        for line in f:
            line_dict = json.loads(line)
            for sent in line_dict['parsed']:
                if is_derived(sent):
                    selected.append(sent)
    with open(out_jsonl_file, "w", encoding="utf-8") as out_f:
        for x in selected:
            out_f.write(json.dumps(x) + '\n')


if __name__ == "__main__":
    main('pos_tagged_mini.json', 'extracted_mini.json')
