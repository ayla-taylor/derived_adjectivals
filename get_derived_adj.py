import stanza
import json
import tqdm
import click
import attrs

ADVERBS = {'recently', 'perfectly', 'completely', 'badly', 'quickly'}
NOT_AFTER = {'out', 'up'}
NOT_BEFORE = {}
TARGETS = {'blackened', 'cleaned', 'cleared', 'cooled', 'deepen', 'dirtied', 'dried', 'hardened', 'toughened',
           'enlarged', 'lengthened', 'reddened', 'sharpened', 'shortened', 'smoothed', 'softened', 'straightened',
           'strengthened', 'sweetened', 'tighten', 'warmed', 'weakened', 'whitened', 'widened'}


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
        if word.text in TARGETS and word.head != 0 and words[word.head].upos == 'NOUN':
            if i + 1 < len(words) and words[i+1].text in NOT_AFTER:
                return False
            else:
                return True


@click.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.argument("out_jsonl_file", type=click.Path())
def main(json_file: str, out_jsonl_file: str):
    selected = []
    with open(json_file, encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            line_dict = json.loads(line)
            for sent in line_dict['parsed']:
                if is_derived(sent):
                    selected.append(sent)
    with open(out_jsonl_file, "w", encoding="utf-8") as out_f:
        for x in tqdm.tqdm(selected):
            out_f.write(json.dumps(x) + '\n')


if __name__ == "__main__":
    main()
