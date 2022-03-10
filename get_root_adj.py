import json
import click
import tqdm


TARGET = {'black', 'clean', 'clear', 'cool', 'deep', 'dirty', 'dry', 'hard', 'tough', 'large', 'long', 'red', 'sharp',
          'short', 'smooth', 'soft', 'straight', 'strength', 'sweet', 'tight', 'warm', 'weak', 'white', 'wide'}


def is_root(sent: list[dict]) -> bool:
    for word in sent:
        if word['text'] in TARGET and word['upos'] == 'ADJ':
            if word['head'] != 0:
                if sent[word['head']-1]['upos'] == 'NOUN':
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
                if is_root(sent):
                    selected.append(sent)
    with open(out_jsonl_file, "w", encoding="utf-8") as out_f:
        for x in tqdm.tqdm(selected):
            out_f.write(json.dumps(x) + '\n')


if __name__ == "__main__":
    main()
