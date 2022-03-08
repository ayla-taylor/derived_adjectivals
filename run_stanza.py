import stanza
import json
import tqdm
import click
import attrs

stanza.download('en')

nlp = stanza.Pipeline(
    lang="en", processors="tokenize,pos", tokenize_pretokenized=False
)


def get_c4_text(jsonl_file: str):
    for line in open(jsonl_file, "r", encoding="utf-8"):
        line_dict = json.loads(line)
        yield line_dict, line_dict["passage"]


# @click.command()
# @click.argument("json_file", type=click.Path(exists=True))
# @click.argument("out_jsonl_file", type=click.Path())
def run_pipeline(json_file: str, out_jsonl_file: str):
    with open(out_jsonl_file, "w", encoding="utf-8") as out_f:
        for i, (doc_dict, doc) in enumerate(tqdm.tqdm(list(get_c4_text(json_file)), desc="parse C4 document")):
            parsed_doc = nlp(doc)
            doc_dict["parsed"] = parsed_doc.to_dict()
            out_f.write(json.dumps(doc_dict) + "\n")


if __name__ == "__main__":
    run_pipeline('data/c4-train.00001-of-01024_adj_filtered.json', 'pos_tagged.json')
