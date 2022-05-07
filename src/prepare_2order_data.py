import json

FILEPATH = '../data/Annotated/'
FILES = ['cooled-annotation-2_annotations.json',
         'blackened-annotation_annotations.json',
         'softened-annotation_annotations.json'
]


def read_files(filelist: list[str]) -> list[dict]:
    data = []
    for filename in filelist:
        with open(FILEPATH + filename, 'r', encoding='utf8') as f:
            for line in f:
                data = json.loads(line.strip())
                for example in data['examples']:
                    start, end = data[annot]
                    data_dict = {'text': data['content']

                }
    return data


def main():
    data = read_files(FILES)
    print(data)

if __name__ == '__main__':
    main()
