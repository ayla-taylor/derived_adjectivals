from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric, ClassLabel, Value


DATAFILE = '../data/embedding_train.csv'
MODEL = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
TOKENIZER = AutoTokenizer.from_pretrained('distilbert-base-uncased')


def encode(examples):
    return TOKENIZER(examples['text'], truncation=True, padding='max_length', max_length=128)


def main():
    # model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = load_dataset('csv', data_files=[DATAFILE])['train']
    print(dataset.features)
    dataset = dataset.map(encode, batched=True)
    print(dataset.features)
    inputs = dataset.remove_columns(['text'])
    inputs = inputs.rename_column('label', 'labels')
    inputs = inputs.with_format('torch')
    print(inputs)


if __name__ == '__main__':
    main()
