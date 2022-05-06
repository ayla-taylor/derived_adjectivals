from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, Trainer, TrainingArguments, BertForNextSentencePrediction
import numpy as np
import torch


# SENSE_LABELS = ['True', 'False' ]  # ?
F1_METRIC = load_metric('f1')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return F1_METRIC.compute(predictions=predictions, references=labels, average='micro')


def make_datasets():
    filepath = '../data/data/glossbert/'
    dataset = load_dataset('csv', data_files={'train': filepath + 'train.csv',
                                              'dev': filepath + 'dev.csv',
                                              'test': filepath + 'test.csv'})
    # class_labels = ClassLabel(num_classes=len(SENSE_LABELS), names=list(SENSE_LABELS))
    print(dataset['train'])
    #
    # def label_str2int(examples):
    #     return {'label': class_labels.str2int(examples['label'])}

    # convert labels to integer ids
    # dataset = dataset.map(label_str2int, batched=False)
    # print("mapped dataset:",dataset)

    # load pretrained transformer and coresponding tokenizer
    model = BertForNextSentencePrediction.from_pretrained('kanishka/GlossBERT')  #, num_labels=class_labels.num_classes)
    tokenizer = BertTokenizer.from_pretrained('kanishka/GlossBERT')

    print('tokenizing...')


    # def encode(examples):
    #     return tokenizer(examples['text'], examples['text2'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)

    inputs = tokenizer(dataset['features']['text'], dataset['features']['text2'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    print(inputs)
    # print(dataset['train'].features['target'])

    # new_features = encoded_dataset['train'].features.copy()
    # new_features['label'] = class_labels
    # encoded_dataset = encoded_dataset.cast(new_features) #cast label from into to ClassLabel
    # print('encoded_dataset:', encoded_dataset)
    # args = TrainingArguments(
    #     'sense-disambiguation',
    #     evaluation_strategy="steps",
    #     learning_rate=2e-5,
    #     num_train_epochs=5,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     # fp16=True
    # )
    #
    # trainer = Trainer(
    #     model,
    #     args,
    #     train_dataset=encoded_dataset['train'],
    #     eval_dataset=encoded_dataset['dev'],
    #     compute_metrics=compute_metrics
    # )
    print("Training...")
    labels = torch.LongTensor([0])
    outputs = model(**inputs, labels=labels)
    print('loss:', outputs.loss.items())
    # trainer.train()

    # tf_train_set =


if __name__ == "__main__":
    make_datasets()