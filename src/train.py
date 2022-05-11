from typing import Any

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW, TrainingArguments, Trainer

LABELS = ['incorrect', 'correct']
CLASSLABEL = ClassLabel(num_classes=len(LABELS), names=list(LABELS))
F1_METRIC = load_metric('f1')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return F1_METRIC.compute(predictions=predictions, references=labels, average='micro')


def make_datasets(data: tuple[str, Any], tokenizer: BertTokenizer) -> DataLoader:
    def encode(examples):
        return tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=128)

    name, dataset = data
    inputs = dataset.map(encode, batched=True)
    # inputs = inputs.map(lambda examples: {'labels': examples['label']}, batched=True)
    inputs = inputs.remove_columns(['text1', 'text2'])
    inputs = inputs.rename_column('label', 'labels')
    inputs = inputs.with_format('torch')
    # inputs.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # dataloader = torch.utils.data.DataLoader(inputs, batch_size=32)
    return inputs


def train(datasets: dict, model: Any, tokenizer: Any):
    print("Training...")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.train().to(device)
    # optim = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    # for epoch in range(3):
    #     for i, batch in enumerate(tqdm(dataloader)):
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs[0]
    #         loss.backward()
    #         optim.step()
    #         optim.zero_grad()
    #         if i % 10 == 0:
    #             print(f"loss: {loss}")
    #
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model.to(device)

    args = TrainingArguments(
        '../model/glossbert_nsp_model/glossbert_model',
        evaluation_strategy="steps",
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # fp16=True
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['dev'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    print("Training...")
    trainer.train()

    # predictions = trainer.predict(datasets['dev'])
    # metrics =

    # print("Evaluating...")
    # trainer.evaluate(datasets['test'])



    # model.train()
    # optim = torch.optim.AdamW(params=model.parameters(), lr=5e-6)
    # epochs = 10
    # for epoch in range(epochs):
    #     loop = tqdm(dataloader, leave=True)
    #     for batch in loop:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs[0]
    #         loss.backward()
    #         optim.step()
    #         optim.zero_grad()
    #
    #         # optim.zero_grad()
    #         # input_ids = batch['input_ids'].to(device)
    #         # attention_mask = batch['attention_mask'].to(device)
    #         # labels = batch['labels'].to(device)
    #         #
    #         # outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #         # loss = outputs.loss
    #         # loss.backwards()
    #         # optim.step()
    #         loop.set_description(f'Epoch {epoch}')
    #         loop.set_postfix(loss=loss.item())


def main():
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('distilbert-base-uncased')
    filepath = '../data/data/glossbert/'
    dataset = load_dataset('csv', data_files={'train': filepath + 'train.csv',
                                              'dev': filepath + 'dev.csv',
                                              'test': filepath + 'test.csv'})
    # print(dataset)
    dataloaders = {}
    for name, data in dataset.items():
        dataloaders[name] = make_datasets((name, data), tokenizer)
    train(dataloaders, model, tokenizer)


if __name__ == '__main__':
    main()
