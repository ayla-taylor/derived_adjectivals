from datasets import load_dataset, load_metric, ClassLabel, Value
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW
import numpy as np
import torch
from tqdm import tqdm


# SENSE_LABELS = ['True', 'False' ]  # ?
F1_METRIC = load_metric('f1')

class DerivedAdjDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return F1_METRIC.compute(predictions=predictions, references=labels, average='micro')


def make_datasets():
    filepath = '../../data/data/glossbert/'
    dataset = load_dataset('csv', data_files={'train': filepath + 'train.csv',
                                              'dev': filepath + 'dev.csv',
                                              'test': filepath + 'test.csv'})
    # class_labels = ClassLabel(num_classes=len(SENSE_LABELS), names=list(SENSE_LABELS))
    # print(dataset['train'].features)
    #
    # def label_str2int(examples):
    #     return {'label': class_labels.str2int(examples['label'])}

    # convert labels to integer ids
    # dataset = dataset.map(label_str2int, batched=False)
    # print("mapped dataset:",dataset)

    # load pretrained transformer and coresponding tokenizer
    model = BertForNextSentencePrediction.from_pretrained('distilbert-base-uncased')  #, num_labels=class_labels.num_classes)
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

    print('tokenizing...')

    def encode(examples):
        # print(examples)
        return tokenizer(examples['text1'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)

    inputs = dataset.map(encode)  #, batched=True)
    # inputs = tokenizer(dataset, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    print(inputs)
    inputs['train'].features['labels'] = torch.LongTensor([inputs['train'].features['label']]).T
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



    dataset = DerivedAdjDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.train()
    optim = AdamW(model.parameters, lr=5e-6)
    epochs = 10
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backwards()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


if __name__ == "__main__":
    make_datasets()
