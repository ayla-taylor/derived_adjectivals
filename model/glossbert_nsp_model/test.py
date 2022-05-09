from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = 'distilbert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# results = classifier("the new student was [cool]")

# print(results)

tokens = tokenizer.tokenize("the new student was [cool]")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids2 = tokenizer("the new student was [cool]")
#
# print(f'Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')
# print(f'Token IDs2: {token_ids2}')

x_train = ["the new student was [cool]", "the cooled pie was ready to eat."]

batch = tokenizer(x_train, padding=True, truncation=True, max_length=512, return_tensors='pt')
print(batch)

with torch.no_grad():
    outputs = model(**batch, labels=torch.tensor([1,0]))
    print(f'outputs: {outputs}')

predictions = F.softmax(outputs.logits, dim=1)
print("predictions:", predictions)
labels = torch.argmax(predictions, dim=1)
print("labels", labels)
labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
print("labels to list: ", labels)

save_directory = "test"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tokenizer = AutoTokenizer.from_pretrained(save_directory)
