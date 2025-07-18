
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

#Read csv files
df_train = pd.read_csv("./2018-E-c-En-train.csv", sep=';')
df_eval = pd.read_csv("./2018-E-c-En-dev.csv", sep=';')

#Get label and turn it to list
not_chosen_columns = ['ID', 'Tweet']
label_columns = [col for col in df_train.columns if col not in not_chosen_columns]

df_train[label_columns] = df_train[label_columns].astype(float) #Typecast into float
df_eval[label_columns] = df_eval[label_columns].astype(float) #Typecast into float

df_labels_train = df_train[label_columns]
df_labels_eval = df_eval[label_columns]

list_labels_train = df_labels_train.values.tolist()
list_labels_eval = df_labels_eval.values.tolist()

#Get datas for training
train_texts = df_train['Tweet'].tolist()
train_labels = list_labels_train

eval_texts = df_eval['Tweet'].tolist()
eval_labels = list_labels_eval

#Turn tweets into encodings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

#Creating classes for training and evaluation datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
train_dataset = Dataset(train_encodings, train_labels)
eval_dataset = Dataset(eval_encodings, eval_labels)

#Model arguments
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=11
)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=3,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=10,
    fp16=True,  # Use mixed precision if using a GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
