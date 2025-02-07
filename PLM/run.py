import json
import numpy as np

import datasets
from datasets.dataset_dict import DatasetDict

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np

metric = evaluate.load("accuracy")
seed = 98


def read_data(path, to_dataset=True):
    '''
    read csv and return list of instances
    '''
    rows = []; label_dict = {'SUPPORTED': 0, 'REFUTED': 1, 'NEI': 2}
    # read data
    with open(path, 'r') as f:
        j_list = list(f)

    for j_str in j_list:
        row_dict = json.loads(j_str)
        
        rows.append({
            'id': row_dict['id'],
            'text': row_dict['gold_evidence_text'].strip() + '[SEP]' + row_dict['claim'].strip(),
            'label': label_dict[row_dict['label']],
            'claim': row_dict['claim'].strip()
        })
    if to_dataset:
        # to object
        rows = datasets.Dataset.from_list(rows)

    return rows


def train_test_split(ds, test_split=0.20):
    '''
    split ds to train and test
    '''
    # 70 train, 30 test + validation
    train_test = ds.train_test_split(shuffle=True, seed=seed, test_size=test_split)

    ds = DatasetDict(
        {
            'train': train_test['train'], 
            'test': train_test['test'], 
        })
    return ds


def compute_metrics(eval_pred):
    '''
    calculate accuracy
    '''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


class BERT:
    '''
    with tokenizer and auto model
    '''
    def __init__(self, MODEL, num_labels):
        '''
        basic
        '''
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
    
    def tokenize(self, samples):
        '''
        run tokenization
        '''
        return self.tokenizer(samples["text"], padding="max_length", truncation=True)
    
    def run(self, sample):
        '''
        run for a single instance
        '''
        encoded_input = self.tokenizer(sample, return_tensors='pt')
        output = self.model(**encoded_input)['logits'][0].detach().numpy()

        return np.argmax(output)


if __name__ == "__main__":
    MODEL = "google-bert/bert-base-uncased"
    # init model
    model_module = BERT(MODEL, num_labels=3)
    # read dataset
    dataset = read_data('./fever_train.jsonl')
    dataset = train_test_split(dataset)
    # tokenize dataset
    tok_dataset = dataset.map(model_module.tokenize, batched=True)
    '''
    train_ds = tok_dataset["train"].shuffle(seed=seed).select(range(1000))
    eval_ds = tok_dataset["test"].shuffle(seed=seed).select(range(1000))
    '''
    # training
    training_args = TrainingArguments(output_dir="./logs", num_train_epochs=3, learning_rate=3e-05, eval_strategy="epoch")

    trainer = Trainer(
        model=model_module.model,
        args=training_args,
        train_dataset=tok_dataset["train"],
        eval_dataset=tok_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
