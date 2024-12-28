import os
import torch
import numpy as np
import datasets

from sklearn.metrics import precision_score, recall_score, accuracy_score
from transformers import SwinForImageClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoImageProcessor

def model_processor(model_name):
    processor = AutoImageProcessor.from_pretrained(model_name)
    return processor

def transform(data):
    inputs = processor([x for x in data['image']], return_tensors='pt')
    inputs['labels'] = data['label']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def Model(model_name, num_labels, id2label, label2id):
    return SwinForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

def training_args(output_dir, num_train_epochs):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True
    )

def Training_func(model, training_args, collate_fn, compute_metrics, train_dataset, eval_dataset, tokenizer):
    return Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    references = p.label_ids
    accuracy = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average="weighted")
    recall = recall_score(references, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

def evaluate_on_test(trainer, test_dataset):
    print("Evaluating on test dataset...")
    metrics = trainer.evaluate(test_dataset)
    print("Test Metrics:")
    print(metrics)
    return metrics

# Inicjalizacja przetwarzania obrazów
processor = model_processor('microsoft/swin-tiny-patch4-window7-224')

# Ścieżki do danych
train_dir = 'C:\\Users\\julia\\Desktop\\agu\\train'
test_dir = 'C:\\Users\\julia\\Desktop\\agu\\test'
valid_dir = 'C:\\Users\\julia\\Desktop\\agu\\valid'

# Ładowanie zbiorów danych
valid_dataset = datasets.load_dataset("imagefolder", data_dir=valid_dir)
train_dataset = datasets.load_dataset("imagefolder", data_dir=train_dir)
test_dataset = datasets.load_dataset("imagefolder", data_dir=test_dir)

# Przekształcanie zbiorów danych
train_ds = train_dataset.with_transform(transform)
valid_ds = valid_dataset.with_transform(transform)
test_ds = test_dataset.with_transform(transform)

# Mapowanie etykiet
Data_dir = train_dir
labels = os.listdir(Data_dir)
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}

# Inicjalizacja modelu Swin Transformer
model = Model('microsoft/swin-tiny-patch4-window7-224', len(labels), id2label, label2id)

# Argumenty treningowe
model_training_args = training_args("./swin-task", 2)

# Trener
training_cat = Training_func(model, model_training_args, collate_fn,
                             compute_metrics, train_ds['train'], valid_ds['train'], processor)

# Trening
train_results = training_cat.train()
training_cat.log_metrics("train", train_results.metrics)
training_cat.save_metrics("train", train_results.metrics)
training_cat.save_state()

# Ocena na zestawie testowym
test_metrics = evaluate_on_test(training_cat, test_ds["train"])
training_cat.log_metrics("test", test_metrics)
training_cat.save_metrics("test", test_metrics)
