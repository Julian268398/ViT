import os
import torch
import numpy as np
import datasets

from sklearn.metrics import precision_score, recall_score, accuracy_score
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer

def set_requires_grad_for_layers(model, freeze_until_layer=None):
    """
    Funkcja ustawiania flagi `requires_grad` dla parametrów modelu, aby kontrolować, które warstwy są trenowane.

    :param model: nn.Module
        Model PyTorch, dla którego ustawiana jest flaga `requires_grad`.
    :param freeze_until_layer: str lub None
        Nazwa warstwy, do której wszystkie poprzednie warstwy będą zamrożone (nie będą trenowane).
        Jeśli None, wszystkie warstwy pozostaną odblokowane.
    """
    freeze = True
    for name, param in model.named_parameters():
        if freeze_until_layer and freeze and freeze_until_layer in name:
            freeze = False
        param.requires_grad = not freeze
    print(f"Zmieniono `requires_grad` w modelu. Głębsze warstwy od '{freeze_until_layer}' są odblokowane.")

def model_processor(model_name):
    """
    Funkcja doboru procesora

    :param model_name: string
        Nazwa modelu wstępnie wytrenowego
    :return: procesor
    """
    processor = ViTImageProcessor.from_pretrained(model_name)
    return processor



def transform(data):
    """
    Funkcja przekształcania danych wejściowych w odpowiedni dla modelu format
    :param data: dict
        Słownik zawierający liste obrazów i liste etykiet
    :return: dict
        Słownik zawierający przetowrzone obrazy jako tensory PyTorch i etykiety przypisane do obrazów
    """
    inputs = processor([x for x in data['image']], return_tensors='pt')

    inputs['labels'] = data['label']
    return inputs

def collate_fn(batch):
    """
    Funkcja przekształcania batchy
    Grupowanie wszystkich tensorów reprezentujących przetoworzony obraz w jeden tenseor wielo wymiarowy
    Grupowannie wszystkich etykiet w jeden tensor jednowymiarowy

    :param batch: list
        lista słowników, gdzie każdy słownik odpowiada jednej próbce
    :return: dict
        Słownik zawierający:
        - 'pixel_values': tensor o wymiarach [batch_size, ...] (batch przetworzonych obrazów).
        - 'labels': tensor o wymiarach [batch_size] (batch etykiet odpowiadających obrazom).
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def Model(model_name, num_labels, id2label, label2id, WithInfo):
    """
    Funkcja wywołująca model z Hugging Face

    :param model_name: string
        Nazwa modelu w bibliotece Hugging Face
    :param num_labels: int
        liczba klas w zadaniu klasyfikacji
    :param id2label: dict
        Słownik mapujący identyfikatory etykiet na ich nazwy
    :param label2id: dict
        Słownik odwrotny do `id2label`, mapujący nazwy etykiet na ich identyfikatory
    :param WithInfo: bool
        Flaga określająca, czy model ma być inicjalizowany z informacjami o etykietach

    :return: ViTForImageClassification
        Instancja modelu Vision Transformer z Hugging Face przygotowanego do klasyfikacji obrazów
    """
    if WithInfo:
        return ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id)
    else:
        return ViTForImageClassification.from_pretrained(model_name)


def training_args(output_dir,num_train_epochs):
    """
    Funkcja ustawiania konfiguracji treningowej

    :param output_dir: string
        Ścieżka do katalogu, w którym będą zapisywane wyniki treningu,
        w tym zapisane modele, logi i inne dane.
    :param num_train_epochs:int
        Liczba epok treningowych

    :return: TrainingArguments
        Obiekt konfiguracji treningowej
    """

    return TrainingArguments(
  output_dir=output_dir,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs= num_train_epochs,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
 # report_to='tensorboard',
  load_best_model_at_end=True)

def Training_func(model,training_args,collate_fn,compute_metrics,train_dataset,eval_dataset,tokenizer):
    """
    Funkcja inicjalizacji obiektu trenera

    :param model: PreTrainedModel
        Model do trenowania, np. załadowany z Hugging Face Transformers
    :param training_args: TrainingArguments
        Obiekt konfiguracji treningowej, definiujący hiperparametry i strategie treningowe.
    :param collate_fn: callable
        Funkcja przekształcająca batch danych (data collator) na odpowiedni format do modelu.
    :param compute_metrics: callable
        Funkcja obliczająca metryki ewaluacji. Przyjmuje wynik modelu i zwraca słownik z metrykami.
    :param train_dataset: Dataset
        Zbiór danych treningowych w formacie kompatybilnym z `Trainer`.
    :param eval_dataset: Dataset
        Zbiór danych walidacyjnych w formacie kompatybilnym z `Trainer`.
    :param tokenizer: PreTrainedTokenizer
        Tokenizer używany do przetwarzania tekstu (jeśli wymagany przez model).

    :return: Trainer
        Instancja klasy `Trainer` z ustawionym modelem, argumentami, danymi i metrykami.
    """
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
    """
    Funkcja obliczania metryk ewaluacji na podstawie wyników modelu.

    :param p: EvalPrediction
        Obiekt zawierający przewidywania modelu i prawdziwe etykiety
    :return: dict
        Słownik z metrykami: dokładność, precyzja, recall.
    """
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
    """
    Ewaluacja modelu na zbiorze testowym.

    :param trainer: Trainer
        Obiekt `Trainer` odpowiedzialny za ewaluację.
    :param test_dataset: Dataset
        Zbiór danych testowych.
    :return: dict
        Słownik z wynikami metryk na zbiorze testowym.
    """
    print("Evaluating on test dataset...")
    metrics = trainer.evaluate(test_dataset)
    print("Test Metrics:")
    print(metrics)
    return metrics


processor = model_processor('google/vit-base-patch16-224-in21k')  # Inicjalizuje procesor obrazu dla modelu Vision Transformer

train_dir = 'ścieżka do folderu zawierającego dane treningowe'
test_dir = 'ścieżka do folderu zawierającego dane testowe'
valid_dir = 'ścieżka do folderu zawierającego dane walidacyjne'

# Ładowanie zbiorów danych
train_dataset = datasets.load_dataset("imagefolder",
                             data_dir=train_dir)
valid_dataset = datasets.load_dataset("imagefolder",
                             data_dir=valid_dir)
test_dataset = datasets.load_dataset("imagefolder",
                             data_dir=test_dir)

# Przekształcanie zbiorów do odpowiedniego formatu dla modelu
train_ds = train_dataset.with_transform(transform)
valid_ds = valid_dataset.with_transform(transform)
test_ds = test_dataset.with_transform(transform)

Data_dir = train_dir
labels = os.listdir(Data_dir)  # Pobieranie listy etykiet z katalogu danych
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}

# Inicjalizacja modelu Vision Transformer z odpowiednią konfiguracją
model = Model('google/vit-base-patch16-224-in21k', len(labels), id2label, label2id, True)

set_requires_grad_for_layers(model, freeze_until_layer="encoder.layer.8")  # Odblokowanie trenowania warstw głębszych

model_training_args = training_args("./vit-task", 2)  # Tworzenie katalogu wyników

# Inicjalizacja obiektu "Trainer" do trenowania modelu
training_cat = Training_func(model, model_training_args, collate_fn,
                             compute_metrics, train_ds['train'], valid_ds['train'], processor)


train_results = training_cat.train()  # Rozpoczyna trening modelu
training_cat.log_metrics("train", train_results.metrics)  # Loguje metryki treningowe
training_cat.save_metrics("train", train_results.metrics)  # Zapisuje metryki treningowe do pliku
training_cat.save_state()  # Zapisuje stan trenera (np. checkpoint modelu)

test_metrics = evaluate_on_test(training_cat, test_ds["train"])  # Przeprowadza ewaluację modelu na zbiorze testowym
training_cat.log_metrics("test", test_metrics)  # Loguje metryki testowe
training_cat.save_metrics("test", test_metrics)  # Zapisuje metryki testowe do pliku