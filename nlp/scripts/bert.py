# Binary or Multiclass Text Classification
# Reference: https://github.com/abhishekkrthakur/bert-sentiment

import os
import tez
import torch
import logging
import numpy as np
from torch._C import ResolutionCallback
import transformers
import torch.nn as nn
from sklearn import metrics
from typing import Dict, Text
from transformers import AdamW, get_linear_schedule_with_warmup


class BERTDataset:
    def __init__(self, text: str, targets: float, max_len: int = 64):

        self.text = text
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=False
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item: int) -> Dict[Text, torch.tensor]:
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[item], dtype=torch.long),
        }


class BERTClassifier(tez.Model):
    def __init__(self, n_classes: int, train_steps: int, lr: float = 1e-4):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.20)
        self.out = nn.Linear(768, n_classes)
        self.train_steps = train_steps
        self.step_scheduler_after = "batch"
        self.lr = lr

    def fetch_optimizer(self):
        return AdamW(self.parameters(), lr=self.lr)

    def fetch_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.train_steps,
        )

    def loss_fn(self, outputs: torch.tensor, targets: torch.tensor):
        if targets is None:
            return None
        return nn.CrossEntropyLoss()(outputs, targets)

    def monitor_metrics(self, outputs: torch.tensor, targets: torch.tensor):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, axis=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        return {
            "acc": metrics.accuracy_score(targets, outputs),
            "f1_score": metrics.f1_score(targets, outputs),
        }

    def forward(self, inputs: Dict[Text, torch.tensor]):
        outputs = self.bert(
            inputs["ids"],
            attention_mask=inputs["mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        outputs = self.bert_drop(outputs)
        outputs = self.out(outputs)
        return outputs

    def forward(
        self,
        ids: torch.tensor,
        mask: torch.tensor,
        token_type_ids: torch.tensor,
        targets=None,
    ):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.bert_drop(x)
        output = self.out(x)

        if targets is not None:
            loss = self.loss_fn(output, targets)
            metrics = self.monitor_metrics(output, targets)
            return x, loss, metrics
        return x, None, {}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
):

    # create datasets
    train_dataset = BERTDataset(X_train, y_train)
    valid_dataset = BERTDataset(X_val, y_val)

    # callbacks
    os.makedirs("models", exist_ok=True)
    cb = tez.callbacks.EarlyStopping(
        monitor="valid_loss", patience=3, model_path=os.path.join("models", "model.bin")
    )

    # train model
    steps = int(len(X_train) / batch_size * epochs)
    model = BERTClassifier(n_classes=len(np.unique(y_train)), train_steps=steps)

    logging.info("Model Training....")
    history = model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        epochs=epochs,
        callbacks=[cb],
        train_bs=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # save modes
    model.save(os.path.join("models", "model.bin"))
    return history
