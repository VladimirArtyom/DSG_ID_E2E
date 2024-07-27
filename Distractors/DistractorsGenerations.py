from pytorch_lightning import LightningModule, LightningDataModule
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW)

from typing import Dict, List
from pandas import DataFrame
from numpy.random import rand, choice

class DGModel(LightningModule):
    def __init__ (this, 
                  model: T5ForConditionalGeneration,
                  new_tokenizer_len: int,
                  optimizer: Optimizer,
                  optimizer_lr: float = 1e-4):
        super().__init__()
        this.model = model
        this.model.resize_token_embeddings(new_tokenizer_len)
        this.lr = optimizer_lr
        this.opt = optimizer

    def forward(this, input_ids, attention_mask, labels=None):
        output: Tensor = this.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
        return output.loss, output.logits

    def training_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(this, batch: Dict, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def exe_step(this, batch: Dict, batch_indx: int):
        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        labels: Tensor = batch["labels"]
        loss, output = this(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return loss

    def configure_optimizers(this):
        return this.opt(this.parameters(),
                        lr=this.lr)


class DGDataset(Dataset):
    def __init__(this,
                 data: DataFrame,
                 tokenizer: T5Tokenizer,
                 sep_token: str,
                 max_source_token_len: int = 512,
                 max_target_token_len: int = 512):
        this.tokenizer = tokenizer
        this.sep_token = sep_token
        this.max_source_token_len = max_source_token_len
        this.max_target_token_len = max_target_token_len
        this.data = data

    def __len__(this):
        return this.data.shape[0]

    def __getitem__(this, index: int):
        row = this.data.iloc[index]
        source_encoding = this.tokenizer(
            "{} {} {} {} {}".format(row["answer"],
                                    this.sep_token,
                                    row["question"],
                                    this.sep_token,
                                    row["context"]),
            max_length=this.max_source_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        target_encoding = this.tokenizer(
            "{} {} {} {} {}".format(row["incorrect_1"],
                                    this.sep_token,
                                    row['incorrect_2'],
                                    this.sep_token,
                                    row['incorrect_3']),
            max_length=this.max_target_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        labels = target_encoding['input_ids']
        labels[labels == 0] = -100
        return dict(
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten())


class DGDataModule(LightningDataModule):
    def __init__(this,
                 train_df: DataFrame,
                 val_df: DataFrame,
                 test_df: DataFrame,

                 tokenizer: T5Tokenizer,
                 sep_token: str = "<sep>",

                 batch_size: int = 16,
                 max_source_token_len: int = 512,
                 max_target_token_len: int = 512,
                 ):
        super().__init__()
        this.train_df: DataFrame = train_df
        this.val_df: DataFrame = val_df
        this.test_df: DataFrame = test_df

        this.tokenizer = tokenizer
        this.sep_token = sep_token
        this.batch_size = batch_size

        this.max_source_token_len = max_source_token_len
        this.max_target_token_len = max_target_token_len

    def setup(this, stage: str = None) -> None:
        this.train_dataset = DGDataset(this.train_df, this.tokenizer,
                                       this.sep_token,
                                       this.max_source_token_len,
                                       this.max_target_token_len)

        this.val_dataset = DGDataset(this.train_df, this.tokenizer,
                                     this.sep_token,
                                     this.max_source_token_len,
                                     this.max_target_token_len)

        this.test_dataset = DGDataset(this.train_df, this.tokenizer,
                                      this.sep_token,
                                      this.max_source_token_len,
                                      this.max_target_token_len)
        return

    def train_dataloader(this):
        return DataLoader(this.train_dataset, batch_size=this.batch_size,
                          shuffle=True, num_workers=-1)

    def val_dataloader(this):
        return DataLoader(this.val_dataset, batch_size=this.batch_size,
                          num_workers=-1)

    def test_dataloader(this):
        return DataLoader(this.test_dataset, batch_size=this.batch_size,
                          num_workers=-1)

