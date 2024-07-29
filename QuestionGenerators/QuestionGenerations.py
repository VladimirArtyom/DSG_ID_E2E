from pytorch_lightning import LightningModule, LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from typing import Dict
from pandas import DataFrame
from numpy.random import rand, choice


class QGModel(LightningModule):
    def __init__(this, model: T5ForConditionalGeneration,
                 new_tokenizer_len: int,
                 optimizer: AdamW,
                 optimizer_lr: float = 1e-4):
        super().__init__()
        this.model: T5ForConditionalGeneration = model
        this.model.resize_token_embeddings(new_tokenizer_len)
        this.lr: float = optimizer_lr
        this.opt: AdamW = optimizer

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
        return this.optimizers(this.parameters(), lr=this.lr)


class QGDataset(Dataset):
    def __init__(this, data: DataFrame,
                 tokenizer: T5Tokenizer,
                 sep_token: str,
                 masking_chance: float = 0.6,
                 source_max_token_len: int = 512,
                 target_max_token_len: int = 80):
        super().__init__()
        this.data = data
        this.sep_token = sep_token
        this.tokenizer = tokenizer
        this.source_max_token_len = source_max_token_len
        this.target_max_token_len = target_max_token_len
        this.masks_token = [f"MASK_{i}" for i in range(5)]
        this.masking_chance = masking_chance

    def __len__(this):
        return this.data.shape[0]

    def __getitem__(this, index):
        data_row = this.data.iloc[index]

        if rand() < this.masking_chance:
            answer = data_row['answer']
        else:
            answer = choice(this.masks_token)

        source_encoding = this.tokenizer(
            '{} {} {}'.format(answer, this.sep_token, data_row['context']),
            max_length=this.source_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        target_encoding = this.tokenizer(
            '{} {} {}'.format(data_row['answer'], this.sep_token, data_row['question']),
            max_length=this.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            answer_text=data_row['answer'],
            context=data_row['context'],
            question=data_row['question'],
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )


class QGDataModule(LightningDataModule):
    def __init__(this, train_df: DataFrame, val_df: DataFrame,
                 test_df: DataFrame, tokenizer: T5Tokenizer,
                 sep_token: str = "<sep>",
                 masking_chance: float = 0.6,
                 batch_size: int = 16, source_max_token_len: int = 512,
                 target_max_token_len: int = 80):
        super().__init__()
        this.train_df: DataFrame = train_df
        this.val_df: DataFrame = val_df
        this.test_df: DataFrame = test_df

        this.masking_chance: float = masking_chance
        this.batch_size: int = batch_size
        this.source_max_token_len: int = source_max_token_len
        this.target_max_token_len: int = target_max_token_len
        this.tokenizer: T5Tokenizer = tokenizer

    def setup(this, stage: str = None):
        this.train_dataset = QGDataset(this.train_df, this.tokenizer,
                                       this.sep_token,
                                       this.masking_chance,
                                       this.source_max_token_len,
                                       this.target_max_token_len)

        this.val_dataset = QGDataset(this.val_df, this.tokenizer,
                                     this.sep_token,
                                     this.masking_chance,
                                     this.source_max_token_len,
                                     this.target_max_token_len)

        this.test_dataset = QGDataset(this.test_df, this.tokenizer,
                                      this.sep_token,
                                      this.masking_chance,
                                      this.source_max_token_len,
                                      this.target_max_token_len)

    def train_dataloader(this) -> DataLoader:
        return DataLoader(this.train_dataset, batch_size=this.batch_size,
                          shuffle=True, num_workers=-1)

    def val_dataloader(this) -> DataLoader:
        return DataLoader(this.val_dataset,
                          batch_size=this.batch_size,
                          num_workers=-1)

    def test_dataloader(this) -> DataLoader:
        return DataLoader(this.test_dataset,
                          batch_size=this.batch_size,
                          num_workers=-1)
