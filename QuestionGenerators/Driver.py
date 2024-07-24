from QuestionGenerators.QuestionGenerations import QGModel, QGDataModule
from pandas import DataFrame
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from pytorch_lightning import Trainer
from typing import List


class Driver():
    def __init__(this,
                 batch_size: int,
                 source_max_token_len: int,
                 target_max_token_len: int,
                 masking_chance: float):
        this.batch_size = batch_size
        this.source_max_token_len = source_max_token_len
        this.target_max_token_len = target_max_token_len
        this.masking_chance = masking_chance

    def prepare_question_generators_data(this,
                                         tokenizer: T5Tokenizer,
                                         train_df: DataFrame,
                                         val_df: DataFrame,
                                         test_df: DataFrame):
        this.qgDataModule = QGDataModule(train_df, val_df, test_df, tokenizer,
                                         this.masking_chance, this.batch_size,
                                         this.source_max_token_len,
                                         this.target_max_token_len)

    def prepare_question_generators_model(this, model: T5ForConditionalGeneration,
                                          new_tokenizer_len: int,
                                          optimizer: AdamW,
                                          optimizer_lr: float = 1e-4):
        this.qgModel = QGModel(model, new_tokenizer_len,
                               optimizer, optimizer_lr)
        return

    def train_question_generator(this,
                                 callbacks,
                                 epochs: int,
                                 accelerator: str = "gpu"):
        if this.qgModel is None or this.qgDataModule is None:
            raise ValueError("QGModel or QGDataModule not initialized")

        this.trainer = Trainer(
            callbacks=callbacks,
            max_epochs=epochs,
            accelerator=accelerator,
        )
        this.trainer.fit(this.qgModel, this.qgDataModule)
        return

    def test_question_generator(this):
        if this.trainer is None:
            raise ValueError("Trainer not initialized")

        this.trainer.test(this.qgModel)

    def val_question_generator(this):
        if this.trainer is None:
            raise ValueError("Trainer not initialized")

        this.trainer.validate(this.qgModel)

    def load_qg_model(this, model_path: str):
        if this.qgModel is None:
            raise ValueError("QGModel not initialized")

        this.qgmodel = this.qgModel.load_from_checkpoint(model_path)
        return


    def run_qg(this):
        result: List = []
        this.prepare_question_generators()
        this.prepare_question_generators_model()
        result.append(this.train_question_generator())
        result.append(this.val_question_generator())
        result.append(this.test_question_generator())
        return result

    def test_qg(this):
        this.prepare_question_generators()
        this.prepare_question_generators_model()
        this.load_qg_model()

