from QuestionGenerators.QuestionGenerations import QGModel, QGDataModule
from pandas import DataFrame
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from pytorch_lightning import Trainer
from typing import List


class Driver():
    def __init__(this,
                 sep_token: str,
                 batch_size: int,
                 source_max_token_len: int,
                 target_max_token_len: int,
                 masking_chance: float):
        this.batch_size = batch_size
        this.source_max_token_len = source_max_token_len
        this.target_max_token_len = target_max_token_len
        this.masking_chance = masking_chance
        this.sep_token = sep_token

    def prepare_question_generators_data(this,
                                         tokenizer: T5Tokenizer,
                                         train_df: DataFrame,
                                         val_df: DataFrame,
                                         test_df: DataFrame):

        this.qgDataModule = QGDataModule(train_df, val_df, test_df, tokenizer,
                                         this.sep_token,
                                         this.masking_chance, this.batch_size,
                                         this.source_max_token_len,
                                         this.target_max_token_len)

    def prepare_question_generators_model(this,
                                          model: T5ForConditionalGeneration,
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
        return this.trainer.fit(this.qgModel, this.qgDataModule)

    def test_question_generator(this):
        if this.trainer is None:
            raise ValueError("Trainer not initialized")

        return this.trainer.test(this.qgModel, this.qgDataModule)

    def val_question_generator(this):
        if this.trainer is None:
            raise ValueError("Trainer not initialized")

        return this.trainer.validate(this.qgModel, this.qgDataModule)

    def load_qg_model(this, model_path: str):
        if this.qgModel is None:
            raise ValueError("QGModel not initialized")

        this.qgmodel = this.qgModel.load_from_checkpoint(model_path)
        return

    def run_qg(this,
               train_df: DataFrame,
               val_df: DataFrame,
               test_df: DataFrame,
               tokenizer: T5Tokenizer,

               model: T5ForConditionalGeneration,
               new_tokenizer_len: int,
               optimizer: AdamW,
               optimizer_lr: float,

               callbacks,
               epochs: int = 3,
               accelerator: str = "gpu"

               ):
        result: List = []
        this.prepare_question_generators_data(tokenizer, train_df,
                                              val_df, test_df)

        this.prepare_question_generators_model(model, new_tokenizer_len,
                                               optimizer, optimizer_lr)

        result.append(this.train_question_generator(callbacks,
                                                    epochs,
                                                    accelerator))
        result.append(this.val_question_generator())
        result.append(this.test_question_generator())
        return result

    def test_qg(this,
                model_path: str,
                train_df: DataFrame,
                val_df: DataFrame,
                test_df: DataFrame,
                tokenizer: T5Tokenizer,

                model: T5ForConditionalGeneration,
                new_tokenizer_len: int,
                optimizer: AdamW,
                optimizer_lr: float):
        this.prepare_question_generators_data(tokenizer, train_df,
                                              val_df, test_df)

        this.prepare_question_generators_model(model, new_tokenizer_len,
                                               optimizer, optimizer_lr)
        this.load_qg_model(model_path)

        print("Model loaded successfully")

    def generate(this, answer: str, context: str,
                 tokenizer: T5Tokenizer,
                 beams: int = 1, rep_penalty: float = 1.0,
                 length_penalty: float = 1.0) -> str:

        source_encoding = tokenizer(
            "{} {} {}".format(answer, this.sep_token, context),
            max_length=this.source_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        generated_ids=this.qgModel.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            num_beams=beams,
            max_length=this.target_max_token_len,
            repetition_penalty=rep_penalty,
            length_penalty=length_penalty,
            early_stopping=True,
            use_cache=True
        )

        preds = {
            tokenizer.decode(generated_id, skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        }

        return ''.join(preds)

    def show_results(this, generated: str, answer: str,
                     context: str, original_question: str = ''):

        print('Original Question: {}'.format(original_question))
        print()
        print('Answer: {}'.format(answer))
        print('Context: {}'.format(context))
        print('Generated Question: {}'.format(generated))
        print("--------------------------------------------")
        print()


    def try_generate(this, tokenizer,
                     test_df: DataFrame,
                     n: int = 10, save: bool = False):
        if this.qgModel is None:
            raise ValueError("QGModel not initialized")

        for i in range(n):
            row = test_df.iloc[i]

            answer = row["answer"]
            context = row["context"]
            original_question = row["question"]
            generated = this.generate(answer, context,
                                      this.qgModel, original_question)
            print("Generated: ", generated)

            if not save:
                this.show_results(answer,
                                  context,
                                  generated)

