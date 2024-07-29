from Distractors.DistractorsGenerations import DGModel, DGDataModule
from torch import load, Tensor
from torch.optim import Optimizer
from pandas import DataFrame, concat
from transformers import T5TokenizerFast as T5Tokenizer, T5PreTrainedModel
from typing import List, Dict
from pytorch_lightning import Trainer


class Driver():
    def __init__(this,
                 sep_token: str,
                 batch_size: int,
                 max_source_token_len: int,
                 max_target_token_len: int):
        this.sep_token = sep_token
        this.batch_size = batch_size
        this.max_source_token_len = max_source_token_len
        this.max_target_token_len = max_target_token_len

    def prepare_distractor_generator_datasets(this,
                                              train_df: DataFrame,
                                              val_df: DataFrame,
                                              test_df: DataFrame,
                                              tokenizer: T5Tokenizer):
        this.dgDataModule = DGDataModule(train_df,
                                         val_df,
                                         test_df,
                                         tokenizer,
                                         this.sep_token,
                                         this.batch_size,
                                         this.max_source_token_len,
                                         this.max_target_token_len)
        return

    def prepare_distractor_generator_model(this,
                                           model: T5PreTrainedModel,
                                           new_tokenizer_len: int,
                                           optimizer: Optimizer,
                                           optimizer_lr: float
                                           ):
        this.dgModel = DGModel(model,
                               new_tokenizer_len,
                               optimizer,
                               optimizer_lr)
        return

    def check_is_prepared(this):
        if this.dgModel is None or this.dgDataModule is None:
            raise ValueError("DGModel or DGDataModule not initialized")

    def train_distractor_generator(this,
                                   callbacks: List,
                                   epochs: int,
                                   accelerator: str = "gpu"):
        this.check_is_prepared()
        this.trainer = Trainer(
            callbacks=callbacks,
            max_epochs=epochs,
            accelerator=accelerator)
        this.trainer.fit(this.dgModel,
                         this.dgDataModule)

    def val_distractor_generator(this) -> List[Dict]:
        this.check_is_prepared()
        res: List[Dict] = this.trainer.test(this.dgModel,
                                            this.dgDataModule)
        return res

    def test_distractor_generator(this):
        this.check_is_prepared()
        res: List[Dict] = this.trainer.validate(this.dgModel,
                                                this.dgDataModule)
        return res

    def load_dg_model(this,
                      model_path: str,
                      map_location: str = "cpu"):
        if this.dgModel is None:
            raise ValueError("DGModel not initialized")

        this.dgModel.load_state_dict(load(model_path,
                                          map_location=map_location)['state_dict'])
        return

    def run_dg(this,
               train_df: DataFrame,
               val_df: DataFrame,
               test_df: DataFrame,
               tokenizer: T5Tokenizer,

               model: T5PreTrainedModel,
               new_tokenizer_len: int,
               optimizer: Optimizer,
               optimizer_lr: float,

               callbacks,
               max_source_token_len: int,
               max_target_token_len: int,

               epochs: int = 3,
               accelerator: str = "gpu",
               ):
        results: List = []
        this.prepare_distractor_generator_model(model,
                                                new_tokenizer_len,
                                                optimizer,
                                                optimizer_lr,
                                                )
        this.prepare_distractor_generator_datasets(train_df,
                                                   val_df,
                                                   test_df,
                                                   tokenizer)

        results.append(this.train_distractor_generator(callbacks,
                                                       epochs,
                                                       accelerator))
        results.append(this.val_distractor_generator())
        results.append(this.test_distractor_generator())

        return results

    def test_dg(this,
                model_path: str,
                model: T5PreTrainedModel,
                train_df: DataFrame,
                val_df: DataFrame,
                test_df: DataFrame,
                tokenizer: T5Tokenizer,
                new_tokenizer_len: int,
                optimizer: Optimizer,
                optimizer_lr: float,
                map_location: str = "cpu",):
        this.prepare_distractor_generator_model(model, new_tokenizer_len,
                                                optimizer, optimizer_lr)
        this.prepare_distractor_generator_datasets(
            train_df,
            val_df,
            test_df,
            tokenizer)
        this.load_dg_model(model_path,
                           map_location)
        print("Distractor generation model loaded succesfully")

    def generate(this, answer: str,
                 question: str,
                 context: str,
                 incorrect_1: str,
                 incorrect_2: str,
                 incorrect_3: str,
                 tokenizer: T5Tokenizer,
                 num_beams: int = 2,

                 ):
        source_encoding = tokenizer(
            "{} {} {} {} {}".format(answer, this.sep_token,
                                    question, this.sep_token,
                                    context),
            max_length=this.max_source_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        outputs: Tensor = this.dgModel.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            max_length=this.max_target_token_len,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=2.5,
            length_penalty=1.0,
            use_cache=True
        )

        preds = [tokenizer.decode(output,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True) for output in outputs]

        return "".join(preds)

    def show_results(this, generated: str, answer: str,
                     question: str, context: str,
                     incorrect_1: str, incorrect_2: str,
                     incorrect_3: str
                     ):
        print("Context:\n")
        print(context)

        print("Question and answer:")
        print("Question {}\nAnswer: {}".format(question, answer))

        print("========================================Generated Results===========================================================")
        print("Original Distractors: Incorrect-1:{}\nIncorrect-2:{}\nIncorrect-3 {}".format(incorrect_1, incorrect_3, incorrect_3))
        print("Generated: {}".format(generated))
        print("====================================================================================================================")
        print()

    def save_generated_result(this, df: DataFrame,
                              tokenizer: T5Tokenizer,
                              file_name: str,
                              ):
        if this.dgModel is None:
            raise ValueError("DG Model is not initialized")

        result = DataFrame(columns=["correct",
                                    "context",
                                    "question",
                                    "incorrect_1",
                                    "incorrect_2",
                                    "incorrect_3",
                                    "generated"])
        for i in range(len(df)):
            row = df.iloc[i]
            answer = row["correct"]
            context = row["context"]
            question = row["question"]
            incorrect_1 = row["incorrect_1"]
            incorrect_2 = row["incorrect_2"]
            incorrect_3 = row["incorrect_3"]
            generated = this.generate(answer, question,
                                      context, incorrect_1,
                                      incorrect_2, incorrect_3, tokenizer)
            new_row = DataFrame({"correct": answer,
                                 "context": context,
                                 "question": question,
                                 "incorrect_1": incorrect_1,
                                 "incorrect_2": incorrect_2,
                                 "incorrect_3": incorrect_3,
                                 "generated": generated},
                                index=[0])

            result = concat([result, new_row], ignore_index=True)

        result.to_csv(file_name, index=False)
        print("Results saved to {}".format(file_name))

    def try_generate(this, tokenizer,
                     df: DataFrame, n: int):
        if this.dgModel is None:
            raise ValueError("DGModel is not initialized")

        for i in range(n):
            row = df.iloc[i]
            answer = row["correct"]
            context = row["context"]
            question = row["question"]
            incorrect_1 = row["incorrect_1"]
            incorrect_2 = row["incorrect_2"]
            incorrect_3 = row["incorrect_3"]

            generated = this.generate(answer, question, context,
                                      incorrect_1, incorrect_2,
                                      incorrect_3, tokenizer)

            this.show_results(generated, answer, question,
                              context, incorrect_1, incorrect_2,
                              incorrect_3)

