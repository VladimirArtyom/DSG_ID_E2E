from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch import device
from torch.cuda import is_available as cuda_available
from typing import List
from pandas import DataFrame, read_csv
from QuestionGenerators.Driver import Driver as QGDriver
from Distractors import Driver as DGDriver
from transformers import T5TokenizerFast as T5Tokenizer, T5ForConditionalGeneration, AdamW

qg_config = OmegaConf.load("./QuestionGenerators/qg.yaml")
dg_config = OmegaConf.load("./Distractors/dg.yaml")
main_config = OmegaConf.load("./main.yaml")
device: str = device("cuda" if cuda_available() else "cpu")

allowed_run: List[str] = ["none", "all", "qg", "dg"]
allowed_types: List[str] = ["test", "train"]

arg = ArgumentParser()
# QG Arguments
arg.add_argument("-qglr", "--qg_learning_rate",
                 type=float, default=qg_config.qg_lr)

arg.add_argument("-qgb", "--qg_batch_size",
                 type=int, default=qg_config.qg_batch_size)

arg.add_argument("-qgmc", "--qg_masking_chance",
                 type=float, default=qg_config.qg_masking_chance)

arg.add_argument("-qgsmax", "--qg_source_max_token_len",
                 type=int, default=qg_config.qg_src_max)

arg.add_argument("-qgtmax", "--qg_target_max_token_len",
                 type=int, default=qg_config.qg_tgt_max)

arg.add_argument("-qge", "--qg_epochs",
                 type=int, default=qg_config.qg_epochs)

arg.add_argument("-qgmp", "--qg_model_path",
                 type=str, default=qg_config.qg_model_path)

# Main Arguments
arg.add_argument("-run", type=str,
                 default=main_config.run,
                 choices=allowed_run,
                 )

arg.add_argument("-t", "--type_run",
                 type=str,
                 default=main_config.type_run,
                 choices=allowed_types,
                 )

arg.add_argument("-sep", "--separator",
                 type=str,
                 default=main_config.sep_token)

# DG Arguments
arg.add_argument("-dglr", "--dg_learning_rate",
                 type=float, default=dg_config.dg_lr)

arg.add_argument("-dgb", "--dg_batch_size",
                 type=int, default=dg_config.dg_batch_size)

arg.add_argument("-dgmc", "--dg_masking_chance",
                 type=float, default=dg_config.dg_masking_chance)

arg.add_argument("-dgsmax", "--dg_source_max_token_len",
                 type=int, default=dg_config.dg_src_max)

arg.add_argument("-dgtmax", "--dg_target_max_token_len",
                 type=int, default=dg_config.dg_tgt_max)

arg.add_argument("-dge", "--dg_epochs",
                 type=int, default=dg_config.dg_epochs)

arg.add_argument("-dgmodelpth" "--dg_model_path",
                 type=str, default=dg_config.dg_model_path)


args = arg.parse_args()

print(args)

train_df: DataFrame = read_csv(main_config.train_path)
val_df: DataFrame = read_csv(main_config.val_path)
test_df: DataFrame = read_csv(main_config.test_path)
tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(main_config.model_name)
#model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(main_config.model_name)
new_tokenizer_token = tokenizer.add_tokens(main_config.sep_token)

model = T5ForConditionalGeneration.from_pretrained(main_config.model_name)
new_tokenizer_len = len(tokenizer)
optimizer = AdamW(model.parameters(), lr=args.qg_learning_rate)
optimizer_lr = args.qg_learning_rate

if args.type_run == "test":
    if args.run == "all":
        QGdriver = QGDriver(args.separator,
                            args.qg_batch_size,
                            args.qg_source_max_token_len,
                            args.qg_target_max_token_len,
                            args.qg_masking_chance)

        QGdriver.test_qg(args.qg_model_path, train_df,
                         val_df, test_df, tokenizer,
                         model, new_tokenizer_len,
                         optimizer, optimizer_lr
                         )

    elif args.run == "none":
        ...
    elif args.run == "qg":
        ...
    elif args.run == "dg":
        ...
elif args.type_run == "train":
    if args.run == "all":
        ...
    elif args.run == "qg":
        ...
    elif args.run == "dg":
        ...


