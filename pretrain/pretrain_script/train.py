import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
)

logger = logging.getLogger(__name__)


# 返回一个列表，包含指定目录下的所有文件路径
def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []

    for root, dirs, files in os.walk(dir_name):  # 当前目录、子目录列表、文件列表
        for temp_file in files:
            standard_path = f"{root}/{temp_file}"
            all_file_list.append(standard_path)

    return all_file_list


# 返回dataset对象
def load_dataset_from_path(
    data_path: Optional[str] = None,
    cache_dir: Optional[str] = "cache_data",
    data_file_number: Optional[int] = None,  # 使用前data_file_number个文件进行训练
    use_streaming: bool = False,
) -> Dataset:
    if data_file_number is None:
        all_file_list = get_all_datapath(data_path)
    else:
        all_file_list = get_all_datapath(data_path)[:data_file_number]
    data_files = {"train": all_file_list}
    extension = all_file_list[0].split(".")[-1]  # 获取文件后缀名,txt文件使用text格式

    logger.info("load %d files ", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
        streaming=use_streaming,
    )["train"]

    return raw_datasets


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
                `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    """

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_num_limit: int = field(
        default=None, metadata={"help": "the numbers of data file"}
    )
    data_proc_num: int = field(
        default=None, metadata={"help": "the numbers of process"}
    )
    use_streaming: bool = field(
        default=False, metadata={"help": "use stream mode to process big data"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


# 对数据集进行预处理(tokenize等)，返回dataset对象
def make_train_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    data_file_number: int,
    data_proc_num: int,
    use_streaming: bool,
) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(  # 纯文本数据集
        data_path=data_path,
        data_file_number=data_file_number,
        use_streaming=use_streaming,
    )

    logging.warning("Formatting inputs...")

    def generate_sources_targets(
        examples: Dict,
        tokenizer: transformers.PreTrainedTokenizer,  # example 针对单条数据字典
    ):
        ins_data = examples["content"]

        input_output = tokenizer(
            ins_data,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length - 1,
            truncation=True,
        )
        examples["input_ids"] = input_output["input_ids"]
        return examples

    # 将tokenizer作为generate_sources_targets的固定参数
    generate_sources_targets_p = partial(generate_sources_targets, tokenizer=tokenizer)

    if use_streaming:
        dataset = dataset.map(
            function=generate_sources_targets_p, batched=True
        ).shuffle(42, buffer_size=50000)
    else:
        dataset = dataset.map(
            function=generate_sources_targets_p,
            batched=True,
            desc="Running tokenizer on train dataset",
            num_proc=data_proc_num,
        ).shuffle()

    return dataset


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    torch.cuda.empty_cache()

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = make_train_dataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_file_number=data_args.data_num_limit,
        data_proc_num=data_args.data_proc_num,
        use_streaming=data_args.use_streaming,
    )
    train_dataset = train_dataset.remove_columns(
        [
            "uniqueKey",
            "title",
            "titleUkey",
            "dataType",
            "id",
            "content",
        ]  # 移除不需要的列
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )  # mlm：是否使用mask语言模型 pad_to_multiple_of：填充到8的倍数

    # 正数表示训练步数，覆盖epochs
    # 如果流式处理数据，按照step训练
    # 如果不是流式处理数据，按照epochs训练
    if not data_args.use_streaming:
        training_args.max_steps = -1

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
