#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from generate_pretrain_data import RawPretrainDataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


# 模型参数
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


# 数据集参数
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


# 训练参数
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


# 添加新的special token，并取embedding的均值
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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",  # padding到最长的序列的长度
            truncation=False,
        )
        for text in strings
    ]

    raw_input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    # split input_ids with model_max_length
    input_ids = []
    labels = []
    max_length = tokenizer.model_max_length

    for instance_ids in raw_input_ids:
        for i in range(0, len(instance_ids), max_length):
            input_ids.append(instance_ids[i: i + max_length])
            labels.append(instance_ids[i: i + max_length])
            if len(instance_ids[i: i + max_length]) < max_length:
                logging.warning(
                    f"len(instance_ids[i : i + max_length]) < max_length: {len(instance_ids[i: i + max_length])} < {max_length}")

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def preprocess(
        examples: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""

    examples_tokenized = _tokenize_fn(examples, tokenizer)
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    return dict(input_ids=input_ids, labels=labels)


# dataset class
class PretrainDataset(Dataset):
    """Dataset for pretraining."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(PretrainDataset, self).__init__()
        logging.warning("Loading data...")
        # 加载数据 -> [str，str，...]
        raw_data = RawPretrainDataset(data_path=data_path, tokenizer=tokenizer)
        files_ls = raw_data.files_ls

        # Tokenize -> { 'input_ids': [[ids], [ids], ...], 'labels': [[ids], [ids], ...]}
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(files_ls, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForPretrainDataset(object):
    """Collate examples for pretraining."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # padding的部分不计算loss
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            # padding的部分作为attention mask
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# 准备dataset and collator
def make_pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for pretraining."""
    train_dataset = PretrainDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    # 1. 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2.load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        # 模型单个batch的最大长度
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 3.判断是否缺少special token，并更新embedding
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

    # 4.准备dataset and collator
    data_module = make_pretrain_data_module(tokenizer=tokenizer, data_args=data_args)

    # 5.训练
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()  # 保存整个训练状态
    trainer.save_model(output_dir=training_args.output_dir)  # 保存模型


if __name__ == "__main__":
    train()
