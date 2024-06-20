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

import logging
import os
import random

import transformers
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class RawPretrainDataset(Dataset):
    """Raw Dataset for llm pretraining."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(RawPretrainDataset, self).__init__()
        logging.warning("Loading data...")

        self.files_ls = []
        for file in os.listdir(data_path):
            if file.endswith(".md") or file.endswith(".txt"):
                # 遍历所有文件，如果是.md或.txt文件，则拼接路径，读取后存储在files_ls中
                pretrain_data_path = data_path + '/' + file
                self.files_ls.append(self.read_md(pretrain_data_path, tokenizer))

        logging.warning("Loading data finished.")

        # shuffle files 重新排列
        # 从0-len(self.files_ls)-1中随机取样，取样个数为len(self.files_ls)，并按照取样结果重新排列self.files_ls
        random_index = random.sample(range(len(self.files_ls)), len(self.files_ls))
        self.files_ls = [self.files_ls[i] for i in random_index]

    def read_md(self, pretrain_data_path, tokenizer):
        logging.warning('Loading data from ' + pretrain_data_path)
        with open(pretrain_data_path, "r") as f:
            raw_file = f.read()
        # 添加结束符，用于区分不同文件
        return f"{raw_file}{tokenizer.eos_token}"


# main
if __name__ == "__main__":
    raw_dataset = RawPretrainDataset(data_path='data', tokenizer=transformers.GPT2Tokenizer.from_pretrained('gpt2'))
