## 介绍

```python
PreTrainedTokenizer.apply_chat_template(messages, add_generation_prompt=True)
```

为对话添加模板作为模型的输入

## add_generation_prompt

添加模型开始答复的标记

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

add_generation_prompt=**False**

```python
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

add_generation_prompt=**True**

```python
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant # 添加模型开始回复的模板
"""
```

并非所有模型都需要生成提示。一些模型，如 BlenderBot 和 LLaMA，在模型回复之前没有任何特殊标记。 在这些情况下，`add_generation_prompt`参数将不起作用。`add_generation_prompt`参数取决于你所使用的模板。

## 训练中使用聊天模板

将聊天模板应用为数据集的预处理步骤

在训练时，通常应该设置`add_generation_prompt=False`，因为添加的助手标记在训练过程中并不会有帮助。



更多内容可参考：https://zhuanlan.zhihu.com/p/680744631