## 1.贪心策略

每次选择概率最大的token，可能会忽略稍好的token

![img](https://pic4.zhimg.com/80/v2-9c258a796b002c4a6858c4aa09253c87_720w.webp)

## 2.Beam Search

选择n个候选组合。

翻译和摘要这类可以大致预测生成长度的场景中表现还可以，可人的语言往往不是令语言模型概率最大化的那个词

![img](https://pic2.zhimg.com/80/v2-1288d68b4c58806947ceb93075292f51_720w.webp)

## 3.Top-k 采样

从排名前k的token种进行抽样

引入随机性的另一种方式是temperature。温度 𝑇 是一个范围从0到1的参数，它影响softmax函数生成的概率，使得最可能的token更有影响力。实际上，它只是将输入 logits 除以我们称为temperature的值：

![image-20240618124909959](assets/image-20240618124909959.png)

当temperature较高时，会更平均地分配概率给各个token，结合top_k,使生成的文本更具随机性和多样性；temperature较低接近0时，会倾向于选择概率最高的token，从而使生成的文本更加确定和集中。

## 4. top-p 采样

在累积概率超过 *p* 的最小单词集中进行，在这组词中重新分配概率质量

