看看李一舟讲得啥，销量如此高

#### AI思维

GPT（Generative Pre-trained Transformer）是变革性的技术，可以改变我们的工作流。唯一的问题是在于你能不能提出个好问题，去调用ta的能力。

#### 什么是算力什么是token

用ChatGPT我们是怎么花钱的。可以联想运营商的流量收费。

GPT能不能按照字符数收费呢？听起来合理其实也不合理。比如说在中文语境中间一句唐诗听起来很简洁，但将它翻译成英语会是很长的句子。又比如我们可以用非常简单的句子，讲一个非常复杂的逻辑关系。那这个对于AI算起来是挺复杂的。反过来说，很长的句子也可能只表达一个非常简单的意思。因此直接按照字符的多少来收费也不合理。

因此引入我们真正在AI上通用的计算方法——token

在NLP中，token是指一组相关的字符序列。

> 将文本分解为token是NLP的一项基本任务，因为它是许多其他任务的先决条件。因此，对于NLP系统来说，选择正确的<font color="brown">分词方法（tokenization）</font>非常重要，它将直接影响到其他任务的准确性和效率。

`GPT models consume unstructured text, which is represented to the model as a sequence of "tokens"`

如果想查询一串指定的文本到底需要耗费多少个token，OpenAI官方有提供一个[计算器（tokenizer）](https://platform.openai.com/tokenizer)。

> 感觉GPT-3.5和GPT-4的tokenizer比GPT-3强大，比如import pandas as pd，GPT-3.5会解析为3个token，而GPT-3是6个token。究其原因是GPT-3用的词汇表比较小，不认识pandas和pd。

参考[ChatGPT的计费方式：Token](https://www.aiyzh.com/chatgpt/88/)

#### 与GPT对话

**Token的计算** : 首先 OpenAI token 的计算包含两部分。输入给 GPT 模型的 token 数和 GPT 模型生成文本的 token 数。

>例如，你提问耗费了 100 token，GPT 根据你的输入，生成文本（也就是回答）了 200 token，那么一共消费的 token 数就是 300 。

**带上下文的Token的计算**：

在同一次会话中，GPT4U 将会通过您此次对话的上下文联想并回复。当然，上下文记忆将由您来控制，您可以在 ***设置页面 - 附带历史消息数\*** 设置 GPT4U 对于上下文的联想数量，如果上下文联想越多，您消耗的 Tokens 数将会增加。举个例子：

```markdown
对话1：您：1+1是不是等于3   （消耗 9 Tokens）
      GPT4U：3不是正确答案 （消耗 6 Tokens）
对话2：您：等于几？         （消耗 9+6+3=18 Tokens）
      GPT4U：2           （消耗 1 Tokens）
```

> 值得注意的是：
>
> 1. **GPT4U 目前并没有将用户的上下文联想损耗计入到Tokens 消耗内**，即Tokens 消耗仅计算用户发送的最后一句话以及GPT4U 最新回答内容的Tokens 数量。
> 2. 目前采用的是**单次对话先完成后结算**的计费方式，因此不必担心 GPT4U 在无法使用的时候会损耗您的 Tokens。

参考[What Is Tokens](https://afdian.net/p/87981998dc2611edaa3c52540025c377)

GPT模型有一个最大单词对话（Token Limits），表示了单次对话范围内支持的Token上限。

> GPT-3.5的最大单词对话（Token Limits）是4096个token，大概是2000字，也就是前面说1000个字，后面回1000个字就没有了。GPT-4.0升级到了32K，比之前容量大了8倍。大大扩展了交流的空间。

#### New Bing

