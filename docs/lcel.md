# LangChain Expression Language (LangChain表达式语言)

LangChain 表达式语言（LCEL）是一种轻松地将链组合在一起的声明性方式。 LCEL 从第一天起就被设计为**支持将原型投入生产，无需更改代码
**，从最简单的“提示词 + LLM”链到最复杂的链（我们已经看到人们在生产中成功运行了 100 个步骤的 LCEL 链）。下面是使用 LCEL
的一些原因：

- 支持流式传输（Streaming support）：使用 LCEL 构建链时，您可以获得最短的首次代币时间（直到第一个输出块出现之前经过的时间）。对于某些
  chain 来说，直接将 token 从 LLM 流式传输到流式输出解析器，可以以 LLM 提供者输出原始 toekn 相同的速率输出解析的结果。
- 支持异步（Async support）：LCEL 构建的任何链都可以使用同步 API（例如，在 Jupyter 中进行原型设计时）和异步 API（例如，在
  LangServe 服务器中）进行调用。这使得能够在原型设计和生产中使用相同的代码，并且具有出色的性能，能够在同一服务器中处理许多并发请求。
- 优化过并行执行（Optimized parallel execution）：只要你的 LCEL Chain
  具有可以并行执行的步骤（例如，从多个检索器获取文档），我们就会在同步和异步接口中自动执行此操作，以尽可能减少延迟。
- 重试和回退（Retries and fallbacks）：为 LCEL Chain 的任何部分配置重试和回退。这是让你的 Chain 在规模化的时候更加可靠的好方法。
- 输入和输出模式（Input and output modes）：输入和输出模式为每个 LCEL Chain 提供从链结构推断出的 Pydantic 和 JSONSchema
  模式。这可用于验证输入和输出，并且是 LangServe 的组成部分。
- 无缝集成 LangSmith（Seamless LangSmith tracing integration）：随着 Chain 变得越来越复杂，了解每一步到底发生了什么变得越来越重要。借助
  LCEL，所有步骤都会自动记录到 LangSmith，以实现最大程度的可观察性和可调试性。
- 无缝集成LangServe（Seamless LangServe deployment integration）：使用 LCEL 创建的任何 Chain 都可以使用 LangServe 轻松部署。

LCEL 可以轻松地从基本组件构建复杂的链，并支持开箱即用的功能，例如流式传输、并行性和日志记录。

### 基础案例: 提示词+LLM+输出解析器

最基本和常见的用例是将提示模板和模型链接在一起。为了看看这是如何工作的，让我们创建一个接受主题并生成笑话的链：

```python

import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


def test_basic_demo():
    prompt = ChatPromptTemplate.from_template("给我将一个关于{topic}的笑话")
    model = ChatOpenAI()
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    res = chain.invoke({"topic": "冰淇淋"})
    print(f"type(res): {type(res)}, res: {res}")


def main():
    load_dotenv()
    test_basic_demo()


if __name__ == '__main__':
    main()
```

![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-FCb96u.png)

请注意这行代码，我们使用 LCEL 将不同的组件拼凑成一个链：

```python
chain = prompt | model | output_parser
```

`|` 符号类似于 unix 管道运算符，它将不同的组件链接在一起，将一个组件的输出作为下一个组件的输入。

在此链中，用户输入传递到提示词模板，然后提示模板词输出传递到模型，然后模型输出传递到输出解析器。让我们分别看一下每个组件，以真正了解发生了什么。

### 提示词（Prompt）

`prompt` 是一个 `BasePromptTemplate`，他接受一个关于模板变量的字典，然后返回`PromptValue`。

PromptValue 是一个完整提示的包装器，可以传递给 LLM （它接受一个字符串作为输入）或 ChatModel （它接受一个序列作为输入的消息）。

它可以与任何一种语言模型类型一起使用，因为它定义了生成 BaseMessage 和生成字符串的逻辑。

```python
from langchain_core.prompts import ChatPromptTemplate, BasePromptTemplate


def test_prompt():
    prompt = ChatPromptTemplate.from_template("给我将一个关于{topic}的笑话")
    print(f"{type(prompt)=}, {prompt=}\n")
    print(f"{isinstance(prompt, BasePromptTemplate)=}\n")
    prompt_value = prompt.invoke({"topic": "冰淇淋"})
    print(f"{type(prompt_value)=}, {prompt_value=}\n")
    msgs = prompt_value.to_messages()
    print(f"{type(msgs)=}, {msgs=}\n")
    s = prompt_value.to_string()
    print(f"{type(s)=}, {s=}")


def main():
    test_prompt()


if __name__ == '__main__':
    main()

```

![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-qZMZP2.png)

### 模型（Model)

然后 `PromptValue` 被传递给 `model` 。

在本例中，我们的 `model` 是 `ChatModel` ，这意味着它将输出 `BaseMessage` 。

如果我们的 `model` 是 `LLM` ，它将输出一个字符串。

```python
import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.openai import OpenAI


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


def test_chat_model():
    print(f"{'*' * 20} test_chat_model {'*' * 20}")
    prompt = ChatPromptTemplate.from_template("给我将一个关于{topic}的笑话")
    prompt_value = prompt.invoke({"topic": "冰淇淋"})
    print(f"{type(prompt_value)=}, {prompt_value=}\n")

    model = ChatOpenAI()

    message = model.invoke(prompt_value)
    print(f"{type(message)=}, message: {message}")


def test_llm_model():
    print(f"{'*' * 20} test_llm_model {'*' * 20}")
    prompt = ChatPromptTemplate.from_template("给我将一个关于{topic}的笑话")
    prompt_value = prompt.invoke({"topic": "冰淇淋"})
    print(f"{type(prompt_value)=}, {prompt_value=}\n")

    model = OpenAI(model="gpt-3.5-turbo-instruct")

    message = model.invoke(prompt_value)
    print(f"{type(message)=}, message: {message}")


def main():
    load_dotenv()
    test_chat_model()
    test_llm_model()


if __name__ == '__main__':
    main()

```

### 输出解析器（Output parser）

最后，我们将 `model` 输出传递给 `output_parser` ，这是一个 `BaseOutputParser` ，意味着它接受字符串或 `BaseMessage` 作为输入。

`StrOutputParser` 特别简单地将任何输入转换为字符串。

### 管道（Pipeline)

## RAG搜索示例

