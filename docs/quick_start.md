# 快速开始

在本快速入门中，我们将向您展示如何：

- 使用 LangChain、LangSmith 和 LangServe 进行设置
- 使用LangChain最基本、最常用的组件：提示模板（prompt templates)、模型（models）和输出解析器（output parsers)
- 使用 LangChain 表达式语言，这是 LangChain 构建的协议，有助于组件链接
- 使用LangChain构建一个简单的应用程序
- 使用 LangSmith 追踪您的应用程序
- 使用 LangServe 为您的应用程序提供服务

## 设置

### 安装

```bash
pip install langchain
```

有关更多详细信息，请参阅我们的[安装指南](https://python.langchain.com/docs/get_started/installation)。

### 环境变量

使用 LangChain 通常需要与一个或多个模型提供者（model providers）、数据存储（data stores）、API（APIs） 等集成。在本示例中，我们将使用
OpenAI 的模型 API。

首先我们需要安装他们的 Python 包：

```bash
pip install openai
```

访问 API 需要 API 密钥。

在当前目录下创建一个名为 `.env` 的文件，并将以下内容添加到其中：

```bash
OPENAI_API_KEY=YOUR_API_KEY
```

### LangSmith

您使用 LangChain 构建的许多应用程序将包含多个步骤以及多次调用 LLM
调用。随着这些应用程序变得越来越复杂，能够检查链或代理内部到底发生了什么变得至关重要。做到这一点的最佳方法是与 [LangSmith](https://smith.langchain.com/)
合作。

请注意，LangSmith 不是必需的，但它很有帮助。如果您确实想使用 LangSmith，请在上面的链接注册后，确保设置环境变量以开始记录跟踪：

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

### LangServe

LangServe帮助开发者将LangChain链部署为REST API。您不需要使用 LangServe 来使用 LangChain，但在本指南中，我们将向您展示如何使用
LangServe 部署您的应用程序。

安装命令:

```bash
pip install "langserve[all]"
```

## 使用 LangChain 构建一个简单的应用程序

LangChain提供了许多可用于构建语言模型应用程序的模块。模块可以在简单的应用程序中独立使用，也可以组合起来用于更复杂的用例。
应用由LangChain表达式语言（LangChain Expression Language: LCEL）提供支持，它定义了许多模块实现的统一 `Runnable`
接口，使得无缝链接组件成为可能。

最简单、最常见的链包含三个部分：

- LLM/Chat Model：语言模型是这里的核心推理引擎。为了使用 LangChain，您需要了解不同类型的语言模型以及如何使用它们
- Prompt Template：这向语言模型提供指令。这控制着语言模型的输出内容，因此了解如何构建提示和不同的提示策略至关重要。
- Output Parser：将来自语言模型的原始响应转换为更可行的格式，从而可以轻松使用下游的输出。

在本指南中，我们将分别介绍这三个组件，然后介绍如何组合它们。了解这些概念将为您使用和定制 LangChain 应用程序做好准备。大多数
LangChain 应用程序都允许您配置模型和/或提示，因此了解如何利用这一点将是一个很大的推动因素。

### LLM/Chat Model

有两种类型的语言模型：

- `LLM`: 底层模型将字符串作为输入并返回字符串
- `ChatModel`: 底层模型将消息列表作为输入并返回消息

字符串很简单，但是消息到底是什么？基本消息接口由 `BaseMessage` 定义，它有两个必需的属性：

- `content`：消息的内容。通常是一个字符串。
- `role`: `BaseMessage` 所来自的实体。

LangChain提供了几个对象来方便区分不同的角色：

- `HumanMessage`：来自人类/用户的 `BaseMessage` 。
- `AIMessage`：来自AI/助手的 `BaseMessage` 。
- `SystemMessage`：来自系统的 `BaseMessage` 。
- `FunctionMessage` / `ToolMessage`：包含函数或工具调用的输出的 `BaseMessage`。

如果这些角色听起来都不正确，还有一个 `ChatMessage` 类，您可以在其中手动指定角色。

LangChain 提供了一个由 `LLM` 和 `ChatModel` 共享的通用接口。然而，为了最有效地为给定的语言模型构建提示，理解差异是很有用的。

调用 `LLM` 或 `ChatModel` 的最简单方法是使用 `.invoke()`，这是所有 LangChain 表达式语言（LCEL）对象的通用同步调用方法:

- `LLM.invoke`：接收一个字符串，返回一个字符串。
- `ChatModel.invoke`：接受 `BaseMessage` 列表，返回 `BaseMessage` 。

> `LLM.invoke` 和 `ChatModel.invoke` 实际上都支持 `Union[str, List[BaseMessage], PromptValue]` 作为输入。
> `PromptValue` 是一个对象，它定义了自己的自定义逻辑，用于将其输入作为字符串或消息返回。
> `LLM` 具有将其中任何一个强制转换为字符串的逻辑，而 `ChatModel` 具有将其中任何一个强制转换为消息的逻辑。
> `LLM` 和 `ChatModel` 接受相同的输入这一事实意味着您可以在大多数链中直接将它们彼此交换而不会破坏任何内容，尽管考虑输入的方式当然很重要受到强制以及这可能如何影响模型性能。
> 要更深入地了解模型，请前往[语言模型]( https://python.langchain.com/docs/modules/model_io/)部分。


这些方法的输入类型实际上比这更通用，但为了简单起见，我们可以假设 `LLM` 仅接受字符串，而 `Chat` 模型仅接受消息列表。

让我们看看如何使用这些不同类型的模型和这些不同类型的输入。首先，我们导入一个 `LLM` 和一个 `ChatModel`。

`LLM` 和 `ChatModel` 对象是有效的配置对象。您可以使用 `temperature` 等参数初始化它们，然后传递它们。

```python
import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from dotenv import load_dotenv


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


def test_llm_and_chat_model():
    llm = OpenAI()
    chat_model = ChatOpenAI()

    text = "对于一家生产彩色袜子的公司来说，一个好的公司名称是什么？"
    messages = [HumanMessage(content=text)]

    llm_res = llm.invoke(text)
    print(llm_res)

    chat_res = chat_model.invoke(messages)
    print(chat_res)


def main():
    # 加载环境变量
    load_env()

    test_llm_and_chat_model()


if __name__ == '__main__':
    main()

```

![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-GhCWEK.png)

### Prompt Template(提示词模板)

大多数 `LLM` 应用不会直接将用户输入传递给 `LLM` 。通常，他们会将用户输入添加到较大的文本中，称为提示模板（prompt
template），该文本提供有关当前特定任务的附加上下文。

在前面的示例中，我们传递给模型的文本包含生成公司名称的指令。如果应用程序是一个给公司起名的需求，根据用户输入的产品名称来给公司取名，可以用提示词模板（Prompt
Templates）解决。

```python
import os

from langchain.prompts import PromptTemplate

from dotenv import load_dotenv


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


def test_prompt_template():
    prompt = PromptTemplate.from_template("对于一家生产{product}的公司来说，一个好的公司名称是什么？")
    s = prompt.format(product="彩色袜子")
    print(s)


def main():
    # 加载环境变量
    load_env()

    test_prompt_template()


if __name__ == '__main__':
    main()

```

![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-e2cQlp.png)

与 Python 原始字符串格式化相比，使用它们有几个优点。您可以“部分”输出变量 -
例如您一次只能格式化部分变量。您可以将它们组合在一起，轻松地将不同的模板组合成一个提示。有关这些功能的说明，请参阅有关[提示词](https://python.langchain.com/docs/modules/model_io/prompts)
的部分以了解更多详细信息。

`PromptTemplate`
也可用于生成消息列表。在这种情况下，提示不仅包含有关内容的信息，还包含每条消息（其角色、在列表中的位置等）。最常用的是 `ChatPromptTemplates`
，一个元素为 `ChatMessageTemplate` 的列表。每个 `ChatMessageTemplate` 都包含有关如何格式化 `ChatMessage` 的说明, 包括角色和内容。

```python
import os

from langchain.prompts.chat import ChatPromptTemplate

from dotenv import load_dotenv


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


def test_chat_prompt_template():
    template = "您是一个有用的助手，可以将 {input_language} 转换为 {output_language}。"
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    msg = chat_prompt.format_messages(input_language="English", output_language="中文", text="I love programming.")
    print(msg)


def main():
    # 加载环境变量
    load_env()

    test_chat_prompt_template()



if __name__ == '__main__':
    main()

```

![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-WR6pTj.png)

### Output parser(输出解析器)

`OutputParser` 将语言模型的原始输出转换为可以在下游使用的格式。 `OutputParser` 有几种主要类型，包括：

- 将 `LLM` 生成的文本转换为结构化信息（例如 JSON）
- 将 `ChatMessage` 转换为字符串
- 将 `__call__` 方法返回的除消息之外的额外信息（如 OpenAI 函数调用）转换为字符串。

有关这方面的完整信息，请参阅有关[输出解析器](https://python.langchain.com/docs/modules/model_io/output_parsers)的部分。

在本入门指南中，我们将编写自己的输出解析器 - 将逗号分隔列表转换为列表的解析器。

```python
import os
from typing import List

from langchain.schema import BaseOutputParser

from dotenv import load_dotenv


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """将 LLM 调用的输出解析为逗号分隔的列表。"""

    def parse(self, text: str) -> List[str]:
        """解析 LLM 调用的输出"""
        return text.strip().split(", ")


def test_my_output_parser():
    parser = CommaSeparatedListOutputParser()
    print(parser.parse("hi, bye"))


def main():
    # 加载环境变量
    load_env()

    test_my_output_parser()


if __name__ == '__main__':
    main()

```

![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-GS2vkR.png)

### 使用 LCEL 组合

我们现在可以将所有这些组合成一条链。该链将获取输入变量，将这些变量传递给提示词模板以创建提示词，将提示词传递给语言模型，然后通过（可选）输出解析器传递输出。

```python
#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: quick_start.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 快速开始

参考链接： https://python.langchain.com/docs/get_started/quickstart
"""
import os
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

from dotenv import load_dotenv


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """将 LLM 调用的输出解析为逗号分隔的列表。"""

    def parse(self, text: str) -> List[str]:
        """解析 LLM 调用的输出"""
        return text.strip().split(", ")


def test_composing_with_lcel():
    template = """你是一个有用的助手，可以生成逗号分隔的列表。
    用户将传入一个类别，你应该在列表中生成 5 个该类别的对象。
    只返回一个逗号分隔的列表，仅此而已。"""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
    res = chain.invoke({"text": "colors"})
    print(f"res type: {type(res)}, res: {res}")


def main():
    # 加载环境变量
    load_env()

    test_composing_with_lcel()


if __name__ == '__main__':
    main()

```

结果如下：
![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-ku6cdf.png)


## 使用LangSmith

由 LangSmith 还未开发测试，需要邀请码才能使用，暂时使用不了。


## 使用 LangServe 提供服务

### web server

要为我们的应用程序创建一个Web服务，我们将创建一个包含三项内容的 `serve.py` 文件：

1. 定义 chain
2. 添加 fastapi app
3. 通过 `langserve.add_routes` 给 app 添加路由

```python
import os
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langserve import add_routes


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """将 LLM 调用的输出解析为逗号分隔的列表。"""

    def parse(self, text: str) -> List[str]:
        """解析 LLM 调用的输出"""
        return text.strip().split(", ")


def define_chain():
    template = """你是一个有用的助手，可以生成逗号分隔的列表。
    用户将传入一个类别，你应该在列表中生成 5 个该类别的对象。
    只返回一个逗号分隔的列表，仅此而已。"""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
    return chain


def define_app():
    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple API server using LangChain's Runnable interfaces",
    )
    return app


def main():
    # 加载环境变量
    load_env()
    # 定义一个chain
    chain = define_chain()
    # 定义一个fastapi的app
    app = define_app()
    # 添加路由
    add_routes(app, chain, path="/category_chain")
    # 启动服务
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == '__main__':
    main()
```

执行 Python 文件启动服务。

### Playground

每个 LangServe 服务都带有一个简单的内置 UI，用于配置和调用具有流输出和中间步骤可见性的应用程序。前往 [http://localhost:8000/category_chain/playground/](http://localhost:8000/category_chain/playground/) 尝试一下！

![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-3oACNC.png)


### Client

现在让我们设置一个客户端，以便以编程方式与我们的服务进行交互。我们可以使用 `langserve.RemoteRunnable` 轻松做到这一点。使用它，我们可以与服务链进行交互，就像它在客户端运行一样。

```python
from langserve import RemoteRunnable


def main():
    remote_chain = RemoteRunnable("http://localhost:8000/category_chain/")
    res = remote_chain.invoke({"text": "colors"})
    print(f"type res: {type(res)}, res: {res}")
```
![](https://danerlt-1258802437.cos.ap-chongqing.myqcloud.com/2023-12-27-dqPiqG.png)

