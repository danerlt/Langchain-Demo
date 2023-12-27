#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: test_chat_prompt_template.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 参考链接： https://python.langchain.com/docs/get_started/quickstart
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser


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


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """将 LLM 调用的输出解析为逗号分隔的列表。"""

    def parse(self, text: str) -> List[str]:
        """解析 LLM 调用的输出"""
        return text.strip().split(", ")


def main():
    # 加载环境变量
    load_env()

    test_chat_prompt_template()


if __name__ == '__main__':
    main()
