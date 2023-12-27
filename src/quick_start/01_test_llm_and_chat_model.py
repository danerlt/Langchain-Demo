#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: 01_test_llm_and_chat_model.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 参考链接： https://python.langchain.com/docs/get_started/quickstart
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage


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
