#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: test_tongyi.py
@time: 2023-12-28
@contact: danerlt001@gmail.com
@desc: 
"""

import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi


def load_env():
    load_dotenv(verbose=True)

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_api_key is None:
        print("Please set DASHSCOPE_API_KEY in your environment.")
        raise ValueError("Please set DASHSCOPE_API_KEY in your environment.")


def get_prompt_template() -> ChatPromptTemplate:
    template = """
    <指令>
        根据已知信息，简洁和专业的来回答问题。
        如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。
    </指令>
    <已知信息>
        {context}
    </已知信息>
    <问题>
        {question}
    </问题>
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def get_model():
    model = ChatTongyi()
    return model


def main():
    load_env()
    model = get_model()
    prompt = get_prompt_template()
    prompt_value = prompt.invoke({"context": "猫吃老鼠", "question": "猫吃什么？"})
    print(prompt_value)
    res = model.invoke(prompt_value)
    print(res)


if __name__ == '__main__':
    main()
