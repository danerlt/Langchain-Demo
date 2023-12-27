#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: basic_demo.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 参考链接： https://python.langchain.com/docs/expression_language/get_started
"""
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
