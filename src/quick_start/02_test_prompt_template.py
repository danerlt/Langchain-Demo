#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: 02_test_prompt_template.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 参考链接： https://python.langchain.com/docs/get_started/quickstart
"""

import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


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
