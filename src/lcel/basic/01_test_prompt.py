#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: 01_test_prompt.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 参考链接：https://python.langchain.com/docs/expression_language/get_started#prompt
"""
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
