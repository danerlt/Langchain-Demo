#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: 03_output_parser.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.openai import OpenAI
from langchain_core.output_parsers import StrOutputParser


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
    return message


def test_llm_model():
    print(f"{'*' * 20} test_llm_model {'*' * 20}")
    prompt = ChatPromptTemplate.from_template("给我将一个关于{topic}的笑话")
    prompt_value = prompt.invoke({"topic": "冰淇淋"})
    print(f"{type(prompt_value)=}, {prompt_value=}\n")

    model = OpenAI(model="gpt-3.5-turbo-instruct")

    message = model.invoke(prompt_value)
    print(f"{type(message)=}, message: {message}")
    return message


def test_output_parser():
    print(f"{'*' * 20} test_output_parser {'*' * 20}")
    parser = StrOutputParser()

    chat_message = test_chat_model()
    chat_res = parser.invoke(chat_message)
    print(f"{type(chat_res)=}, {chat_res=}")

    llm_message = test_llm_model()
    llm_res = parser.invoke(llm_message)
    print(f"{type(llm_res)=}, {llm_res=}")


def main():
    load_dotenv()
    test_output_parser()


if __name__ == '__main__':
    main()
