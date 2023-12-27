#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: 06_serve.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 参考链接： https://python.langchain.com/docs/get_started/quickstart#serving-with-langserve
"""
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
