#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: test_retriever.py
@time: 2023-12-28
@contact: danerlt001@gmail.com
@desc: 
"""

import os
import typing as t

from dotenv import load_dotenv
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever


def load_env():
    load_dotenv(verbose=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Please set OPENAI_API_KEY in your environment.")
        raise ValueError("Please set OPENAI_API_KEY in your environment.")


def get_test_texts():
    """获取测试用的文本信息"""
    texts = [
        "猴子吃香蕉”",
        "猫吃老鼠",
        "狗吃骨头",
        "王八吃绿豆糕"
    ]
    return texts


def get_retriever(texts: t.List[str]) -> VectorStoreRetriever:
    """获取检索器

    Args:
        texts: 向量库中存储的文本内容

    Returns: VectorStoreRetriever

    """
    vectorstore = DocArrayInMemorySearch.from_texts(texts, embedding=DashScopeEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever


def main():
    load_env()
    texts = get_test_texts()
    retriever = get_retriever(texts)
    res = retriever.invoke("猫吃什么")
    print(f"{type(res)=}, {res=}")


if __name__ == '__main__':
    main()
