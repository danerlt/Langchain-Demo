#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: rag.py
@time: 2023-12-28
@contact: danerlt001@gmail.com
@desc: 参考链接： https://python.langchain.com/docs/expression_language/get_started#rag-search-example
"""
import os
import typing as t

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
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


def get_output_parser():
    output_parser = StrOutputParser()
    return output_parser


def get_rag_chain(retriever: VectorStoreRetriever) -> RunnableSequence:
    """获取rag chain

    Args:
        retriever: 检索器

    Returns: RunnableSequence

    """
    prompt = get_prompt_template()
    model = get_model()
    output_parser = get_output_parser()
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | model | output_parser
    print(f"{type(chain)=}, {chain=}")
    return chain


def test_chain(chain, user_inputs: t.List[str]):
    outputs = chain.batch(user_inputs)
    for i, output in enumerate(outputs):
        user_input = user_inputs[i]
        print(f"Q:{user_input}\nA:{output}")


def main():
    load_env()
    texts = get_test_texts()
    retriever = get_retriever(texts)
    chain = get_rag_chain(retriever)
    test_inputs = ["猫吃什么？",
                   "王八吃什么？",
                   "猪吃什么？"]
    test_chain(chain, test_inputs)


if __name__ == '__main__':
    main()
