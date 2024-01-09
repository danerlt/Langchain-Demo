#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: llama_index_demo.py
@time: 2024-01-04
@contact: danerlt001@gmail.com
@desc: 
"""

import os

from dotenv import load_dotenv
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import download_loader, ListIndex
from llama_index import GPTKeywordTableIndex, SimpleDirectoryReader
from IPython.display import Markdown, display
from langchain_community.chat_models.tongyi import ChatTongyi

from common.const import DATA_PATH

logger.add("index.log")


def load_env():
    load_dotenv(verbose=True)

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_api_key is None:
        logger.error("Please set DASHSCOPE_API_KEY in your environment.")
        raise ValueError("Please set DASHSCOPE_API_KEY in your environment.")


def load_file(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")
    logger.debug("before download loader")
    UnstructuredReader = download_loader('UnstructuredReader', refresh_cache=False)
    logger.debug("after download loader")
    loader = UnstructuredReader()
    logger.debug("after init loader")
    documents = loader.load_data(file_path, split_documents=False)
    logger.debug("after load data")
    logger.debug(documents)
    return documents


def list_index(documents):
    index = ListIndex.from_documents(documents)

    query_engine = index.as_query_engine()
    response = query_engine.query("针对扭矩控制法紧固点，最终控制的是什么？")
    display(Markdown(f"<b>{response}</b>"))

    ## Check the logs to see the different between th
    ## if you wish to not build the index during the index construction
    # then need to add retriever_mode=embedding to query engine
    # # query with embed_model specified
    # new_index = ListIndex.from_documents(documents, retriever_mode="embedding")
    # query_engine = new_index.as_query_engine(
    #     retriever_mode="embedding",
    #     verbose=True
    # )
    # response = query_engine.query("针对扭矩控制法紧固点，最终控制的是什么？")
    # display(Markdown(f"<b>{response}</b>"))


def main():
    file_path = DATA_PATH.joinpath("test.txt")
    documents = load_file(file_path)
    list_index(documents)


if __name__ == '__main__':
    main()
