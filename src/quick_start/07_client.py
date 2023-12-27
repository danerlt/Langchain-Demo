#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: 07_client.py
@time: 2023-12-27
@contact: danerlt001@gmail.com
@desc: 参考连接： https://python.langchain.com/docs/get_started/quickstart#client
"""

from langserve import RemoteRunnable


def main():
    remote_chain = RemoteRunnable("http://localhost:8000/category_chain/")
    res = remote_chain.invoke({"text": "colors"})
    print(f"type res: {type(res)}, res: {res}")


if __name__ == '__main__':
    main()
