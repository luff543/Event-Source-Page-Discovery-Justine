# -*- coding: utf-8 -*-
# @Time : 2023/01/09 20:48
# @Author : luff543
# @Email : luff543@gmail.com
# @File : io.py
# @Software: PyCharm

import os


def fold_check(configures):

    the_item = "log_dir"
    the_value = getattr(configures, the_item)
    if not os.path.exists(the_value):
        print(f"{the_value} fold not found, creating...")
        if hasattr(configures, the_item):
            os.mkdir(the_value)

    the_item = "html_dir"
    the_value = getattr(configures, the_item)
    if not os.path.exists(the_value):
        print(f"{the_value} fold not found, creating...")
        if hasattr(configures, the_item):
            os.mkdir(the_value)

    the_item = "img_dir"
    the_value = getattr(configures, the_item)
    if not os.path.exists(the_value):
        print(f"{the_value} fold not found, creating...")
        if hasattr(configures, the_item):
            os.mkdir(the_value)

def make_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)