# -*- coding: utf-8 -*-

# @Time : 2023/01/09 20:56
# @Author : luff543
# @Email : luff543@gmail.com
# @File : logger.py
# @Software: PyCharm

import datetime
import logging
import os.path


def get_logger(log_dir=None):
    log_file = os.path.join(log_dir,  (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    # formatter = logging.Formatter('%(message)s')
    formatter = logging.Formatter("[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s")

    # log into file
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.propagate = False

    return logger
