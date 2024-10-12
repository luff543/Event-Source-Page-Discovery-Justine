# -*- coding: utf-8 -*-
# @Time : 2022/01/09 20:56
# @Author : luff543
# @Email : luff543@gmail.com
# @File : configure.py
# @Software: PyCharm

import sys


class Configure:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)
        the_item = 'log_dir'
        if the_item in config:
            self.log_dir = config[the_item]

        the_item = 'html_dir'
        if the_item in config:
            self.html_dir = config[the_item]

        the_item = 'img_dir'
        if the_item in config:
            self.img_dir = config[the_item]

    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        fins = open(input_file, 'r', encoding='utf-8').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == '#':
                continue
            if '=' in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                # noinspection PyBroadException
                try:
                    if item in config:
                        print('Warning: duplicated config item found: {}, updated.'.format((pair[0])))
                    if value[0] == '[' and value[-1] == ']':
                        value_items = list(value[1:-1].split(','))
                        config[item] = value_items
                    else:
                        config[item] = value
                except Exception:
                    print('configuration parsing error, please check correctness of the config file.')
                    exit(1)
        return config

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False

    @staticmethod
    def str2none(string):
        if string == 'None' or string == 'none' or string == 'NONE':
            return None
        else:
            return string

    def show_data_summary(self, logger):
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY' + '++' * 20)
        logger.info('     log               dir: {}'.format(self.log_dir))
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY END' + '++' * 20)
        sys.stdout.flush()
