#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
class Tokenize:
    def __init__(self):
        print('Init Token')
        self.dictionary = None
    def TokenizeList(self, elems):
        self.dictionary = {v: k for k, v in enumerate(set(elems))}
    def SaveObj(self, filename):
        f = open(filename+ '.txt', 'w')
        json.dump(self.dictionary, f)
    def LoadObj(self, filename):
        f = open(filename+ '.txt', 'r')
        self.dictionary = json.load(f)
    def GetIDs(self, elems):
        idlist = [self.dictionary.get(elem,len(self.dictionary)) for elem in elems]
        return idlist
    def GetDictLength(self):
        return len(self.dictionary)+1

