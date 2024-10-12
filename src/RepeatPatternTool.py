#!/usr/bin/env python
# coding: utf-8

# In[1]:


#refer: https://github.com/bornabr/SuffixTree


# In[2]:


import requests
from selenium import webdriver
from selenium.webdriver import Chrome
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
# from collections import defaultdict
class RepeatTool():
    def __init__(self):
        pass
    def GetLeafElement(self,url):
        sourcePage = self.GetSourcePage(url)
        soup = BeautifulSoup(sourcePage, features="html.parser")
        tagelems = self.recursiveChildren(soup)
        return tagelems
    def GetTagElement(self,url):
        sourcePage = self.GetSourcePage(url)  # GetSourcePag會去進行爬蟲
        soup = BeautifulSoup(sourcePage, features="html.parser")
        tagFeatures = []
        # print('soup',soup)
        for tag in soup.find_all():
            if tag.name == 'script':
                continue
            tagFeatures.append([tag.name,tag.text])
        return tagFeatures
    def ConstructTagPathDictionary(self, tagelems):
        tagencodedict = {}
        num = 0
        for tmp in tagelems:
            if str(tmp[0]) not in tagencodedict:
                tagencodedict[str(tmp[0])] = chr(num)
                num += 1
        """self.tagencodeinversedict = {v: k for k, v in self.tagencodedict.items()}
        self.tagstring = ''
        for tag in self.tagelems:
             self.tagstring += self.tagencodedict[str(tag[0])]"""
        return tagencodedict
    def ConvertTagToString(self, dictionary, tagelems):
        tagstring = ""
        for tag in tagelems:
             tagstring += dictionary[str(tag[0])]
        return tagstring
    def GetTagAppearTimeAndContent(self, tagelems):
        appeartime = {}
        tagcontentMap = {}
        for elem in tagelems:
            if str(elem[0]) in appeartime:
                appeartime[str(elem[0])] += 1
                tagcontentMap[str(elem[0])] += elem[1] + ' '
            else:
                appeartime[str(elem[0])] = 1
                tagcontentMap[str(elem[0])] = elem[1] + ' '
        return appeartime, tagcontentMap


    def GetSourcePage(self, url):
        try:
            res = requests.post("http://140.115.54.45:6789/post/crawler/static/html",
								json={"urls": [url], "cache": False})
            webContent = eval(res.content.decode("utf-8"))[0]
        except:
            return ''

        return webContent
    def GetSourcePage_pre(self, url):
        try:
            headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
                        'From': 'justine8811@gmail.com'
                       }
            response = requests.get(url, headers=headers,timeout=15)
            webContent = response.text
            return webContent
        except:
            return ''
        finally:
            # print('cannot get source page')
            chrome_options = webdriver.ChromeOptions()
            # 添加 User-Agent
            chrome_options.add_argument(
                'user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"')

            # 指定瀏覽器解析度
            chrome_options.add_argument('window-size=1920x1080')
            # 不載入圖片，提升速度
            chrome_options.add_argument('blink-settings=imagesEnabled=false')
            # 禁用 JavaScript
            chrome_options.add_argument("--disable-javascript")

            # 禁用瀏覽器彈出視窗
            prefs = {
                'profile.default_content_setting_values': {
                       'notifications': 2
                }
            }
            chrome_options.add_experimental_option('prefs', prefs)

            driver = Chrome(chrome_options=chrome_options)
            url = url.encode('ascii', 'ignore').decode('unicode_escape')
            driver.get(url)
            time.sleep(5)
            # 取得網頁原始碼
            webContent = driver.page_source
            # webContent = BeautifulSoup(html, 'html.parser')
            # print(webContent)
            driver.close()
            return webContent
        # return webContent
    def recursiveChildren(self, x, paths = []):
        childrens = x.findChildren()
        leafs = []
        paths.append(x.name)
        if len(childrens)!=0:
            for child in childrens:
                if child is not None:
                    output = self.recursiveChildren(child,paths = copy.deepcopy(paths))
                    if output is not None:  
                        leafs += output
            return leafs
        else:
            if self.CleanString(x.text)!='': #Just to avoid printing "\n" parsed from document.
                #leafs.append(paths)
                #leafs.append(self.CleanString(x.text))
                return [(copy.deepcopy(paths),self.CleanString(x.text))]#.append(x.name)
        #return leafs
    def CleanString(self, string):
        #for string in liststring:
            #print(string)
        cleantext = string.replace('\n','')
        cleantext = cleantext.replace('\t','')
        cleantext = cleantext.replace('\r','')
        cleantext = cleantext.replace(' ','')
            #if(cleantext!='' and (not cleantext.isdecimal())):
            #newlist.append(cleantext)

        return cleantext
    def ConstructSuffixTree(self,string):
        tree = SuffixTree([string], True)
        return 
class KMP:
    def partial(self, pattern):
        """ Calculate partial match table: String -> [Int]"""
        ret = [0]
        
        for i in range(1, len(pattern)):
            j = ret[i - 1]
            while j > 0 and pattern[j] != pattern[i]:
                j = ret[j - 1]
            ret.append(j + 1 if pattern[j] == pattern[i] else j)
        return ret
        
    def search(self, T, P):
        """ 
        KMP search main algorithm: String -> String -> [Int] 
        Return all the matching position of pattern string P in T
        """
        partial, ret, j = self.partial(P), [], 0
        
        for i in range(len(T)):
            while j > 0 and T[i] != P[j]:
                j = partial[j - 1]
            if T[i] == P[j]: j += 1
            if j == len(P): 
                ret.append(i - (j - 1))
                j = partial[j - 1]
            
        return ret


# In[3]:


from operator import attrgetter

class Node:
	"""The Suffix-tree's node."""

	def __init__(self, tree, leaf):
		self.children = {}
		self.leaf = leaf
		self.suffixIndex = None
		self.start = None
		self.end = None 
		self.suffixLink = None
		self.forwardIndices = set()
		self.reverseIndices = set()
		self.tree = tree

	def edge_length(self):
		if(self == self.tree.root):
			return 0
		return self.end - self.start + 1

	def __eq__(self, node):
		atg = attrgetter('start', 'end', 'suffixIndex')
		return atg(self) == atg(node)

	def __ne__(self, node):
		atg = attrgetter('start', 'end', 'suffixIndex')
		return atg(self) != atg(node)

	def __str__(self):
		atg = attrgetter('start', 'end', 'suffixIndex')
		return str(atg(self))

	def __hash__(self):
		atg = attrgetter('start', 'end', 'suffixIndex')
		return hash(atg(self))

	def __getattribute__(self, name):
		if name == 'end':
			if self.leaf:
				return self.tree.leafEnd
		if name == 'childrenArray':
			return self.children.values()
		return super(Node, self).__getattribute__(name)


import sys

class SuffixTree(object):
	"""The Generilized Suffix Tree"""

	def __init__(self, strings: list, buildNow=False, checkForPalindrom=False):
		self.terminalSymbolsGenerator = self._terminalSymbolsGenerator()
		self.checkForPalindrom = checkForPalindrom
		self.strings = list()
		self.titles = list()
		if self.checkForPalindrom:
			if isinstance(strings[0], tuple):
				self.strings.append(strings[0][1] + '#')
				self.strings.append(strings[0][1][::-1] + '$')
				self.titles.append(strings[0][0])
			else:
				self.strings.append(strings[0] + '#')
				self.strings.append(strings[0][::-1] + '$')
				self.titles.append('0')
		else:
			for index, string in enumerate(strings):
				if isinstance(string, tuple):
					self.strings.append(string[1] + self.terminalSymbolsGenerator.__next__())
					self.titles.append(string[0])
				else:
					self.strings.append(string + self.terminalSymbolsGenerator.__next__())
					self.titles.append(str(index))
		self.string = "".join(self.strings)
		self.lastNewNode = None

		""" active point is stored a dict {node, edge, length},
			also activeEdge is presented by input string index """
		self.activePoint = {
			"node": None,
			"edge": -1,
			"length": 0
		}

		""" remainingSuffixCount, to track how many extensions are yet to be performed explicitly in any phase """
		self.remainingSuffixCount = 0

		self.rootEnd = None
		self.size = len(self.string)  # Length of input string
		self.root = None
		if buildNow:
			self.build()

	def _terminalSymbolsGenerator(self):
		""" Generator of unique terminal symbols used for building the Generalized Suffix Tree.
		Unicode Private Use Area U+E000..U+F8FF is used to ensure that terminal symbols
		are not part of the input string. """
		py2 = sys.version[0] < '3'
		UPPAs = list(list(range(0xE000,0xF8FF+1)) + list(range(0xF0000,0xFFFFD+1)) + list(range(0x100000, 0x10FFFD+1)))
		# UPPAs = map(lambda x: ord(x), ["$", "#", "%"])
		for i in UPPAs:
			if py2:
				yield(unichr(i))
			else:
				yield(chr(i))
		raise ValueError("To many input strings.")

	def new_node(self, start, end=None, leaf=False):
		""" For root node, suffixLink will be set to NULL
		For internal nodes, suffixLink will be set to root
		by default in  current extension and may change in
		next extension """
		node = Node(self, leaf)
		node.suffixLink = self.root
		node.start = start
		node.end = end
		""" suffixIndex will be set to -1 by default and
		   actual suffix index will be set later for leaves
		   at the end of all phases """
		node.suffixIndex = -1
		return node

	def walk_down(self, current_node):
		""" Walk down from current node.
		activePoint change for walk down (APCFWD) using
		Skip/Count Trick  (Trick 1). If activeLength is greater
		than current edge length, set next  internal node as
		activeNode and adjust activeEdge and activeLength
		accordingly to represent same activePoint. """
		length = current_node.edge_length()
		if (self.activePoint["length"] >= length):
			self.activePoint["edge"] += length
			self.activePoint["length"] -= length
			self.activePoint["node"] = current_node
			return True
		return False
	
	def extend(self, phase):
		""" Extension Rule 1, this takes care of extending all
		leaves created so far in tree (trick 3 - Once a leaf, always a leaf) """
		self.leafEnd = phase

		self.remainingSuffixCount += 1

		""" Set lastNewNode to None while starting a new phase,
		indicating there is no internal node waiting for
		it's suffix link reset in current phase """
		self.lastNewNode = None

		""" Run a loop for the remaining extensions """
		while(self.remainingSuffixCount > 0):

			if (self.activePoint["length"] == 0):
				""" activePoint change for Active Length ZERO (APCFALZ) """
				self.activePoint["edge"] = phase

			if (self.activePoint["node"].children.get(self.string[self.activePoint["edge"]]) is None):
				""" There is no outgoing edge
				starting with activeEdge from activeNode """

				""" Extension Rule 2 (A new leaf edge gets created) """
				self.activePoint["node"].children[self.string[self.activePoint["edge"]]] = self.new_node(
					phase, leaf=True)

				if (self.lastNewNode is not None):
					""" If there is an internal node waiting for it's suffix link
					point the suffix link from that internal node to current active node """
					self.lastNewNode.suffixLink = self.activePoint["node"]
					self.lastNewNode = None
			else:
				""" There is an outgoing edge starting with activeEdge from activeNode """
				next_node = self.activePoint["node"].children.get(
					self.string[self.activePoint["edge"]])

				if(self.walk_down(next_node)):
					""" Start from the next_node """
					continue

				""" Extension Rule 3 (current character being processed
					is already on the edge) """

				if (self.string[next_node.start + self.activePoint["length"]] == self.string[phase]):

					if((self.lastNewNode is not None) and (self.activePoint['node'] != self.root)):
						""" If there is an internal node waiting for it's suffix link
							point the suffix link from that internal node to current active node """
						self.lastNewNode.suffixLink = self.activePoint["node"]
						self.lastNewNode = None

					""" Now it's time to go to next phase,
						we increament active length by one before that (APCFER3) """
					self.activePoint["length"] += 1
					break

				""" We will be here when activePoint is in middle of
					the edge being traversed and current character
					being processed is not on the edge (we fall off
					the tree). In this case, this is Extension Rule 2 """
				splitEnd = next_node.start + self.activePoint['length'] - 1
				splitNode = self.new_node(next_node.start, splitEnd)
				self.activePoint["node"].children[self.string[self.activePoint["edge"]]] = splitNode
				splitNode.children[self.string[phase]] = self.new_node(phase, leaf=True)
				next_node.start += self.activePoint['length']
				splitNode.children[self.string[next_node.start]] = next_node
				
				if (self.lastNewNode is not None):
					""" If there is an internal node waiting for it's suffix link
						point the suffix link from that internal node to new splitNode """
					self.lastNewNode.suffixLink = splitNode
				
				""" Now we set splitNode as the lastNewNode
					so that it's suffix like be set in the future """
				self.lastNewNode = splitNode
			
			""" One suffix got added in tree, decrement the count of
			   suffixes yet to be added. Note that this below code won't be run for APCFER3"""
			self.remainingSuffixCount -= 1
			
			if ((self.activePoint["node"] == self.root) and (self.activePoint['length'] > 0)):
				""" APCFER2C1 """
				self.activePoint['length'] -= 1
				self.activePoint["edge"] = phase - self.remainingSuffixCount + 1
			elif (self.activePoint["node"] != self.root):
				""" APCFER2C2 """
				self.activePoint["node"] = self.activePoint["node"].suffixLink
	
	
	def setSuffixIndexByDFS(self, node, labelHeight):
		if(node is Node):
			return
		
		isLeaf = True
		for child in node.children.values():
			isLeaf = False
			self.setSuffixIndexByDFS(child, labelHeight + child.edge_length())
			if node != self.root:
				node.forwardIndices = node.forwardIndices.union(child.forwardIndices)
				node.reverseIndices = node.reverseIndices.union(child.reverseIndices)
		if(isLeaf):
			for i in range(node.start, node.end + 1):
				if(self.string[i] == '#'):
					node.end = i
			node.suffixIndex = self.size - labelHeight
			
			if (node.suffixIndex < len(self.strings[0])):
				node.forwardIndices.add(node.suffixIndex)
			else:
				node.reverseIndices.add(node.suffixIndex - len(self.strings[0]))


	def build(self):
		self.rootEnd = -1
		self.leafEnd = -1
		self.root = self.new_node(-1, self.rootEnd)
		self.activePoint["node"] = self.root
		for phase in range(self.size):
			self.extend(phase)
		self.setSuffixIndexByDFS(self.root, 0)
	
	def walk_dfs(self, current, parent=None):
		start, end = current.start, current.end
		index = self.index
		self.nodes.append({ 'id': index, 'label': str(index) })
		if(parent is not None):
			self.edges.append({
				'from': parent,
				'to': index,
				'label': str(start) + ':' + str(end)
			})

		for node in current.children.values():
			self.index += 1
			self.walk_dfs(node, index)
	
	def __dict__(self):
		self.nodes = []
		self.edges = []
		self.index = 1
		self.walk_dfs(self.root)
		return {
			'nodes': self.nodes,
			'edges': self.edges
		}
	
	def __str__(self):
		return str(self.__dict__())
    
#from .base import Base
class Base:
	def __init__(self, tree):
		self.tree = tree
		self.string = tree.string
		self.root = tree.root

class LRS(Base):
	"""Longest Repeated Substring"""

	def __init__(self, tree, k):
		super(LRS, self).__init__(tree)
		self.k = k
		self.maxHeight = 0
		self.currentHeight = 0
		self.subStringStartIndex = 0
		self.currentSubStringStartIndex = 0
		self.numberOfInternalNodes = 0

	def doTraversal(self,
					node,
					labelHeight):

		if node is None:
			return 0
		if not node.leaf:
			count = 0
			for child in node.children.values():
				res = self.doTraversal(child, labelHeight + child.edge_length())
				if node == self.root:
					if self.numberOfInternalNodes >= self.k:
						if self.maxHeight < self.currentHeight:
							self.maxHeight = self.currentHeight
							self.substringStartIndex = self.currentSubStringStartIndex
					self.currentHeight = 0
					self.currentSubStringStartIndex = 0
					self.change = False
					self.numberOfInternalNodes = 0
				else:
					count += res
			if self.change:
				self.numberOfInternalNodes = count
				self.change = False
			return count
		elif self.currentHeight < labelHeight - node.edge_length():
			self.change = True
			self.currentHeight = labelHeight - node.edge_length()
			self.currentSubStringStartIndex = node.suffixIndex
		return 1

	def find(self):
		self.change = False
		self.maxHeight = 0
		self.substringStartIndex = 0
		self.numberOfInternalNodes = 0
		self.currentHeight = 0
		self.currentSubStringStartIndex = 0
		self.doTraversal(self.root, 0)
		if self.maxHeight:
			return self.string[self.substringStartIndex: self.substringStartIndex + self.maxHeight]
		else:
			return -1


# In[5]:


import copy
if __name__ == '__main__':
	tool = RepeatTool()
	result = tool.GetSourcePage('https://activity.nlpi.edu.tw/front/index')
	print(result)
	# tool = RepeatTool()
    # elements = tool.GetTagElement("https://www.tys.org.tw/fycm/article_cate_list.asp?article_cate_id=2&article_sub_cate=%E4%BD%9B%E5%AD%B8%E8%AA%B2%E7%A8%8B")
    # dictionary = tool.ConstructTagPathDictionary(elements)
    # #reversedictionary = {v: k for k, v in dictionary.items()}
    # string = tool.ConvertTagToString(dictionary, elements)
	#
    # tags,contents = copy.deepcopy(zip(*elements))
	#
    # tree = SuffixTree([string], True)
    # for times in range(10,1,-1):
    #     lrs = LRS(tree, times)
    #     repeattags = lrs.find()
    #     if repeattags != -1:
    #         break
	#
    # kmptool = KMP()
    # startindexs = kmptool.search([dictionary[tag] for tag in tags], repeattags)
	#
    # for start in startindexs:
    #     tmp = ""
    #     for c in contents[start:start+len(repeattags)]:
    #         if tool.CleanString(c) != '':
    #             tmp += tool.CleanString(c)
    #     print(tmp)






