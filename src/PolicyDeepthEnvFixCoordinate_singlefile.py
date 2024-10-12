#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import time

import gym
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
from url_normalize import url_normalize
import urllib
from urllib.parse import urljoin, urlparse

if __name__ == '__main__':
    import URLTool
    import StringTool
    import Tool
else:
    from src import URLTool
    from src import StringTool
    from src import Tool

import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.preprocessing import MinMaxScaler
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import ssl
import copy

# WINDOW_SIZE = "1920,1080"
chrome_options = Options()
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('--headless')  # 不顯示爬蟲視窗
chrome_options.add_argument('--no-sandbox')  # 以最高權限運行
chrome_options.add_argument("--incognito")  # 無痕
chrome_options.add_argument('--disable-dev-shm-usage')  # docker原本的分享記憶體在 /dev/shm 是 64MB，會造成chrome crash，所以要改成寫入到 /tmp
# chrome_options.add_argument(
# 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36')
chrome_options.set_capability('unhandledPromptBehavior', 'accept')
prefs = {
        "profile.default_content_setting_values.notifications": 2
    }
chrome_options.add_experimental_option("prefs", prefs)
ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == '__main__':
    config = Tool.LoadConfig('../config.ini')
else:
    config = Tool.LoadConfig('./config.ini')
from bs4 import BeautifulSoup


# In[7]:


class Env(gym.Env):
    # def __init__(self, train_group, window_len, negativeReward=0.3, rewardweight = 0.1):
    def __init__(self, train_group, negativeReward=-0.1, rewardweight=0.1):
        # self.useETLCrawlService = useETLCrawlService
        self.homepage = ""
        self.currentURL = ""
        # self.url_idx = ""
        self.anchorText = []
        self.tagPath = []
        self.hrefs = []
        self.coordinates = []
        # self.positivePages = #[i+'/' if i[-1]!='/' else i for i in list(train_group['Event Source Page URL'])]
        # self.positivePages = URLTool.GetNormalizaUrl(list(train_group['Event Source Page URL']))  # ['http'+i[5:] if i[0:5]=='https' else i for i in self.positivePages]
        self.positivePages = URLTool.GetNormalizaUrl(list(train_group['Event source page']))  # ['http'+i[5:] if i[0:5]=='https' else i for i in self.positivePages]
        self.newpositivePages = list(train_group['Event source page'])
        self.train_group = train_group
        self.negativeReward = negativeReward
        self.rewardweight = rewardweight
        self.web = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        # self.web.set_page_load_timeout(240)
        self.pageScore = {}

    # def __del__(self):
    #    self.web.close()
    def LeaveThisURL(self, page, href):
        if (not URLTool.IsUrlString(href)):
            href = urljoin(page, href)
            if (not URLTool.IsUrlString(href)):
                return False
        return True

    def GetCoordinate(self, element):
        while (element.size['width'] == 0):
            # element = element.find_elements(By.CSS_SELECTOR, '..')
            element = element.find_elements(By.XPATH, '..')
            # element = element.find_element_by_xpath(By.XPATH, '..')
        x = element.location['x']
        y = element.location['y']
        width = element.size['width']
        height = element.size['height']
        return x, y, width, height
    def GetCoordinatebyCSS(self, element):
        text = []
        while (element.size['width'] == 0):
            element = element.find_elements(By.CSS_SELECTOR, '..')
        x = element.location['x']
        y = element.location['y']
        width = element.size['width']
        height = element.size['height']
        return x, y, width, height

    def GetTagPath(self, element):
        nowtag = element.tag_name
        thispath = [nowtag]
        # print('thispath',thispath)
        while True:
            try:
                # element = element.find_element_by_xpath('..')
                element = element.find_element(By.XPATH, '..')
                # print('element.tag_name',element.tag_name)
                thispath.append(element.tag_name)
            except Exception as e:
                break
        thispath = list(reversed(thispath))
        return thispath

    def GetPageScore(self,page):
        API_DOMAIN = f"{config.get('EventSourcePageScoring', 'IP')}:{config.get('EventSourcePageScoring', 'PORT')}"
        url = API_DOMAIN + "/GetPageScore"
        payload = {'url': str(page)}
        try:
            r = requests.get(url, params=payload)
        except:
            return 0

    # def GetTagPathBs4(self, element):
    #     nowtag = element.name
    #     thispath = [nowtag]
    #     while True:
    #         try:
    #             # print('hi')
    #             element = element.find_parent()
    #             thispath.append(element.name)
    #         except Exception as e:
    #             break
    #     thispath = list(reversed(thispath))
    #     return thispath

    def GetTagDeepth(self, element):
        depth = 0
        while True:
            try:
                element = element.find_elements(By.XPATH, '..')
                # element = element.find_element_by_xpath('..')
                depth += 1
            except Exception as e:
                break
        return depth

    def GetPositiveHrefIndex(self, hrefs):
        hrefs = URLTool.GetNormalizaUrl(hrefs)
        positiveIndexs = []
        for index, href in enumerate(hrefs):
            if href in self.positivePages:
                positiveIndexs.append(index)
        return positiveIndexs

    def ConvertPath(self, path):
        char_list = ['*', '|', ':', '?', '/', '<', '>', '"', '\\']
        news_title_result = path
        for i in char_list:
            if i in news_title_result:
                news_title_result = news_title_result.replace(i, "_")
        return news_title_result

    # def GetHTMLFile(self, url):
    #     API_DOMAIN = f"{config.get('CrawlService', 'Host')}:{config.get('CrawlService', 'PORT')}"
    #     api = "http://" + API_DOMAIN + "/post/crawler/dynamic/html"
    #     payload = {'urls': [url], 'timeout': 100, 'cache': 'true'}
    #     r = requests.post(api, data=payload)
    #     return r.json()[0]
    #     file = open(r"{}.html".format(self.ConvertPath(url)), "w+")
    #     file.write(r.json()[0])
    #     self.web.get(r"file:///{}.html".format(os.path.realpath(self.ConvertPath(url))))
    #     file.close()
    #     os.remove(r"{}.html".format(self.ConvertPath(url)))

    # def OpenHTMLFile(self, url):
    #     API_DOMAIN = f"{config.get('CrawlService', 'Host')}:{config.get('CrawlService', 'PORT')}"
    #     api = "http://" + API_DOMAIN + "/post/crawler/dynamic/html"
    #     payload = {'urls': [url], 'timeout': 100, 'cache': 'true'}
    #     r = requests.post(api, data=payload)
    #     file = open(r"{}.html".format(os.path.realpath(self.ConvertPath(url))), "w+")
    #     file.write('data:text/html;charset=utf-8,{}'.format(r.json()[0]))
    #     print("file://{}.html".format(os.path.realpath(self.ConvertPath(url))))
    #     self.web.get(r"file://{}.html".format(os.path.realpath(self.ConvertPath(url))))
    #     # self.web.execute_script("document.write(data:text/html;charset=utf-8,{})".format(r.json()[0]))
    #     print(self.web.page_source)
    #     file.close()
    #     os.remove(r"{}.html".format(os.path.realpath(self.ConvertPath(url))))


    def GetTagPath2(self, parse_html,idx):
        thispath = []
        hrefElement = parse_html.select('a')[idx]
        thispath.append(hrefElement.name)
        hrefParent = hrefElement.find_parent()
        while (hrefParent.name != "html"):
            # print(hrefParent.name)
            try:
                thispath.append(hrefParent.name)
                hrefParent = hrefParent.find_parent()
                # print(hrefParent)
            except Exception as e:
                break
        thispath.reverse()
        return thispath

    def GetHrefsContentAndHrefs(self, page=None, getcoordinates=False):

        hrefs = []
        tagPaths = []
        text = []
        print("Page:", page)
        res = requests.post("http://140.115.54.45:6789/post/crawler/static/html",
                            json={"urls": [page], "cache": False})
        try:
            html = eval(res.content.decode("utf-8"))[0]
        except Exception as e:
            # error_url.append(url)
            print("Cannot Open:", e)
            return [], [],[]

        parse_html = BeautifulSoup(html, 'html.parser')
        elems = parse_html.find_all('a',href=True)
        if len(elems) == 0:
            try:
                print("Page:", page)
                res = requests.post("http://140.115.54.45:6788/post/crawler/static/html",
                                    json={"urls": [page], "cache": False})
                html = eval(res.content.decode("utf-8"))[0]
            except Exception as e:
                print("Cannot Open:", e)
                return [], [],[]
            parse_html = BeautifulSoup(html, 'html.parser')
            # elems = parse_html.find_all(href=True)
            elems = parse_html.find_all('a', href=True)
        for idx, elem in enumerate(elems):
            try:
                tag_text = elem.getText()
                tag_href = elem['href']
                if tag_text == '':
                    continue
                if StringTool.CleanString(tag_text) == '':
                    continue
                if not self.LeaveThisURL(page, tag_href):
                    continue
                else:
                    tag_href = urljoin(page, tag_href)
                if tag_href == '' or '#' in tag_href:
                    continue
                text.append(StringTool.CleanString(tag_text))
                hrefs.append(tag_href)
                tagPaths.append(self.GetTagPath2(parse_html, idx))
            except Exception as e:
                print(e)
                continue
        if not getcoordinates:
            return text, hrefs, tagPaths
    def GetHrefContentbySelemium(self,page, getcoordinates=True):
        try:
            # self.web.set_page_load_timeout(180)
            res = urlparse(page)
            if res.netloc == 'www.accupass.com':
                self.web.get(page)
                time.sleep(10)
                print("Selenium_Page:", page)
            else:
                self.web.implicitly_wait(5)
                self.web.get(page)
                print("Selenium_Page:", page)
            # print(web.page_source)
        except Exception as e:
            # error_url.append(url)
            print("selenium Cannot Open:", e)
            # self.web = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            if getcoordinates:
                return [], [], [], []
            return [], [], []
        # elems = self.web.find_elements(By.CSS_SELECTOR, '[href^="https"]')
        elems = self.web.find_elements(By.CSS_SELECTOR, 'a')
        # elems = self.web.find_elements(By.XPATH, 'a')
        # print('HTML', len(elems))
        # if len(elems) == 0:
        #     try:
        #         self.web.implicitly_wait(15)
        #         self.web.get(page)
        #         print("Selenium_Page:", page)
        #     except Exception as e:
        #         # error_url.append(url)
        #         print("selenium Cannot Open:", e)
        #         # self.web = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        #         if getcoordinates:
        #             return [], [], [], []
        #         return [], [], []
        hrefs = []
        coordinates = []
        tagPaths = []
        text = []
        # print('html elem len',len(elems))
        for i, elem in enumerate(elems):
            try:
                if (elem.get_attribute('textContent') == ''):
                    continue
                if StringTool.CleanString(elem.get_attribute('textContent')) == '':
                    continue
                if (not self.LeaveThisURL(page, elem.get_attribute("href"))):
                    continue
                href = elem.get_attribute("href")
                if '#' in href:
                    continue
                if (href == ''):
                    continue
                # print('GetCoordinate(elem)',i,GetCoordinate(elem))
                # text.append(StringTool.CleanString(elem.get_attribute('textContent')))
                # hrefs.append(href)
                # tagPaths.append(self.GetTagPath(elem))
                # if getcoordinates:
                #     coordinates.append(self.GetCoordinatebyCSS(elem))
                if getcoordinates:
                    # coordinates.append(self.GetCoordinate(elem))
                    coordinates.append(self.GetCoordinatebyCSS(elem))
                    text.append(StringTool.CleanString(elem.get_attribute('textContent')))
                    hrefs.append(href)
                    tagPaths.append(self.GetTagPath(elem))
                else:
                    text.append(StringTool.CleanString(elem.get_attribute('textContent')))
                    hrefs.append(href)
                    tagPaths.append(self.GetTagPath(elem))

            except Exception as e:
                # print('Exception',e)
                continue
        coordinates = self.NormalizeCoordinate(coordinates)
        # self.webPageClose()
        if not getcoordinates:
            return text, hrefs, tagPaths
        else:
            return text, hrefs, tagPaths, coordinates

    def GetHrefContentbyHTML(self,page, url_idx, types='train',getcoordinates=True):
        try:
            if types == 'train':
                self.web.implicitly_wait(5)
                self.web.get(r"file:///C:/Users/Kuma/PycharmProjects/dqn_google_data/html_rank_train/{}_with_hidden_elements.html".format(
                    url_idx))
            # print(web.page_source)
            else:
                self.web.implicitly_wait(5)
                self.web.get(
                    r"file:///C:/Users/Kuma/PycharmProjects/dqn_google_data/html_rank_vali/{}_with_hidden_elements.html".format(
                        url_idx))
        except Exception as e:
            # error_url.append(url)
            print("Cannot Open:", e)
            if getcoordinates:
                return [], [], [], []
            return [], [], []
        # elems = self.web.find_elements(By.CSS_SELECTOR, '[href^="https"]')
        elems = self.web.find_elements(By.CSS_SELECTOR, 'a')
        # print('HTML', len(elems))
        if len(elems) == 0:
            try:
                if types == 'train':
                    self.web.get(
                        r"file:///C:/Users/Kuma/PycharmProjects/dqn_google_data/html_rank_train/{}_with_hidden_elements.html".format(
                            url_idx))
                # print(web.page_source)
                else:
                    self.web.get(
                        r"file:///C:/Users/Kuma/PycharmProjects/dqn_google_data/html_rank_vali/{}_with_hidden_elements.html".format(
                            url_idx))
            except Exception as e:
                # error_url.append(url)
                print("Cannot Open:", e)
                if getcoordinates:
                    return [], [], [], []
                return [], [], []
        hrefs = []
        coordinates = []
        tagPaths = []
        text = []
        # print('html elem len',len(elems))
        for i, elem in enumerate(elems):
            try:
                if elem.get_attribute('textContent') == '':
                    continue
                if StringTool.CleanString(elem.get_attribute('textContent')) == '':
                    continue
                if not self.LeaveThisURL(page, elem.get_attribute("href")):
                    continue
                href = elem.get_attribute("href")
                if '#' in href:
                    continue
                if (href == ''):
                    continue
                # print('GetCoordinate(elem)',i,GetCoordinate(elem))
                # text.append(StringTool.CleanString(elem.get_attribute('textContent')))
                # hrefs.append(href)
                # tagPaths.append(self.GetTagPath(elem))
                if getcoordinates:
                    # coordinates.append(self.GetCoordinate(elem))
                    coordinates.append(self.GetCoordinatebyCSS(elem))
                    # print('textContent :{}'.format((elem.get_attribute('textContent'))))
                    # print('textContent clean :{}'.format(StringTool.CleanString(elem.get_attribute('textContent'))))

                    text.append(StringTool.CleanString(elem.get_attribute('textContent')))
                    hrefs.append(href)
                    tagPaths.append(self.GetTagPath(elem))
                else:
                    text.append(StringTool.CleanString(elem.get_attribute('textContent')))
                    hrefs.append(href)
                    tagPaths.append(self.GetTagPath(elem))
            except Exception as e:
                # print('Exception',e)
                continue
        coordinates = self.NormalizeCoordinate(coordinates)
        # self.webPageClose()
        if not getcoordinates:
            return text, hrefs, tagPaths
        else:
            return text, hrefs, tagPaths, coordinates

    # def GetHrefsContentAndHrefs(self, page=None, gettagpath=False, tagpathwithouttree=True):
    #     if self.useETLCrawlService:
    #         # return self.GetHrefsContentAndHrefsBybs4(page, gettagpath, tagpathwithouttree)
    #         return self.GetHrefsContentAndHrefsByLocalFile(page, gettagpath, tagpathwithouttree)
    #     else:
    #         return self.GetHrefsContentAndHrefsBySelenium(page, gettagpath, tagpathwithouttree)

    def NormalizeCoordinate(self, coordinates):
        if (len(list(coordinates)) == 0):
            return []
        Xs, Ys, widths, heights = zip(*list(coordinates))
        minx = min(Xs)
        maxx = max(Xs)
        miny = min(Ys)
        maxy = max(Ys)
        if (maxx - minx != 0 and maxy - miny != 0):
            rate = (maxy - miny) / (maxx - minx)
        else:
            return coordinates
        scaler = MinMaxScaler((0, 1))
        newX = scaler.fit_transform(np.array(Xs).reshape((np.array(Xs).shape[0], 1)))[:, 0]
        xscale = scaler.scale_[0]
        scaler = MinMaxScaler((0, rate))
        newY = scaler.fit_transform(np.array(Ys).reshape((np.array(Ys).shape[0], 1)))[:, 0]
        yscale = scaler.scale_[0]  # xscale * rate
        newWidths = [w * xscale for w in widths]
        newHeights = [h * yscale for h in heights]
        coordinate = zip(newX, newY, newWidths, newHeights)
        return coordinate

    def GetCurrentURL(self):
        return self.currentURL

    # def reset(self, homepage, url_idx, url=None, getcoordinates=False):
    def reset(self, homepage, url_idx=None, types='train',getcoordinates=False):
        self.homepage = homepage
        if url_idx is None:
            # self.anchorText, self.hrefs, self.tagPath = self.GetHrefsContentAndHrefs(homepage, getcoordinates=getcoordinates)
            if getcoordinates:
                self.anchorText, self.hrefs, self.tagPath, self.coordinates = self.GetHrefContentbySelemium(homepage,
                                                                                        getcoordinates=getcoordinates)
            else:
                self.anchorText, self.hrefs, self.tagPath = self.GetHrefContentbySelemium(homepage,
                                                                                          getcoordinates=getcoordinates)
        else:
            if getcoordinates:
                self.anchorText, self.hrefs, self.tagPath, self.coordinates = self.GetHrefContentbyHTML(homepage,
                                                                        url_idx=url_idx,types=types,getcoordinates=getcoordinates)
            else:
                self.anchorText, self.hrefs, self.tagPath = self.GetHrefContentbyHTML(homepage,url_idx=url_idx,
                                                                                      types=types,
                                                                                          getcoordinates=getcoordinates)
                # self.anchorText, self.hrefs, self.tagPath = self.GetHrefsContentAndHrefs(homepage,
                #                                                                          getcoordinates=getcoordinates)

        self.homepage = self.currentURL
        if getcoordinates:
            return self._next_observationWithCoordinateandTagPath()
        else:
            return self._next_observationWithTagPath()

    # def GetPositivePage(self):
    #     return self.positivePages

    def GetAnsAndDiscountedReward(self, hrefs, negativeReward=None, positiveReward=None):
        if negativeReward is None:
            negativeReward = -1  # 0.1
        if positiveReward is None:
            positiveReward = 1
        ans = []
        rewards = []
        city = ['taoyuan', 'taipei', 'keelung', 'hsinchu', 'miaoli', 'taichung', 'nantou', 'changhua', 'yunlin',
                'chiayi', 'tainan', 'kaohsiung', 'pingtung', 'yilan', 'hualien', 'taitung', 'penghu', 'kinmen',
                'lienchiang']
        category = ['expo', 'music', 'drama']
        hrefs = URLTool.GetNormalizaUrl(hrefs)
        for index in range(len(hrefs)):
            # print('idx',index)
            res = urlparse(hrefs[index])
            if (hrefs[index] in self.positivePages):
                ans.append(1)
                rewards.append(positiveReward)
            elif res.netloc == 'kktix.com' or res.netloc == 'www.orchina.net' or res.netloc == 'www.orchina.net':
                path = res.path
                split = path.split('/')
                # print(split)
                if split[1] == 'events':
                    ans.append(1)
                    rewards.append(positiveReward)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            elif res.netloc == 'www.accupass.com':
                path = res.path
                split = path.split('/')
                # print(split)
                if split[1] == 'search':
                    ans.append(1)
                    rewards.append(positiveReward)
                else:
                    ans.append(0)
                    rewards.append(negativeReward)
            elif res.netloc == 'event.moc.gov.tw':
                query_split = res.query.split('/')
                if len(query_split) >= 2:
                    if query_split[1][:15] == 'eventSearchList':
                        ans.append(1)
                        rewards.append(positiveReward)
                    else:
                        ans.append(0)
                        rewards.append(negativeReward)
                else:
                    ans.append(0)
                    rewards.append(negativeReward)
            elif res.netloc == 'www.taichung.gov.tw':
                if not res.query == '':
                    if res.query.find('12026'):
                        ans.append(1)
                        rewards.append(positiveReward)
                    else:
                        rewards.append(negativeReward)
                        ans.append(0)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            elif res.netloc == 'yii.tw':
                path = res.path
                split = path.split('/')
                # print(split)
                if split[1] in city:
                    if split[2] == 'calendar':
                        ans.append(1)
                        rewards.append(positiveReward)
                    elif split[3] in category:
                        ans.append(1)
                        rewards.append(positiveReward)
                    else:
                        rewards.append(negativeReward)
                        ans.append(0)
                elif split[1] == 'events' and len(split) == 2:
                    ans.append(1)
                    rewards.append(positiveReward)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            elif res.netloc == 'www.sce.pccu.edu.tw':
                path = res.path
                split = path.split('/')
                if split[1] == 'search':
                    ans.append(1)
                    rewards.append(positiveReward)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            elif res.netloc == 'www.travel.taipei':
                path = res.path
                split = path.split('/')
                if len(split[2]) == 3:
                    if split[2] in ['event-calendar', 'activity', 'news']:
                        ans.append(1)
                        rewards.append(positiveReward)
                    else:
                        rewards.append(negativeReward)
                        ans.append(0)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            elif res.netloc == 'ddmcht.ddm.org.tw':
                path = res.path
                split = path.split('/')
                if split[2] in ['DdmMsgMore.aspx', 'DdmMsgSearch']:
                    ans.append(1)
                    rewards.append(positiveReward)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            elif res.netloc == 'www.chimeimuseum.org':
                path = res.path
                split = path.split('/')
                if split[1] == 'exhibition-event':
                    ans.append(1)
                    rewards.append(positiveReward)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            elif res.netloc == 'www.ncl.edu.tw':
                path = res.path
                split = path.split('/')
                if split[1] == 'calendar_238.html':
                    ans.append(1)
                    rewards.append(positiveReward)
                else:
                    rewards.append(negativeReward)
                    ans.append(0)
            else:
                rewards.append(negativeReward)
                ans.append(0)
        discountedRewards = rewards.copy()
        for i in range(len(rewards) - 1):
            reward = rewards[i]
            for j in range(i + 1, len(rewards)):
                reward += (self.rewardweight ** j) * rewards[j]
            discountedRewards[i] = reward
        return ans,discountedRewards
    def GetAnsAnddReward(self, href,negativeReward=None, positiveReward=None):
        if negativeReward is None:
            negativeReward = -1  # 0.1
        if positiveReward is None:
            positiveReward = 1
        rewards = []
        city = ['taoyuan', 'taipei', 'keelung', 'hsinchu', 'miaoli', 'taichung', 'nantou', 'changhua', 'yunlin',
                'chiayi', 'tainan', 'kaohsiung', 'pingtung', 'yilan', 'hualien', 'taitung', 'penghu', 'kinmen',
                'lienchiang']
        category = ['expo', 'music', 'drama']
        href = URLTool.GetNormalizaUrl([href])
        href = href[0]
        # for index in range(len(hrefs)):
            # print('idx',index)
        res = urlparse(href)
        if href in self.positivePages:
            ans = 1
            reward = positiveReward
        elif res.netloc =='kktix.com' or res.netloc == 'www.orchina.net':
            path = res.path
            split = path.split('/')
            # print(split)
            if split[1] == 'events':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc == 'www.accupass.com' or res.netloc == 'vvg.com.tw':
            path = res.path
            split = path.split('/')
            # print(split)
            if split[1] == 'search' or split[1] == 'news':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc == 'event.moc.gov.tw':
            query_split = res.query.split('/')
            if len(query_split) >= 2:
                if query_split[1][:15] =='eventSearchList':
                    ans = 1
                    reward = positiveReward
                else:
                    ans = 0
                    reward = negativeReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc == 'www.taichung.gov.tw':
            if not res.query == '':
                if res.query.find('12026'):
                    ans = 1
                    reward = positiveReward
                else:
                    ans = 0
                    reward = negativeReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc =='yii.tw':
            path = res.path
            split = path.split('/')
            # print(split)
            if split[1] in city:
                if split[2] == 'calendar':
                    ans = 1
                    reward = positiveReward
                elif split[3] in category:
                    ans = 1
                    reward = positiveReward
                else:
                    ans = 0
                    reward = negativeReward
            elif split[1] == 'events' and len(split) == 2:
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc =='www.sce.pccu.edu.tw':
            path = res.path
            split = path.split('/')
            if split[1] == 'search':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc =='www.chtf.org.tw':
            path = res.path
            split = path.split('/')
            if split[2] == 'event':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc =='www.taih.ntnu.edu.tw':
            path = res.path
            split = path.split('/')
            if split[3] == 'uncategorized':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc =='www.tpqc.com.tw':
            path = res.path
            split = path.split('/')
            if split[1] == '%E6%89%80%E6%9C%89%E8%AA%B2%E7%A8%8B':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc =='umkt.jutfoundation.org.tw':
            path = res.path
            split = path.split('/')
            if split[2] == '58':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc == 'www.travel.taipei':
            path = res.path
            split = path.split('/')
            if len(split)==3:
                if split[2] in ['event-calendar', 'activity', 'news']:
                    ans = 1
                    reward = positiveReward
                else:
                    ans = 0
                    reward = negativeReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc == 'ddmcht.ddm.org.tw':
            path = res.path
            split = path.split('/')
            if split[2] in ['DdmMsgMore.aspx', 'DdmMsgSearch']:
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc == 'www.chimeimuseum.org':
            path = res.path
            split = path.split('/')
            if split[1] == 'exhibition-event':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        elif res.netloc == 'www.ncl.edu.tw':
            path = res.path
            split = path.split('/')
            if split[1] == 'calendar_238.html':
                ans = 1
                reward = positiveReward
            else:
                ans = 0
                reward = negativeReward
        else:
            ans = 0
            reward = negativeReward

        return ans, reward
    def GetPositivePage(self, href):
        city = ['taoyuan', 'taipei', 'keelung', 'hsinchu', 'miaoli', 'taichung', 'nantou', 'changhua', 'yunlin',
                'chiayi', 'tainan', 'kaohsiung', 'pingtung', 'yilan', 'hualien', 'taitung', 'penghu', 'kinmen',
                'lienchiang']
        category = ['expo', 'music', 'drama']
        href = URLTool.GetNormalizaUrl([href])
        href = href[0]
        # for index in range(len(hrefs)):
            # print('idx',index)
        res = urlparse(href)
        if href in self.positivePages:
            ans = 1
        elif res.netloc =='kktix.com' or res.netloc == 'www.orchina.net':
            path = res.path
            split = path.split('/')
            # print(split)
            if split[1] == '':
                ans = 1
                self.positivePages.append(href)
            else:
                ans = 0
        elif res.netloc == 'www.accupass.com':
            path = res.path
            split = path.split('/')
            # print(split)
            if split[1] == 'search':
                ans = 1
                self.positivePages.append(href)
            else:
                ans = 0
        elif res.netloc == 'event.moc.gov.tw':
            query_split = res.query.split('/')
            if len(query_split) >= 2:
                if query_split[1][:15] =='eventSearchList':
                    ans = 1
                    self.positivePages.append(href)
                else:
                    ans = 0
            else:
                ans = 0
        elif res.netloc == 'www.taichung.gov.tw':
            if not res.query == '':
                if res.query.find('12026'):
                    ans = 1
                    self.positivePages.append(href)
                else:
                    ans = 0
            else:
                ans = 0
        elif res.netloc =='yii.tw':
            path = res.path
            split = path.split('/')
            # print(split)
            if split[1] in city:
                if split[2] == 'calendar':
                    ans = 1
                    self.positivePages.append(href)
                elif split[3] in category:
                    ans = 1
                    self.positivePages.append(href)
                else:
                    ans = 0
            elif split[1] == 'events' and len(split) == 2:
                ans = 1
                self.positivePages.append(href)
            else:
                ans = 0
        elif res.netloc =='www.sce.pccu.edu.tw':
            path = res.path
            split = path.split('/')
            if split[1] == 'search':
                ans = 1
                self.positivePages.append(href)
            else:
                ans = 0
        elif res.netloc == 'www.travel.taipei':
            path = res.path
            split = path.split('/')
            if len(split)==3:
                if split[2] in ['event-calendar', 'activity', 'news']:
                    ans = 1
                    self.positivePages.append(href)
                else:
                    ans = 0
            else:
                ans = 0
        elif res.netloc == 'ddmcht.ddm.org.tw':
            path = res.path
            split = path.split('/')
            if split[2] in ['DdmMsgMore.aspx', 'DdmMsgSearch']:
                ans = 1
                self.positivePages.append(href)
            else:
                ans = 0
        elif res.netloc == 'www.chimeimuseum.org':
            path = res.path
            split = path.split('/')
            if split[1] == 'exhibition-event':
                ans = 1
                self.positivePages.append(href)
            else:
                ans = 0
        elif res.netloc == 'www.ncl.edu.tw':
            path = res.path
            split = path.split('/')
            if split[1] == 'calendar_238.html':
                ans = 1
                self.positivePages.append(href)
            else:
                ans = 0
        else:
            ans = 0

        return self.positivePages,ans
    def GetPageANS(self, pre_hrefs):
        city = ['taoyuan', 'taipei', 'keelung', 'hsinchu', 'miaoli', 'taichung', 'nantou', 'changhua', 'yunlin',
                'chiayi', 'tainan', 'kaohsiung', 'pingtung', 'yilan', 'hualien', 'taitung', 'penghu', 'kinmen',
                'lienchiang']
        category = ['expo', 'music', 'drama']
        # href = URLTool.GetNormalizaUrl([href])
        # href = href[0]
        ans = []
        hrefs = URLTool.GetNormalizaUrl(pre_hrefs)
        for index in range(len(hrefs)):
            # print('idx',index)
            res = urlparse(hrefs[index])
        # res = urlparse(href)
            if hrefs[index] in self.positivePages:
                ans.append(1)
            elif res.netloc =='kktix.com' or res.netloc == 'www.orchina.net':
                path = res.path
                split = path.split('/')
                # print(split)
                if split[1] == '':
                    ans.append(1)
                    self.newpositivePages.append(pre_hrefs[index])
                else:
                    ans.append(0)
            elif res.netloc == 'www.accupass.com':
                path = res.path
                split = path.split('/')
                # print(split)
                if split[1] == 'search':
                    ans.append(1)
                    self.newpositivePages.append(pre_hrefs[index])
                else:
                    ans.append(0)
            elif res.netloc == 'event.moc.gov.tw':
                query_split = res.query.split('/')
                if len(query_split) >= 2:
                    if query_split[1][:15] =='eventSearchList':
                        ans.append(1)
                        self.newpositivePages.append(pre_hrefs[index])
                    else:
                        ans.append(0)
                else:
                    ans.append(0)
            elif res.netloc == 'www.taichung.gov.tw':
                if not res.query == '':
                    if res.query.find('12026'):
                        ans.append(1)
                        self.newpositivePages.append(pre_hrefs[index])
                    else:
                        ans.append(0)
                else:
                    ans.append(0)
            elif res.netloc =='yii.tw':
                path = res.path
                split = path.split('/')
                # print(split)
                if split[1] in city:
                    if split[2] == 'calendar':
                        ans.append(1)
                        self.newpositivePages.append(pre_hrefs[index])
                    elif split[3] in category:
                        ans.append(1)
                        self.newpositivePages.append(pre_hrefs[index])
                    else:
                        ans.append(0)
                elif split[1] == 'events' and len(split) == 2:
                    ans.append(1)
                    self.newpositivePages.append(pre_hrefs[index])
                else:
                    ans.append(0)
            elif res.netloc =='www.sce.pccu.edu.tw':
                path = res.path
                split = path.split('/')
                if split[1] == 'search':
                    ans.append(1)
                    self.newpositivePages.append(pre_hrefs[index])
                else:
                    ans.append(0)
            elif res.netloc == 'www.travel.taipei':
                path = res.path
                split = path.split('/')
                if len(split)==3:
                    if split[2] in ['event-calendar', 'activity', 'news']:
                        ans.append(1)
                        self.newpositivePages.append(pre_hrefs[index])
                    else:
                        ans.append(0)
                else:
                    ans.append(0)
            elif res.netloc == 'ddmcht.ddm.org.tw':
                path = res.path
                split = path.split('/')
                if split[2] in ['DdmMsgMore.aspx', 'DdmMsgSearch']:
                    ans.append(1)
                    self.newpositivePages.append(pre_hrefs[index])
                else:
                    ans.append(0)
            elif res.netloc == 'www.chimeimuseum.org':
                path = res.path
                split = path.split('/')
                if split[1] == 'exhibition-event':
                    ans.append(1)
                    self.newpositivePages.append(pre_hrefs[index])
                else:
                    ans.append(0)
            elif res.netloc == 'www.ncl.edu.tw':
                path = res.path
                split = path.split('/')
                if split[1] == 'calendar_238.html':
                    ans.append(1)
                    self.newpositivePages.append(pre_hrefs[index])
                else:
                    ans.append(0)
            else:
                ans.append(0)

        return self.newpositivePages,ans
    def GetAns(self, hrefs):
        ans = []
        hrefs = URLTool.GetNormalizaUrl(hrefs)
        for index in range(len(hrefs)):
            if hrefs[index] == '':
                continue
            if (hrefs[index][-1] != '/'):
                hrefs[index] += '/'
            if ("https" == hrefs[index][0:5]):
                hrefs[index] = "http" + hrefs[index][5:]
            if (hrefs[index] in self.positivePages):
                ans.append(1)
            else:
                ans.append(0)
        return ans

    def _next_observationWithCoordinateandTagPath(self):
        # return self.anchorText, self.hrefs, self.tagPath
        return self.anchorText, self.hrefs, self.tagPath, self.coordinates

    def _next_observationWithTagPath(self):
        return self.anchorText, self.hrefs, self.tagPath
        # return self.anchorText, self.hrefs, self.coordinates

    def step(self, href, getcoordinates=False):
        # nowscheme = urlparse(self.currentURL).scheme
        # stepparse = urlparse(href)  # .scheme
        # if (stepparse.query != ''):
        #     stephref = nowscheme + "://" + stepparse.netloc + stepparse.path + stepparse.params + '?' + stepparse.query
        # else:
        #     stephref = nowscheme + "://" + stepparse.netloc + stepparse.path + stepparse.params
        # self._take_action(href, getcoordinates=getcoordinates)
        # if getcoordinates:
        #     return self._next_observationWithCoordinateandTagPath()
        # else:
        #     return self._next_observationWithTagPath()
        self.currentURL = href
        if getcoordinates:
            self.anchorText, self.hrefs, self.tagPath, self.coordinates = self.GetHrefContentbySelemium(href,
                                                                                        getcoordinates=getcoordinates)
            return self.anchorText, self.hrefs, self.tagPath, self.coordinates
        else:
            self.anchorText, self.hrefs, self.tagPath = self.GetHrefsContentAndHrefs(href,
                                                                                     getcoordinates=getcoordinates)
            return self.anchorText, self.hrefs, self.tagPath

    def _take_action(self, href, getcoordinates=False):
        self.currentURL = href
        if getcoordinates:
            self.anchorText, self.hrefs, self.tagPath,self.coordinates = self.GetHrefContentbySelemium(href,getcoordinates=getcoordinates)
        else:
            self.anchorText, self.hrefs, self.tagPath = self.GetHrefsContentAndHrefs(href,getcoordinates=getcoordinates)
    def GetPageTitle(self):
        return self.web.title

    def GetAnchorText(self):
        return self.anchorText

    def webQuit(self):
        self.web.quit()
    #     # self.web = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    def webPageClose(self):
        self.web.close()
        # self.web = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)


if __name__ == '__main__':
    #print(URLTool.IsUrlString("www.ymsnp.gov.tw/main_ch/docList.aspx?uid=1583&pid=18&rn=-13693"))
    import pandas as pd
    traindf = pd.read_csv("../dataset/train_0.8.csv")
    validation = pd.read_csv("../dataset/vali_0.2.csv")
    testdf = pd.read_csv("../dataset/newdata_test.csv")
    totalDF = pd.concat([traindf,validation,testdf])
    trainwebsiteInfo = traindf.groupby("Event Message Page URL")
    validationwebsiteInfo = validation.groupby("Event Message Page URL")
    testwebsiteInfo = testdf.groupby("Event Message Page URL")
    totalwebsiteInfo = totalDF.groupby("Event Message Page URL")
    a = Env(testdf, negativeReward=-0.1)
    #a.GetHTMLFile("http://www.academyofastrology.co.uk/")
    b,c,d,e = a.GetHrefsContentAndHrefs("http://www.hondao.org.tw/article?category=%E5%87%BA%E7%89%88%E5%93%81",gettagpath = True)
    # print(b)
    # a.reset("https://chromedriver.chromium.org/downloads")





