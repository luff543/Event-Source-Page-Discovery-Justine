#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import ssl
import matplotlib.pyplot as plt
import matplotlib.patches as pc
import re
from url_normalize import url_normalize
from urllib.parse import urljoin, urlparse
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import gridspec
WINDOW_SIZE = "1920,1080"
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("window-size={}".format(WINDOW_SIZE))
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
ssl._create_default_https_context = ssl._create_unverified_context


# In[2]:


class Coordinate():
    # def __init__(self, chromepath):
    #     self.web = webdriver.Chrome(chromepath, options=chrome_options)
    # def __init__(self):

    # def __del__(self):
    #     self.web.close()
    def IsUrlString(self, string): 
        regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        url = re.findall(regex,string)
        urls = [x[0] for x in url]
        if(len(urls)==0):
            return False
        return True
    def LeaveThisURL(self, page,href):
        if (not self.IsUrlString(href)):
            href = urljoin(page, href)
            if(not self.IsUrlString(href)):
                return False
        return True
    def GetNormalizaUrl(self, page, href):
        try:
            tmp = urlparse(url_normalize(href))
        except:
            return ""
        if(tmp.query!=''):
            path = tmp.scheme + "://" + tmp.netloc + tmp.path + tmp.params + '?' + tmp.query
        else:
            path = tmp.scheme  + "://" + tmp.netloc + tmp.path + tmp.params
        href = url_normalize(path)
        if("https" in href):
            href = "http"+href[5:]
        if("https" in page):
            page = "http"+page[5:]
        if(href == (page+'/') or (href+'/') == page or href == page):
            return ""
        return href
    def GetCoordinate(self, element):
        while(element.size['width']==0):
            element = element.find_element_by_xpath('..')
        x = element.location['x']
        y = element.location['y']
        width = element.size['width']
        height = element.size['height']
        return x,y,width,height
        #return zip(*set(zip(Xs,Ys,widths,heights)))
    def CleanString(self, string):
        cleantext = string.replace('\n','')
        cleantext = cleantext.replace('\t','')
        cleantext = cleantext.replace('\r','')
        cleantext = cleantext.replace(' ','')
        return cleantext
    def GetHrefAndContent(self, page):
        locations = []
        sizes = []
        coordinate = []
        cannotOpen = False
        try:
            self.web.implicitly_wait(20)
            self.web.get(page)
            html = self.web.page_source
            currentURL = self.web.current_url
            page = self.web.current_url
        except Exception as e:
            print("Cannot Open:",e)
            cannotOpen = True
        elems = self.web.find_elements_by_xpath("//*[@href]")
        num = 0
        text = []
        hrefs = []
        for elem in elems:
            if(elem.get_attribute('textContent')==''):
                continue
            if(not self.LeaveThisURL(page,elem.get_attribute("href"))):
                continue
            href = elem.get_attribute("href")
            href = self.GetNormalizaUrl(page, href)
            if(href==''):
                continue
            text.append(self.CleanString(elem.get_attribute('textContent')))
            hrefs.append(href)
            coordinate.append(self.GetCoordinate(elem))
            
            #print(text[-1],CleanString(elem.get_attribute('textContent')),hrefs[-1],coordinate[-1][0],coordinate[-1][1],coordinate[-1][2],coordinate[-1][3])
        # elems = self.web.find_elements_by_xpath("//*[@src]")
        # for elem in elems:
        #     try:
        #         if(elem.get_attribute('alt')==''):
        #             continue
        #         if(not self.LeaveThisURL(page,elem.find_element(By.XPATH, '..').get_attribute('href'))):
        #             continue
        #         href = elem.find_element(By.XPATH, '..').get_attribute('href')
        #         href = self.GetNormalizaUrl(page, href)
        #         if(href==''):
        #             continue
        #         t = elem.get_attribute('alt')
        #         text.append(self.CleanString(t))
        #         hrefs.append(href)
        #         while(elem.size['width']==0):
        #             elem = elem.find_element_by_xpath('..')
        #         coordinate.append(self.GetCoordinate(elem))
        #     except:
        #         continue
        title = ""
        try:
            title = self.web.title
        except:
            pass
        return hrefs, text, coordinate,title#Xs, Ys, widths, heights
    def NormalizeCoordinate(self, Xs,Ys,widths,heights):
        minx = min(Xs)
        maxx = max(Xs)
        miny = min(Ys)
        maxy = max(Ys)
        #try:
        if (maxx-minx) != 0:
            rate = (maxy-miny)/(maxx-minx)
        else:
            rate = maxy-miny
        if rate == 0.0:
            rate = 1
        scaler = MinMaxScaler((0,1))
        newX = scaler.fit_transform(np.array(Xs).reshape((np.array(Xs).shape[0],1)))[:,0]
        xscale = scaler.scale_[0]
        scaler = MinMaxScaler((0,rate))
        newY = scaler.fit_transform(np.array(Ys).reshape((np.array(Ys).shape[0],1)))[:,0]
        yscale = scaler.scale_[0]#xscale * rate
        newWidths = [w*xscale for w in widths]
        newHeights = [h*yscale for h in heights]
        #except:
        #    return Xs,Ys,widths,heights
        return newX, newY, newWidths, newHeights
    
    def PlotMiddlePoint(self, ax,Xs,Ys,widths,heights):
        xs = []
        ys = []
        for i,j in zip(Xs,widths):
            xs.append(i+(j//2))
        for i,j in zip(Ys,heights):
            ys.append(i+(j//2))
        ax.plot(xs, ys, 'o', color='r',markersize=1)
        return ax

    def PlotRectangle(self, ax, Xs,Ys,widths,heights, annotate = False, texts=None):
        if texts is None:
            texts = [0] * len(Xs)
        for tt,x,y,w,h in zip(texts,Xs,Ys,widths,heights):
            #print(x,y)
            ax.add_patch(
                 pc.Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor = 'blue',
                    facecolor = 'red',
                    fill=False      
                 ) )
            if annotate:
                ax.annotate(r, (cx, cy), color='w', weight='bold', 
                            fontsize=6, ha='center', va='center')
        return ax

    def GetImageArray(self, X,Y,W,H):
        # get_ipython().run_line_magic('matplotlib', 'inline')
        # plt.rcParams['font.sans-serif'] = ['SimHei'] # 步驟一（替換sans-serif字型）
        # plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）
        #plt.rcParams['figure.figsize'] = 480, 640
        #plt.figure(figsize=(480,640))
        #plt.gca().invert_yaxis()
        #fig = plt.figure(figsize=(480,640)) 
        #gs = gridspec.GridSpec(1, 1) 
        fig, ax = plt.subplots()
        ax.invert_yaxis()
        ax = self.PlotMiddlePoint(ax,X,Y,W,H)
        ax = self.PlotRectangle(ax, X,Y,W,H)
        """self.PlotMiddlePoint(plt,X,Y,W,H)
        self.PlotRectangle(plt, X,Y,W,H)"""
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        #plt.savefig("tmp.jpg")
        return data


# In[4]:


if __name__ == '__main__':
    # coo = Coordinate('../chromedriver')
    # hrefs, text, coordinate, title = coo.GetHrefAndContent('https://www.geeksforgeeks.org/numpy-take-python/')
    # Xs,Ys,widths,heights = zip(*list(coordinate))
    # newX, newY, newWidths, newHeights = coo.NormalizeCoordinate(Xs,Ys,widths,heights)
    # data = coo.GetImageArray(newX,newY,newWidths,newHeights)
    # print(data.shape)
    coo = Coordinate()
    # coordinate = [(242, 735, 190, 38), (626, 715, 96, 80)]
    coordinate = [(242, 735, 190, 38), (626, 715, 96, 80), (723, 715, 96, 80), (821, 715, 84, 80), (906, 715, 113, 80), (1021, 715, 84, 80), (1107, 715, 96, 80), (1205, 715, 84, 80), (1574, 718, 97, 76), (-667, 1328, 74, 22), (-575, 1330, 58, 21), (-501, 1330, 44, 21), (-442, 1330, 31, 21), (-667, 1359, 735, 74), (-667, 1363, 74, 22), (-575, 1365, 58, 21), (-501, 1365, 58, 21), (-667, 1394, 735, 44), (1554, 1090, 88, 32), (1212, 1417, 74, 22), (1290, 1417, 74, 22), (1369, 1419, 44, 21), (1419, 1419, 58, 21), (1483, 1419, 58, 21), (1212, 1452, 430, 57), (240, 1856, 74, 22), (318, 1858, 58, 21), (240, 1890, 430, 57), (726, 1856, 74, 22), (804, 1858, 44, 21), (854, 1858, 31, 21), (891, 1858, 31, 21), (928, 1858, 58, 21), (726, 1890, 430, 57), (1212, 1856, 74, 22), (1290, 1858, 58, 21), (1354, 1858, 44, 21), (1404, 1858, 31, 21), (1441, 1858, 58, 21), (1212, 1890, 430, 57), (240, 2274, 74, 22), (318, 2276, 31, 21), (355, 2276, 58, 21), (240, 2309, 430, 57), (726, 2274, 124, 22), (726, 2309, 430, 57), (1212, 2274, 74, 22), (1290, 2274, 74, 22), (1369, 2276, 31, 21), (1406, 2276, 58, 21), (1469, 2276, 31, 21), (1506, 2276, 31, 21), (1543, 2276, 58, 21), (1212, 2309, 430, 57), (240, 2714, 114, 22), (240, 2748, 430, 29), (726, 2714, 74, 22), (804, 2716, 58, 21), (868, 2716, 58, 21), (726, 2748, 430, 57), (1212, 2714, 74, 22), (1290, 2716, 58, 21), (1354, 2716, 44, 21), (1404, 2716, 31, 21), (1212, 2748, 430, 57), (802, 2937, 300, 50), (1558, 3489, 109, 36), (240, 3843, 74, 22), (240, 3878, 430, 29), (726, 3843, 114, 22), (726, 3878, 430, 29), (1212, 3843, 114, 22), (1212, 3878, 430, 29), (240, 4254, 139, 22), (240, 4289, 430, 29), (726, 4254, 114, 22), (726, 4289, 430, 29), (1212, 4254, 87, 22), (1212, 4289, 430, 29), (1558, 4892, 109, 36), (240, 5501, 74, 22), (240, 5536, 916, 29), (1446, 5001, 74, 22), (1446, 5035, 196, 72), (1446, 5153, 74, 22), (1446, 5188, 196, 72), (1446, 5306, 74, 22), (1446, 5340, 196, 48), (1446, 5451, 74, 22), (1446, 5486, 196, 48), (1558, 5705, 109, 36), (236, 5808, 700, 280), (966, 5808, 700, 280), (236, 6118, 700, 280), (966, 6118, 700, 280), (630, 6658, 86, 17), (732, 6658, 101, 17), (849, 6658, 58, 17), (923, 6658, 58, 17), (996, 6658, 86, 17), (1099, 6658, 86, 17), (1201, 6658, 72, 17), (596, 6699, 141, 46), (747, 6699, 136, 46), (893, 6699, 141, 46), (1044, 6699, 155, 46), (1209, 6699, 98, 46), (471, 6797, 86, 17), (561, 6797, 58, 17)]
    Xs, Ys, widths, heights = zip(*list(coordinate))
    newX, newY, newWidths, newHeights = coo.NormalizeCoordinate(Xs, Ys, widths, heights)
    data = coo.GetImageArray(newX, newY, newWidths, newHeights)
    print(data)
    print(data.shape)


# In[ ]:


# 288 432

