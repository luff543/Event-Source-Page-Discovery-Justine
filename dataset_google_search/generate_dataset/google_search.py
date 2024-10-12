import os.path
import re
import time
import pandas as pd
import csv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Browser settings
chrome_options = Options()
chrome_options.add_argument('--incognito')  # 無痕
chrome_options.add_argument(
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36')
browser = webdriver.Chrome(ChromeDriverManager().install(),chrome_options=chrome_options)


# Query settings
# querys = ['活動','展覽','講座']
# querys = ['活動講座']
querys = ['最新活動']
# query = '展覽'
for query in querys:
    print('query',query)
    browser.get('https://www.google.com/search?q={}'.format(query))
    # next_page_times = 30
    next_page_times = 20


    # Crawler
    count = 0
    for _page in range(next_page_times):
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        # content = soup.prettify()
        webdiv = soup.find_all('div','MjjYud')
        # Get titles and urls

        # titles = re.findall('<h3 class="[\w\d]{6} [\w\d]{6}">\n\ +(.+)', content)
        # urls = re.findall('<div class="r">\ *\n\ *<a href="(.+)" onmousedown', soup.prettify())

        # for n in range(min(len(titles), len(urls))):
        #     print(titles[n], urls[n])
        domain_pages =[]
        urls = []
        titles = []
        rank = []
        for page in webdiv:
            # if page.find('h3','LC20lb MBeuO DKV0Md') and page.find('a')['href'] and page.find('cite',"iUh30 qLRx3b tjvcx"):
            if page.find('h3', 'LC20lb MBeuO DKV0Md') and page.find('a')['href'] and page.find('cite',
                                                                                             "apx8Vc qLRx3b tjvcx GvPZzd cHaqb"):
                title = page.find('h3','LC20lb MBeuO DKV0Md').text
                # print('title',title)
                url = page.find('a')['href']
                # print('url',url)
                # page = page.find('cite', "iUh30 qLRx3b tjvcx")
                page = page.find('cite', "apx8Vc qLRx3b tjvcx GvPZzd cHaqb")
                domain_page = page.text.split('›')[0]
                # print('domain_page',domain_page)
                if url not in urls:
                    count += 1
                    rank.append(count)
                    urls.append(url)
                    titles.append(title)
                    domain_pages.append(domain_page)

        data = {
                'keyword':[query]*len(urls),
                'rank':rank,
                'domain_page':domain_pages,
                'url':urls,
                'title':titles,
                }
        # print('data',data)
        df = pd.DataFrame(data)
        if _page == 0:
            df.to_csv('./dataset_google_search/generate_dataset/eventsource_{}.csv'.format(query),encoding='utf_8_sig',mode='a',index=False,header=True)
        else:
            df.to_csv('./dataset_google_search/generate_dataset/eventsource_{}.csv'.format(query), encoding='utf_8_sig',mode='a', index=False,header=False)

        # Wait
        time.sleep(5)

        # Turn to the next page
        try:
            browser.find_element("link text","下一頁").click()
        except Exception as e:
            print('Search Early Stopping.',e)
            print('page', _page)
            browser.quit()
            browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chrome_options)
            break

# Close the browser
browser.quit()