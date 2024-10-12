import pandas as pd
import os
from src import Tool,PolicyDeepthEnvFixCoordinate_singlefile
from datetime import datetime
import threading
import time
from multiprocessing import Queue
from func_timeout import func_timeout, FunctionTimedOut
timamap = {}

class savehref():
    def __init__(self):
        self.Hrefs = []
    def SaveData(self, Hrefs):
        self.Hrefs.append(Hrefs)
    def clear(self):
        self.Hrefs = []

def url_to_feature(url,url_idx,env,types):
    for i in range(1):
        try:
            if url in url_featuresMap:
                print('url in url featureMap:', url)
                AT, HREFS, tagpaths = url_featuresMap[url]['attext'], url_featuresMap[url]['hrefs'], \
                                                  url_featuresMap[url]['tagpaths']
            else:
                if types == 'test' or types == 'finetune':
                    AT, HREFS, tagpaths = func_timeout(180, env.reset, args=(url, None, types, False))

                else:
                    AT, HREFS, tagpaths = func_timeout(240, env.reset, args=(url, url_idx, types, False))

                # print(len(AT),len(HREFS),len(tagpaths),len(coordinate))
                url_featuresMap[url] = {'attext': AT, 'hrefs': HREFS, 'tagpaths': tagpaths,
                                         'time': str(datetime.now())}

        except FunctionTimedOut:
            print('this url cannot get:{}'.format(url))
            break
            # return None, None
        if len(AT) == 0 or len(AT) == 1:
            print('AT len is 0 or 1, url: {} '.format(url))
            break
        global_lock.acquire()
        global_save.SaveData(HREFS)
        global_lock.release()

class Worker(threading.Thread):
    def __init__(self, queue, env, url_idx_queue, types):
        threading.Thread.__init__(self)
        self.queue = queue
        self.env = env
        self.types = types
        self.url_idx_queue = url_idx_queue

    def run(self):
        global timamap
        while self.queue.qsize() > 0:
            print('size:{}'.format(self.queue.qsize()))
            url = self.queue.get()
            url_idx = self.url_idx_queue.get()
            ## collect episode
            print('url:{} ,idx:{}'.format(url, url_idx))
            start_time = datetime.now()
            url_to_feature(url, url_idx, self.env, self.types)
            finish_time = datetime.now()
            timamap[url] = (finish_time - start_time).total_seconds()

def UseThreadRunData(datalist, envs, url_idx_list,types):
    # global workers
    workers = []
    my_queue = Queue()
    queue_idx = Queue()
    for (homepage,homepage_idx) in zip(list(datalist), url_idx_list):
        my_queue.put(homepage)
        queue_idx.put(homepage_idx)
    for i in range(len(envs)):
        w = Worker(my_queue, envs[i], queue_idx, types)
        workers.append(w)
    for i in range(len(envs)):
        workers[i].start()
        time.sleep(2)
    for i in range(len(envs)):
        workers[i].join()
if __name__ =='__main__':
    config = Tool.LoadConfig('config.ini')
    if os.path.isfile(config.get('Main', 'url_feature_file_path_no_coor') + '.txt'):
        url_featuresMap = Tool.LoadObj(config.get('Main', 'url_feature_file_path_no_coor'))
    else:
        url_featuresMap = {}
    global_lock = threading.Lock()
    global_save = savehref()
    traindf = pd.read_csv("./dataset_google_search/google_search_rank_train - contain news.csv")
    validation = pd.read_csv("./dataset_google_search/google_search_rank_vali - contain_news.csv")
    testdf = pd.read_csv("./dataset_google_search/newdata_test - remove error url.csv")
    pre_totalDF = pd.concat([traindf, validation,testdf])
    totalDF = pre_totalDF.dropna(subset=['Event source page'])
    trainwebsiteInfo = traindf.groupby("Homepage")
    validationwebsiteInfo = validation.groupby("Homepage")
    testwebsiteInfo = testdf.groupby("Homepage")
    trainlist = list(trainwebsiteInfo.groups)[0:40]
    valilist = list(validationwebsiteInfo.groups)[0:20]
    testlist = list(testwebsiteInfo.groups)[0:96]
    # testlist = list(testwebsiteInfo.groups)[0:20]
    # print('testlist',testlist)
    fintunedf = pd.read_csv("./dataset_google_search/fintunetraindata.csv")
    fintunelist = list(fintunedf['url'])
    numofthread = 3
    envs = []
    for i in range(numofthread):
        envs.append(PolicyDeepthEnvFixCoordinate_singlefile.Env(totalDF, negativeReward=-0.1, rewardweight=0.1))

    # # UseThreadRunData(trainlist, envs, list(range(1, 41)), 'train')
    # # data = {'href':global_save.Hrefs}
    file_name = 'ESPS_dataset'
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    data = {'href': global_save.Hrefs}
    Tool.SaveObj('{}/train_url'.format(file_name),data)
    global_save.clear()

    UseThreadRunData(valilist, envs, list(range(1, 21)), 'validation')
    data = {'href': global_save.Hrefs}
    Tool.SaveObj('{}/vali_url'.format(file_name), data)
    global_save.clear()

    UseThreadRunData(testlist, envs, list(range(1, 96)), 'test')
    data = {'href': global_save.Hrefs}
    Tool.SaveObj('{}/test_url_95'.format(file_name), data)
    global_save.clear()

    UseThreadRunData(fintunelist, envs, list(range(1, 301)), 'finetune')
    data = {'href': global_save.Hrefs}
    Tool.SaveObj('{}/finetune_url'.format(file_name), data)
    global_save.clear()
    Tool.SaveObj(config.get('Main', 'url_feature_file_path_no_coor'), url_featuresMap)
    for env_thread in envs:
        env_thread.webQuit()



