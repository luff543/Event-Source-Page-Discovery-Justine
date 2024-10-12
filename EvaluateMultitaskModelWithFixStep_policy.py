#!/usr/bin/env python
# coding: utf-8

import keras.utils.generic_utils
from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import threading
import copy
from src import PolicyDeepthEnvFixCoordinate_singlefile
from Preprocess.Tokenize import Tokenize
from multiprocessing import Queue
from sklearn.metrics import precision_score
import os
from src import Args
try:
    from bert.tokenization.bert_tokenization import FullTokenizer
except:
    from bert.tokenization import FullTokenizer
from src import Tool
from src import StringTool
from datetime import datetime, timedelta
# import keras_nlp
config = Tool.LoadConfig('config.ini')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def add_arg():
    argstool = Args.ArgparseTool()
    argstool.NeedText('--modelname', '-mn', 'your model name', 'modelname')
    # argstool.NeedText('--dataset', '-d', 'your dataset', 'dataset', required=False, default='testdata')
    argstool.NeedText('--steps', '-s', 'your steps', 'steps', required=False, default='1')
    argstool.NeedText('--numofthread', '-n', 'numofthread', 'numofthread', required=False, default=4)
    arg = argstool.GetArgs()
    modelname = arg.modelname
    # dataset = arg.dataset
    steps = int(arg.steps)
    numofthread = int(arg.numofthread)
    # assert dataset == 'testdata' or dataset == 'mixtestdata' or dataset == 'both'
    # return modelname, dataset,steps,numofthread
    return modelname,steps,numofthread

def GetLeaveTime():
    now_time = datetime.now()
    leave_time = now_time - timedelta(days=7)
    return leave_time
def delOldData():
    if os.path.isfile(config.get('Main', 'url_feature_file_path_no_coor')+'.txt'):
        url_featuresMap = Tool.LoadObj(config.get('Main', 'url_feature_file_path_no_coor'))
    else:
        url_featuresMap = {}
    leavetime = GetLeaveTime()
    delkeys = []
    for key,value in url_featuresMap.items():
        try:
            if datetime.strptime(value['time'], "%Y-%m-%d %H:%M:%S.%f") < leavetime:
                delkeys.append(key)
        except:
            delkeys.append(key)
    for key in delkeys:
        del url_featuresMap[key]
    return url_featuresMap

class GlobalMetric():
    def __init__(self):
        self.metricAns = []
        self.clickHrefs = []
        self.clickATs = []
        self.clickHomepages = []
        self.clickValue = []

    def SaveData(self, anses, hrefs, ats, homepage, clickValue):
        # for i in range(len(anses) - deepth):
        #    anses.append(-1)
        self.metricAns.append(anses)
        self.clickHrefs.append(hrefs)
        self.clickATs.append(ats)
        self.clickHomepages.append(homepage)
        self.clickValue.append(clickValue)


def GetTokenTagPath(path):
    ids = [tokenize.GetIDs(tmp) for tmp in path]
    tagoutput = np.zeros((len(ids), tokenize.GetDictLength()))
    for i in range(len(ids)):
        tagoutput[i] = np.bincount(ids[i],
                                   minlength=tokenize.GetDictLength())  # pad_sequences([np.bincount(ids[i],minlength=tokenize.GetDictLength())],maxlen = tokenize.GetDictLength(), dtype=np.int32,padding='post',truncating='post',value=0)[0]
    return tagoutput


def ProcessFeatures(at, coordinate, tagpath):
    coo = GetCoordinateNeighbor(coordinate)
    tagid = GetTokenTagPath(tagpath)
    tagid = tf.repeat(np.array([tagid]), len(at), axis=0)
    # print('len:{}'.format(len(at)))
    return StringTool.GetTextListEmbedding(at, bertlayer, config), list(coo), list(tagid.numpy()), GetIndexMetrix(
        tagid.shape)
def ProcessFeatures_nocoor(at, tagpath):
    tagid = GetTokenTagPath(tagpath)
    tagid = tf.repeat(np.array([tagid]),len(at),axis=0)
    return StringTool.GetTextListEmbedding(at,bertlayer,config), list(tagid.numpy()), GetIndexMetrix(tagid.shape)

def GetIndexMetrix(tagshape):
    # shape = np.array(tagid).shape
    shape = (tagshape[0], tagshape[1], 36)
    index = np.zeros(shape)
    # print('shape:{}'.format(shape))
    for i in range(shape[0]):
        index[i][i] = 1
    return index


def GetCoordinateNeighbor(coordinate):
    neighborCoordinate = []
    for nowIndex in range(len(coordinate)):
        tmpInfor = []  # list(coordinate[nowIndex])
        tmpInfor.extend(list(coordinate[nowIndex]))
        assert len(tmpInfor) == 4
        previousIndex = nowIndex - 1
        nextIndex = nowIndex + 1
        if (previousIndex >= 0):
            for previous, now in zip(coordinate[previousIndex], coordinate[nowIndex]):
                tmpInfor.append(now - previous)
        else:
            for j in range(len(coordinate[nowIndex])):
                tmpInfor.append(0)
        assert len(tmpInfor) == 8
        if (nextIndex < len(coordinate)):
            for nextn, now in zip(coordinate[nextIndex], coordinate[nowIndex]):
                tmpInfor.append(now - nextn)
        else:
            for j in range(len(coordinate[nowIndex])):
                tmpInfor.append(0)
        assert len(tmpInfor) == 12
        neighborCoordinate.append(list(tmpInfor))
    return neighborCoordinate


def GetLeaveFeature(leaveIndex, trainFeatures):
    leaveFeatures = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for index in leaveIndex:
        i = 0
        tmp = index
        global_lock.acquire()
        while tmp >= len(trainFeatures[i]):
            tmp -= len(trainFeatures[i])
            i += 1
        global_lock.release()
        # print('leave index:{},i:{}'.format(tmp,i))
        leaveFeatures[i].append(trainFeatures[i][tmp])
        # leaveFeatures.append(trainFeatures[i][tmp])
    leaveFeatures = [l for l in leaveFeatures if l != []]
    return leaveFeatures


def CleanAlreadyClick(features, hrefindex, alreadyClick):
    features = copy.deepcopy(features)
    popiIndex = []
    popjIndex = []
    newalreadyclick = []
    for click in alreadyClick:
        try:
            newalreadyclick.append(click[click.index(":"):])
        except:
            newalreadyclick.append(click)
    newalreadyClick = newalreadyclick
    print('******************************new:{}'.format(alreadyClick))
    for i in reversed(range(len(features))):
        for j in reversed(range(len(features[i]))):
            try:
                if features[i][j][hrefindex][features[i][j][hrefindex].index(':'):] in newalreadyClick:
                    popjIndex.append(j)
                    popiIndex.append(i)
            except:
                if features[i][j][hrefindex] in alreadyClick:
                    popjIndex.append(j)
                    popiIndex.append(i)
    for i, j in zip(popiIndex, popjIndex):
        features[i].pop(j)
    # print(popiIndex)
    # print(popjIndex)
    return features


def Step(url, env, depth,threshold):
    previousTrainFeatures = []
    previousActionIndex = []
    leaveTrainFeatures = []
    alreadyClickHrefs = []
    alreadyClickATs = []
    predictvalue = []
    try:
        if url in url_featuresMap:
            print('in1')
            AT, HREFS, tagpaths = url_featuresMap[url]['attext'], url_featuresMap[url]['hrefs'], \
                                  url_featuresMap[url]['tagpaths']
        else:
            AT, HREFS, tagpaths = func_timeout(180, env.reset, args=(url, None, None, False))
            url_featuresMap[url] = {'attext': AT, 'hrefs': HREFS, 'tagpaths': tagpaths, 'time': str(datetime.now())}
        alreadyClickHrefs.append(env.currentURL)
    except FunctionTimedOut:
        global_metric.SaveData([], [], [], url, [])
        return '0'
    assert len(AT) == len(HREFS) == len(tagpaths)
    if len(AT) == 0:
        global_metric.SaveData([], [], [], url, [])
        return 0
    atembedding, tagid, tagindex = ProcessFeatures_nocoor(AT, tagpaths)
    leaveTrainFeatures.append(list(zip(atembedding, tagid, tagindex, HREFS, AT)))
    step = 0
    while True:
        if step == depth:
            break
        actionIndex, leaveATIndex, clickhref, clickvalue, maxvalue = Tool.GetActionIndex(agentmodel, leaveTrainFeatures,
                                                                                         3, [0, 1, 2],
                                                                                         alreadyClickHrefs,
                                                                                         randomjump=False)
        step += 1
        global_lock.acquire()
        predictvalue.append(clickvalue)
        firdimension, secdimension = Tool.ConvertToTwoDimensionIndex(actionIndex, leaveTrainFeatures)
        alreadyClickATs.append(leaveTrainFeatures[firdimension][secdimension][4])
        previousTrainFeatures.append(leaveTrainFeatures)
        previousActionIndex.append(actionIndex)
        # print(leaveTrainFeatures)
        print('homepage:{},already:{},now:{},click AT:{}, toHREFS:{}'.format(url, alreadyClickHrefs,
                                                                             alreadyClickHrefs[-1],
                                                                             leaveTrainFeatures[firdimension][
                                                                                 secdimension][4], clickhref))
        alreadyClickHrefs.append(clickhref)
        global_lock.release()
        try:
            if clickhref in url_featuresMap:
                print('in')
                tmpAT, tmpHREFS, tmptagpath = url_featuresMap[clickhref]['attext'], \
                                              url_featuresMap[clickhref]['hrefs'], \
                                              url_featuresMap[clickhref]['tagpaths']
            else:
                tmpAT, tmpHREFS, tmptagpath = func_timeout(180, env.step, args=(clickhref, False))
                # step to next step 跳轉到現在要點擊的 href
                url_featuresMap[clickhref] = {'attext': tmpAT, 'hrefs': tmpHREFS, 'tagpaths': tmptagpath,
                                              'time': str(datetime.now())}

        except FunctionTimedOut:
            break
        if (len(tmpAT) == 0):
            break
        tmpatembedding, tmptagid, tmptagindex = ProcessFeatures_nocoor(tmpAT, tmptagpath)
        leaveFeatures = GetLeaveFeature(leaveATIndex, leaveTrainFeatures)
        atembedding = tmpatembedding
        tagid = tmptagid
        tagindex = tmptagindex
        HREFS = tmpHREFS
        AT = tmpAT
        leaveTrainFeatures = leaveFeatures + [list(zip(atembedding, tagid, tagindex, HREFS, AT))]
        leaveTrainFeatures = CleanAlreadyClick(leaveTrainFeatures, 3, alreadyClickHrefs)
        assert len(AT) == len(HREFS) == len(tmptagpath)
    ans, rewards = env.GetAnsAndDiscountedReward(alreadyClickHrefs[1:], negativeReward=-0.1)
    global_lock.acquire()
    global_metric.SaveData(ans, alreadyClickHrefs[1:], alreadyClickATs, url, predictvalue)
    global_lock.release()



class Worker(threading.Thread):
    def __init__(self, queue, steps, env, threshold):
        threading.Thread.__init__(self)
        self.queue = queue
        self.steps = steps
        self.env = env
        self.threshold = threshold

    def run(self):
        global_lock.acquire()
        while self.queue.qsize() > 0:
            print('size:{}'.format(self.queue.qsize()))
            msg = self.queue.get()
            try:
                global_lock.release()
            except:
                pass
            Step(msg, self.env, self.steps,self.threshold)
        try:
            global_lock.release()
        except:
            pass


# %%
def UseThreadRunData(datalist, steps, envs,threshold):
    #global workers
    workers = []
    my_queue = Queue()
    for homepage in list(datalist):
        my_queue.put(homepage)
    for i in range(len(envs)):
        w = Worker(my_queue,steps,envs[i],threshold)
        workers.append(w)
    for i in range(len(envs)):
        workers[i].start()
    for i in range(len(envs)):
        workers[i].join()

if __name__ == '__main__':
    modelname, steps, numofthread = add_arg()
    bertlayer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", trainable=False)
    global_lock = threading.Lock()
    # traindf = pd.read_csv("dataset_google_search/google_search_rank_train.csv")
    # validation = pd.read_csv("dataset_google_search/google_search_rank_vali.csv")
    traindf = pd.read_csv("dataset_google_search/google_search_rank_train - contain news.csv")
    validation = pd.read_csv("dataset_google_search/google_search_rank_vali - contain_news.csv")
    testdf = pd.read_csv("dataset_google_search/newdata_test - remove error url.csv")
    totalDF = pd.concat([traindf, validation, testdf])
    testwebsiteInfo = testdf.groupby("Homepage")
    tokenize = Tokenize()
    tokenize.LoadObj(config.get('TagTokenize', 'vocab_file'))

    # testlist = Tool.LoadList('testdata')
    # cu_ob = {'TransformerEncoder':keras_nlp.layers.TransformerEncoder,
    #          'TokenAndPositionEmbedding':keras_nlp.layers.TokenAndPositionEmbedding}
    # with keras.utils.generic_utils.custom_object_scope(cu_ob):
    #     agentmodel = tf.keras.models.load_model('{}'.format(modelname))
    agentmodel = tf.keras.models.load_model('{}'.format(modelname))
    url_featuresMap = delOldData()

    envs = []
    for i in range(numofthread):
        envs.append(PolicyDeepthEnvFixCoordinate_singlefile.Env(totalDF, negativeReward=-0.1, rewardweight=0.1))

    # if dataset == 'testdata' or dataset == 'both':
    global_metric = GlobalMetric()
    UseThreadRunData(list(testwebsiteInfo.groups),steps,envs,0)
    clickvalue = []
    for i in global_metric.clickValue:
        tmp = []
        for j in i:
            tmp.append(str(j))
        clickvalue.append(tmp)
    data = {'ClickHomepage':global_metric.clickHomepages,'Ans':global_metric.metricAns,
            'ClickProb':clickvalue,'ClickAT':global_metric.clickATs,'ClickHref':global_metric.clickHrefs}
    Tool.SaveObj('{}/{}_evaluationdata_fixstep={}'.format('/'.join(modelname.split('/')[0:-1]),
                                                          modelname.split('/')[-1],steps),data)

    # if dataset == 'mixtestdata' or dataset == 'both':
    #     global_metric = GlobalMetric()
    #     UseThreadRunData(testlist,steps,envs,0)
    #     clickvalue = []
    #     for i in global_metric.clickValue:
    #         tmp = []
    #         for j in i:
    #             tmp.append(str(j))
    #         clickvalue.append(tmp)
    #     data  ={'ClickHomepage':global_metric.clickHomepages,'Ans':global_metric.metricAns
    #         ,'ClickProb':clickvalue,'ClickAT':global_metric.clickATs,'ClickHref':global_metric.clickHrefs}
    #     Tool.SaveObj('{}/{}_mixtestdata_fixstep={}'.format('/'.join(modelname.split('/')[0:-1]),
    #                                                        modelname.split('/')[-1],steps),data)
    Tool.SaveObj(config.get('Main', 'url_feature_file_path_no_coor'),url_featuresMap)
    for env_thread in envs:
       env_thread.webQuit()
