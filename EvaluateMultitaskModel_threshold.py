#!/usr/bin/env python
# coding: utf-8

# In[17]:
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

config = Tool.LoadConfig('config.ini')
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def add_arg():
    argstool = Args.ArgparseTool()
    argstool.NeedText('--modelname', '-mn', 'your model name', 'modelname')
    argstool.NeedText('--withthreshold', '-wbl', 'your threshold', 'threshold', required=False, default='0')
    argstool.NeedBoolean('--updatethreshold', '-u', 'updatethreshold', 'updatethreshold', required=False, default=True)
    argstool.NeedText('--numofthread', '-n', 'numofthread', 'numofthread', required=False, default=4)
    arg = argstool.GetArgs()
    modelname = arg.modelname
    threshold = arg.threshold
    updatethreshold = arg.updatethreshold
    numofthread = int(arg.numofthread)
    return modelname, threshold, updatethreshold, numofthread


def GetLeaveTime():
    now_time = datetime.now()
    leave_time = now_time - timedelta(days=7)
    return leave_time


def delOldData():
    if os.path.isfile(config.get('Main', 'url_feature_file_path_no_coor') + '.txt'):
        url_featuresMap = Tool.LoadObj(config.get('Main', 'url_feature_file_path_no_coor'))
    else:
        url_featuresMap = {}
    leavetime = GetLeaveTime()
    delkeys = []
    for key, value in url_featuresMap.items():
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
    return StringTool.GetTextListEmbedding(at, bertlayer, config), list(coo), list(tagid.numpy()), GetIndexMetrix(
        tagid.shape)

def ProcessFeatures_nocoor(at, tagpath):
    tagid = GetTokenTagPath(tagpath)
    tagid = tf.repeat(np.array([tagid]), len(at), axis=0)
    return StringTool.GetTextListEmbedding(at, bertlayer, config), list(tagid.numpy()), GetIndexMetrix(tagid.shape)

def ProcessFeatures_sequnece_output(at, tagpath):
    tagid = GetTokenTagPath(tagpath)
    tagid = tf.repeat(np.array([tagid]),len(at),axis=0)
    return StringTool.GetTextListEmbedding_sequence_output(at,bertlayer,config), list(tagid.numpy()), GetIndexMetrix(tagid.shape)
def GetIndexMetrix(tagshape):
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


def get_action_index(model, state_features, actionClickHref, data, step, dynamicstep, FeatureNotNull):
    pred_value = []
    null = []
    if FeatureNotNull:
        url = []
        AT = []
        for batch in copy.deepcopy(state_features[len(state_features) - 1]):
            features = list(zip(*batch))
            features2 = copy.deepcopy(features)

            for idx, f in enumerate(features2):
                if idx == 3:
                    url.append(f)
                elif idx == 4:
                    AT.append(f)
            model_features = [np.array(f) for index, f in enumerate(features) if index in [0, 1, 2]]
            if len(model_features) == 0:
                continue
            atpredd = model(model_features)  # [[0.xx],[0.xx],...]
            pred_value += list(atpredd[:, 0].numpy())  # [<tf.Tensor: shape=(), dtype=float32, numpy=0.xxx>,<>,...]
            # print('predvalue',predvalue)
            # if len(features) == 5:
            #     for idx, q_value in enumerate(pred_value):
            #         q_value = tf.convert_to_tensor(q_value, dtype=float)
            #         state_features[len(state_features) - 1][0][idx] += (q_value,)
        try:
            data['url'].extend(list(url[0]))
            data['q(s,a)'].extend(pred_value)
            data['AT'].extend(list(AT[0]))
            # print(data)
        except Exception:
            data['url'].extend(null)
            data['q(s,a)'].extend(null)
            data['AT'].extend(null)
            # print(data)
    print('round:{} get action data len:{}'.format(step, len(data['url'])))
    originalactionIndexs = list(tf.argsort(data['q(s,a)'], axis=-1, direction='DESCENDING').numpy())
    actionIndex = originalactionIndexs[0]
    count_idx = 1
    while data['url'][actionIndex] in actionClickHref:
        actionIndex = originalactionIndexs[count_idx]
        # print('pick {} max actionIndex:{}'.format(count_idx + 1, actionIndex))
        count_idx += 1
    clickhref = data['url'][actionIndex]
    predictvalue = data['q(s,a)'][actionIndex]
    clickAT = data['AT'][actionIndex]

    if step == 5 and not dynamicstep:
        data.clear()
    # print('actionIndex:{}, clickAT:{}, clickhref:{}'.format(actionIndex, clickAT, clickhref))
    return actionIndex, clickhref, clickAT, predictvalue


def url_to_feature(url, env):
    previousTrainFeatures = []
    for i in range(1):
        try:
            if url in url_featuresMap:
                print('url in url featureMap:', url)
                AT, HREFS, tagpaths = url_featuresMap[url]['attext'], url_featuresMap[url]['hrefs'], \
                                      url_featuresMap[url]['tagpaths']
            else:
                AT, HREFS, tagpaths = func_timeout(180, env.reset, args=(url, None, None, False))
                # print(len(AT),len(HREFS),len(tagpaths),len(coordinate))
                url_featuresMap[url] = {'attext': AT, 'hrefs': HREFS, 'tagpaths': tagpaths, 'time': str(datetime.now())}

        except FunctionTimedOut:
            print('this url cannot get')
            global_metric.SaveData([], [], [], url, [])
            return None, None
        if len(AT) == 0 or len(AT) == 1:
            print('len(AT)==0 or len(AT) == 1 ')
            global_metric.SaveData([], [], [], url, [])
            return None, None
        assert len(AT) == len(HREFS)

        atembedding, tagid, tagindex = ProcessFeatures_sequnece_output(AT, tagpaths)
        previousTrainFeatures.append(list(zip(atembedding, tagid, tagindex, HREFS, AT)))
    return previousTrainFeatures, url


def Step(url, previousTrainFeatures, threshold, dynamicstep, env):
    TrainFeatures = copy.deepcopy(previousTrainFeatures)
    TrainFeatures = [TrainFeatures]
    actionClickHref = []
    actionClickATs = []
    actionpredictvalue = []
    data = {'url': [], 'q(s,a)': [], 'AT': []}
    FeatureNotNull = True
    step = 0
    while True:
        nextTrainFeatures = []
        if step == 6 and not dynamicstep:
            print('not dynamic lower break step',step)
            break
        elif dynamicstep and step == 10:
            print('step ==10 break step',step)
            data.clear()
            break
        if FeatureNotNull:
            actionIndex, clickhref, clickAT, predictvalue = get_action_index(agentmodel, TrainFeatures, actionClickHref,
                                                                             data, step, dynamicstep, True)
        else:
            actionIndex, clickhref, clickAT, predictvalue = get_action_index(agentmodel, TrainFeatures, actionClickHref,
                                                                             data, step, dynamicstep, False)
        if predictvalue < threshold and dynamicstep:
            print('clickAT:{}, pred:{} '.format(clickAT, predictvalue))
            print('predvalue lower break step',step)
            data.clear()
            break
        step += 1
        global_lock.acquire()
        actionClickHref.append(clickhref)
        actionClickATs.append(clickAT)
        actionpredictvalue.append(predictvalue)
        print('clickAT:{}, toHref:{} '.format(actionClickATs[-1], actionClickHref[-1]))
        print('step',step)
        global_lock.release()

        try:
            if clickhref in url_featuresMap:
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
        tmpatembedding, tmptagid, tmptagindex = ProcessFeatures_sequnece_output(tmpAT, tmptagpath)
        # if i == 9 and len(tmpHREFS) != 0 and len(tmpHREFS) < 1000:
        #     # print('tmpHREFS len',len(tmpHREFS))
        #     q_value = np.zeros(len(tmpHREFS))
        #     q_value = tf.convert_to_tensor(q_value, dtype=float)
        #     nextTrainFeatures.append(list(zip(tmpatembedding, tmptagid, tmptagindex, tmpHREFS, tmpAT, q_value)))
            # nextTrainFeatures.append(list(zip(tmpatembedding, tmpHREFS, tmpAT, q_value)))
        if len(tmpHREFS) == 0 or len(tmpAT) > 600:
            nextTrainFeatures.append(list(zip([], [], [], [], [])))
        else:
            nextTrainFeatures.append(list(zip(tmpatembedding, tmptagid, tmptagindex, tmpHREFS, tmpAT)))
        # print('nextTrainFeatures[0] len ', len(nextTrainFeatures[0]))
        if len(nextTrainFeatures[0]) != 0:
            newTrainFeatures = [*TrainFeatures, nextTrainFeatures]
            FeatureNotNull = True
        else:
            newTrainFeatures = TrainFeatures
            FeatureNotNull = False
        TrainFeatures = newTrainFeatures
    ans, reward = env.GetAnsAndDiscountedReward(actionClickHref, negativeReward=-0.1)
    global_lock.acquire()
    global_metric.SaveData(ans, actionClickHref, actionClickATs, url, actionpredictvalue)
    global_lock.release()


class Worker(threading.Thread):
    def __init__(self, queue, env, threshold, dynamicstep):
        threading.Thread.__init__(self)
        self.queue = queue
        self.env = env
        self.threshold = threshold
        self.dynamicstep = dynamicstep


    def run(self):
        global_lock.acquire()
        while self.queue.qsize() > 0:
            print('size:{}'.format(self.queue.qsize()))
            url = self.queue.get()
            try:
                global_lock.release()
            except:
                pass
            # Step(url, self.steps,self.env,self.threshold)
            prev_observation, url = url_to_feature(url, self.env)
            start_time = datetime.now()
            if prev_observation == None or url == None:
                continue
            else:
                Step(url, prev_observation, self.threshold, self.dynamicstep, self.env)
        try:
            global_lock.release()
        except:
            pass


def UseThreadRunData(datalist, envs, threshold, dynamicstep):
    # global workers
    workers = []
    my_queue = Queue()
    for homepage in list(datalist):
        my_queue.put(homepage)
    for i in range(len(envs)):
        w = Worker(my_queue, envs[i], threshold, dynamicstep)
        workers.append(w)
    for i in range(len(envs)):
        workers[i].start()
    for i in range(len(envs)):
        workers[i].join()


if __name__ == '__main__':
    modelname, threshold, updatethreshold, numofthread = add_arg()
    bertlayer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", trainable=False)
    global_lock = threading.Lock()
    # traindf = pd.read_csv("dataset_google_search/google_search_rank_train.csv")
    # validation = pd.read_csv("dataset_google_search/google_search_rank_vali.csv")
    traindf = pd.read_csv("dataset_google_search/google_search_rank_train - contain news.csv")
    validation = pd.read_csv("dataset_google_search/google_search_rank_vali - contain_news.csv")
    testdf = pd.read_csv("dataset_google_search/newdata_test - remove error url.csv")
    totalDF = pd.concat([traindf, validation, testdf])
    trainwebsiteInfo = traindf.groupby("Homepage")
    testwebsiteInfo = testdf.groupby("Homepage")
    tokenize = Tokenize()
    tokenize.LoadObj(config.get('TagTokenize', 'vocab_file'))

    testlist = Tool.LoadList('testdata')
    # cu_ob = {'TransformerEncoder': keras_nlp.layers.TransformerEncoder,
    #          'TokenAndPositionEmbedding': keras_nlp.layers.TokenAndPositionEmbedding}
    # with keras.utils.generic_utils.custom_object_scope(cu_ob):
    #     agentmodel = tf.keras.models.load_model('{}'.format(modelname))
    agentmodel = tf.keras.models.load_model('{}'.format(modelname))

    url_featuresMap = delOldData()
    envs = []
    for i in range(numofthread):
        envs.append(PolicyDeepthEnvFixCoordinate_singlefile.Env(totalDF, negativeReward=-0.1, rewardweight=0.1))

    if updatethreshold:
        tmp = []
        # if dynamicStep:
        global_metric = GlobalMetric()
        UseThreadRunData(list(trainwebsiteInfo.groups)[0:40], envs, float(0), False)
        for r in global_metric.clickValue:
            for v in r:
                tmp.append(v)
        if len(tmp) > 0:
            threshold = sum(tmp) / len(tmp)
        else:
            threshold = 0

        clickvalue = []
        for i in global_metric.clickValue:
            tmp = []
            for j in i:
                tmp.append(str(j))
            clickvalue.append(tmp)
        data = {'ClickHomepage': global_metric.clickHomepages, 'Ans': global_metric.metricAns, 'ClickProb': clickvalue,
                'ClickAT': global_metric.clickATs, 'ClickHref': global_metric.clickHrefs}
        Tool.SaveObj('{}_traindata_base_{}'.format(os.path.join(modelname), str(threshold)), data)

    if updatethreshold:
        global_metric = GlobalMetric()
        UseThreadRunData(list(testwebsiteInfo.groups), envs, threshold, True)
        clickvalue = []
        for i in global_metric.clickValue:
            tmp = []
            for j in i:
                tmp.append(str(j))
            clickvalue.append(tmp)
        data = {'ClickHomepage': global_metric.clickHomepages, 'Ans': global_metric.metricAns,
                'ClickProb': clickvalue, 'ClickAT': global_metric.clickATs, 'ClickHref': global_metric.clickHrefs}
        Tool.SaveObj('{}_evaluationdata_base_{}'.format(os.path.join(modelname), str(threshold)), data)

    Tool.SaveObj(config.get('Main', 'url_feature_file_path_no_coor'), url_featuresMap)
    for env_thread in envs:
        env_thread.webQuit()
