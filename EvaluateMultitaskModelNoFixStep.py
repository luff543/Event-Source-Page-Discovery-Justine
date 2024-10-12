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
import random
from collections import OrderedDict

#import keras_nlp
config = Tool.LoadConfig('config.ini')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def add_arg():
    argstool = Args.ArgparseTool()
    argstool.NeedText('--modelname', '-mn', 'your model name', 'modelname')
    # argstool.NeedText('--dataset', '-d', 'your dataset', 'dataset', required=False, default='testdata')
    argstool.NeedText('--numofthread', '-n', 'numofthread', 'numofthread', required=False, default=4)
    argstool.NeedBoolean('--multitask', '-m', 'multitask', 'multitask', required=False, default=False)
    argstool.NeedBoolean('--bilstm', '-b', 'bilstm', 'bilstm', required=False, default=False)
    argstool.NeedBoolean('--top3', '-t', 'top3', 'top3', required=False, default=False)
    # argstool.NeedText('--unitcost', '-uc', 'your unitcost', 'unitcost', required=False, default=0.2)
    arg = argstool.GetArgs()
    modelname = arg.modelname
    # dataset = arg.dataset
    numofthread = int(arg.numofthread)
    multitask = arg.multitask
    bilstm = arg.bilstm
    top3 = arg.top3
    # unitcost = round(float(arg.unitcost),2)
    # assert dataset == 'testdata' or dataset == 'mixtestdata' or dataset == 'both'
    # return modelname, dataset,numofthread,unitcost
    # return modelname, dataset,numofthread,multitask,bilstm,top3
    return modelname,numofthread,multitask,bilstm,top3



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
        self.metricAns.append(anses)
        self.clickHrefs.append(hrefs)
        self.clickATs.append(ats)
        self.clickHomepages.append(homepage)
        self.clickValue.append(clickValue)

def GetTokenTagPath(path):
    ids = [tokenize.GetIDs(tmp) for tmp in path]
    tagoutput = np.zeros((len(ids),tokenize.GetDictLength()))
    for i in range(len(ids)):
        tagoutput[i] = np.bincount(ids[i],minlength=tokenize.GetDictLength())#pad_sequences([np.bincount(ids[i],minlength=tokenize.GetDictLength())],maxlen = tokenize.GetDictLength(), dtype=np.int32,padding='post',truncating='post',value=0)[0]
    return tagoutput
def ProcessFeatures(at, coordinate, tagpath):
    coo = GetCoordinateNeighbor(coordinate)
    tagid = GetTokenTagPath(tagpath)
    tagid = tf.repeat(np.array([tagid]),len(at),axis=0)
    return StringTool.GetTextListEmbedding(at,bertlayer,config), list(coo), list(tagid.numpy()), GetIndexMetrix(tagid.shape)
def ProcessFeatures_sequnece_output(at, tagpath):
    tagid = GetTokenTagPath(tagpath)
    tagid = tf.repeat(np.array([tagid]),len(at),axis=0)
    return StringTool.GetTextListEmbedding_sequence_output(at,bertlayer,config), list(tagid.numpy()), GetIndexMetrix(tagid.shape)
def ProcessFeatures_nocoor(at, tagpath):
    tagid = GetTokenTagPath(tagpath)
    tagid = tf.repeat(np.array([tagid]),len(at),axis=0)
    return StringTool.GetTextListEmbedding(at,bertlayer,config), list(tagid.numpy()), GetIndexMetrix(tagid.shape)
def GetIndexMetrix(tagshape):
    shape = (tagshape[0],tagshape[1],36)
    index = np.zeros(shape)
    #print('shape:{}'.format(shape))
    for i in range(shape[0]):
        index[i][i] = 1
    return index
def GetCoordinateNeighbor(coordinate):
    neighborCoordinate = []
    for nowIndex in range(len(coordinate)):
        tmpInfor = []#list(coordinate[nowIndex])
        tmpInfor.extend(list(coordinate[nowIndex]))
        assert len(tmpInfor) == 4
        previousIndex = nowIndex-1
        nextIndex = nowIndex+1
        if(previousIndex>=0):
            for previous,now in zip(coordinate[previousIndex],coordinate[nowIndex]):
                tmpInfor.append(now-previous)
        else:
            for j in range(len(coordinate[nowIndex])):
                tmpInfor.append(0)
        assert len(tmpInfor) == 8
        if(nextIndex<len(coordinate)):
            for nextn,now in zip(coordinate[nextIndex],coordinate[nowIndex]):
                tmpInfor.append(now-nextn)
        else:
            for j in range(len(coordinate[nowIndex])):
                tmpInfor.append(0)
        assert len(tmpInfor) == 12
        neighborCoordinate.append(list(tmpInfor))
    return neighborCoordinate

def get_action_index(model,state_features,actionClickHref, data, top3_data,step,multitask, FeatureNotNull):
    print('multitask',multitask)
    # print('FeatureNotNull',FeatureNotNull)
    pred_value = []
    null =[]
    url = []
    AT = []
    if FeatureNotNull:
        for batch in copy.deepcopy(state_features[len(state_features)-1]):
            features = list(zip(*batch))
            features2 = copy.deepcopy(features)

            for idx, f in enumerate(features2):
               if idx == 3:
                   url.append(f)
               elif idx == 4:
                   AT.append(f)
            model_features = [np.array(f) for index, f in enumerate(features) if index in [0,1,2]]
            if len(model_features) == 0:
                continue
            if multitask:
                atpredd, _ = model(model_features)
            else:
                atpredd = model(model_features)  # [[0.xx],[0.xx],...]
            pred_value += list(atpredd[:, 0].numpy())  # [<tf.Tensor: shape=(), dtype=float32, numpy=0.xxx>,<>,...]
            # print('len pred_value',len(pred_value))
        pageActionIndexs = list(tf.argsort(pred_value, axis=-1, direction='DESCENDING').numpy())
        # print('len pageActionIndexs',len(pageActionIndexs))
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
        try:
            top3_data['url'].extend(list(url[0]))
            top3_data['q(s,a)'].extend(pred_value)
            top3_data['AT'].extend(list(AT[0]))
            if step == 0:
                top3_data['index'].extend(list(range(0, len(data['url']))))
            else:
                top3_data['index'].extend(list(range(len(data['url']) - len(pageActionIndexs), len(data['url']))))
            # data['flag'].extend(np.zeros(len(list(url[0]))))
            # print(data)
        except Exception:
            top3_data['url'].extend(null)
            top3_data['q(s,a)'].extend(null)
            top3_data['AT'].extend(null)
            top3_data['index'].extend(null)
    # originalactionIndexs = list(tf.argsort(data['q(s,a)'], axis=-1, direction='DESCENDING').numpy())
    else:
        for batch in state_features[len(state_features) - 1]:
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
            # out, _ = model(model_features)
            if multitask:
                out, _ = model(model_features)
            else:
                out = model(model_features)  # [[0.xx],[0.xx],...]
            pred_value += list(out[:, 0].numpy())
        pageActionIndexs = list(tf.argsort(pred_value, axis=-1, direction='DESCENDING').numpy())
        try:
            top3_data['url'].extend(list(url[0]))
            top3_data['q(s,a)'].extend(pred_value)
            top3_data['AT'].extend(list(AT[0]))
            if step == 0:
                top3_data['index'].extend(list(range(0, len(data['url']))))
            else:
                top3_data['index'].extend(list(range(len(data['url']) - len(pageActionIndexs), len(data['url']))))
            # data['flag'].extend(np.zeros(len(list(url[0]))))
            # print(data)
        except Exception:
            top3_data['url'].extend(null)
            top3_data['q(s,a)'].extend(null)
            top3_data['AT'].extend(null)
            top3_data['index'].extend(null)
    print('round:{} get action top3_data len:{}'.format(step,len(top3_data['url'])))
    # print('data len',len(data['url']))
    if step == 0:
        top3_correct_index = list(pageActionIndexs[:3])
    else:
        top3_correct_index = [idx+(len(data['url']) - len(pageActionIndexs)) for idx in list(pageActionIndexs[:3])]

    listlactionIndexs = list(tf.argsort(top3_data['q(s,a)'], axis=-1, direction='DESCENDING').numpy())
    action_idx = listlactionIndexs[0]
    count_idx = 1
    last = False
    # print('actionClickHref', actionClickHref)
    # print('top3_data_url len:{},url:{}'.format(len(top3_data['url']),top3_data['url']))
    while top3_data['url'][action_idx] in actionClickHref and last == False:
        if len(top3_data['url'][action_idx]) - count_idx <= 1:
            action_idx = random.randint(0, len(top3_data['AT']) - 1)
            last = True
            # print('use last')
        else:
            try:
                # print('count_idx',count_idx)
                action_idx = listlactionIndexs[count_idx]
                tryclick = top3_data['url'][action_idx]
                # print('tryclick id:{},url:{}'.format(action_idx,tryclick))
            except Exception as e:
                # print('error',e)
                action_idx = random.randint(0, len(top3_data['AT']) - 1)
            last = False
        count_idx += 1
    # print('top3_data index',top3_data['index'])
    # print('top3_data url', top3_data['url'])
    new_action_index = top3_data['index'][action_idx]
    clickhref = data['url'][new_action_index]
    if count_idx >=2:
        tryclick_ = [tryclick]
        clickhref_ = [clickhref]
        if tryclick_ != clickhref_:
            tryclick_http = ['http' + i[5:] if i[0:5] == 'https' else i for i in tryclick_]
            clickhref_http = ['http' + i[5:] if i[0:5] == 'https' else i for i in clickhref_]
            if tryclick_http == clickhref_http:
                print('only different of http and https')
                clickhref = tryclick
    predictvalue=data['q(s,a)'][new_action_index]
    clickAT = data['AT'][new_action_index]
    del top3_data['url'][len(top3_data['url']) - len(pageActionIndexs):]
    del top3_data['q(s,a)'][len(top3_data['q(s,a)']) - len(pageActionIndexs):]
    del top3_data['AT'][len(top3_data['AT']) - len(pageActionIndexs):]
    del top3_data['index'][len(top3_data['index']) - len(pageActionIndexs):]
    # print('del this page now top3 index',top3_data['index'])
    # print('del this page now top3 url', top3_data['url'])
    if FeatureNotNull:
        top3_data['url'].extend([url[0][i] for i in pageActionIndexs[:3]])
        top3_data['q(s,a)'].extend([pred_value[i] for i in pageActionIndexs[:3]])
        top3_data['AT'].extend([AT[0][i] for i in pageActionIndexs[:3]])
        top3_data['index'].extend(top3_correct_index)
    # print('add top3 now top3 index', top3_data['index'])
    # print('add top3 now top3 url', top3_data['url'])
    # 找到所有匹配的键在 url 列表中的位置
    indices_to_remove = [index for index, value in reversed(list(enumerate(top3_data['url']))) if value == clickhref]
    # 逆序删除匹配位置的值
    for key in top3_data:
        top3_data[key] = [value for index, value in enumerate(top3_data[key]) if index not in indices_to_remove]
    # print('del click url top3 index', top3_data['index'])
    # print('add click url now top3 url', top3_data['url'])
    # 再次檢查有無重複的 url
    seen = set()
    indices_to_remove2 = []
    for index, value in reversed(list(enumerate(top3_data['url']))):
        if value in seen:
            indices_to_remove2.append(index)
        else:
            seen.add(value)
    # 删除重複的值
    for key in top3_data:
        top3_data[key] = [value for index, value in enumerate(top3_data[key]) if index not in indices_to_remove2]
    # print('del double url top3 index', top3_data['index'])
    # print('add double url now top3 url', top3_data['url'])
    # print(top3_data)
    # print('data', data)
    # if step == 5:
    #     data.clear()
    # print('actionIndex:{}, clickAT:{}, clickhref:{}'.format(actionIndex, clickAT, clickhref))
    return new_action_index, clickhref, clickAT,predictvalue
def get_action_index_notop3(model, state_features, actionClickHref, data, step, multitask,FeatureNotNull):
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
            if multitask:
                atpredd ,_= model(model_features)  # [[0.xx],[0.xx],...]
            else:
                atpredd = model(model_features)
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
    last = False

    while data['url'][actionIndex] in actionClickHref and last == False:
        if len(data['url'][actionIndex]) - count_idx <= 1:
            actionIndex = random.randint(0, len(data['AT']) - 1)
            last = True
        else:
            try:
                actionIndex = originalactionIndexs[count_idx]
            except Exception:
                actionIndex = random.randint(0, len(data['AT']) - 1)
            last = False

        # actionIndex = originalactionIndexs[count_idx]
        # print('pick {} max actionIndex:{}'.format(count_idx + 1, actionIndex))
        count_idx += 1
    clickhref = data['url'][actionIndex]
    predictvalue = data['q(s,a)'][actionIndex]
    clickAT = data['AT'][actionIndex]

    # print('actionIndex:{}, clickAT:{}, clickhref:{}'.format(actionIndex, clickAT, clickhref))
    return actionIndex, clickhref, clickAT, predictvalue
def url_to_feature(url,bilstm,env):
    print('bilstm',bilstm)
    previousTrainFeatures = []
    for i in range(1):
        try:
            if url in url_featuresMap:
                print('url in url featureMap:', url)
                AT, HREFS, tagpaths = url_featuresMap[url]['attext'], url_featuresMap[url]['hrefs'],\
                                    url_featuresMap[url]['tagpaths']
            else:
                AT, HREFS, tagpaths = func_timeout(180, env.reset, args=(url, None, None, False))
                # print(len(AT),len(HREFS),len(tagpaths),len(coordinate))
                url_featuresMap[url] = {'attext': AT, 'hrefs': HREFS, 'tagpaths': tagpaths,'time': str(datetime.now())}

        except FunctionTimedOut:
            print('this url cannot get')
            global_metric.SaveData([], [], [], url, [])
            return None,None
        if len(AT) == 0 or len(AT) == 1 :
            print('len(AT)==0 or len(AT) == 1 ')
            global_metric.SaveData([], [], [], url, [])
            return None,None
        assert len(AT) == len(HREFS)
        if bilstm:
            atembedding, tagid, tagindex = ProcessFeatures_sequnece_output(AT, tagpaths)
        else:
            atembedding, tagid, tagindex = ProcessFeatures_nocoor(AT, tagpaths)
        previousTrainFeatures.append(list(zip(atembedding, tagid, tagindex, HREFS, AT)))
    return previousTrainFeatures, url
def Step(url, previousTrainFeatures,multitask,bilstm,top3,env):
    TrainFeatures = copy.deepcopy(previousTrainFeatures)
    TrainFeatures = [TrainFeatures]
    actionClickHref = []
    actionClickATs = []
    actionpredictvalue = []
    data = {'url': [], 'q(s,a)': [],'AT': []}
    top3_data = {'url': [], 'q(s,a)': [], 'AT': [], 'index': []}
    FeatureNotNull = True
    ans = []
    reward = []
    step = 0
    do_step = True
    assent = 1
    print('use top3',top3)
    while do_step:
    # while assent > 0 or s_prime_count < 5:
        nextTrainFeatures = []
        global_lock.acquire()
        if FeatureNotNull:
            if top3:
                actionIndex, clickhref, clickAT,predictvalue = get_action_index(agentmodel, TrainFeatures, actionClickHref,
                                                                            data, top3_data, step, multitask,True)
            else:
                actionIndex, clickhref, clickAT, predictvalue = get_action_index_notop3(agentmodel, TrainFeatures,
                                                                                 actionClickHref,
                                                                                 data, step, multitask, True)
        else:
            if top3:
                actionIndex, clickhref, clickAT,predictvalue = get_action_index(agentmodel, TrainFeatures,actionClickHref,
                                                                            data, top3_data,step, multitask,False)
            else:
                actionIndex, clickhref, clickAT, predictvalue = get_action_index_notop3(agentmodel, TrainFeatures,
                                                                                 actionClickHref,
                                                                                 data, step, multitask, False)

        actionClickHref.append(clickhref)
        actionClickATs.append(clickAT)
        actionpredictvalue.append(predictvalue)
        print('clickAT:{}, toHref:{} '.format(actionClickATs[-1], actionClickHref[-1]))
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
        if bilstm:
            tmpatembedding, tmptagid, tmptagindex = ProcessFeatures_sequnece_output(tmpAT, tmptagpath)
        else:
            tmpatembedding, tmptagid, tmptagindex = ProcessFeatures_nocoor(tmpAT, tmptagpath)


        if len(tmpHREFS) == 0 or len(tmpAT) > 500:
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
        a, r = env.GetAnsAnddReward(clickhref, negativeReward=0)
        reward.append(r)
        ans.append(a)
        unit_cost = 0.4
        assent = assent - unit_cost + r
        if assent <= 0 or step>=300:  # 回合結束
            if top3:
                data.clear()
                top3_data.clear()
                do_step = False
            else:
                data.clear()
                do_step = False
        # if a or step == 2:  #沒找到最多就是5步就停止
        #     if top3:
        #         data.clear()
        #         top3_data.clear()
        #         do_step = False
        #     else:
        #         data.clear()
        #         do_step = False
        step += 1

    # ans, reward = env.GetAnsAndDiscountedReward(actionClickHref, negativeReward=-0.1)
    global_lock.acquire()
    global_metric.SaveData(ans,actionClickHref,actionClickATs,url,actionpredictvalue)
    global_lock.release()


class Worker(threading.Thread):
    def __init__(self, queue, env, threshold,multitask,bilstm,top3):
        threading.Thread.__init__(self)
        self.queue = queue
        self.env = env
        self.threshold = threshold
        self.multitask = multitask
        self.bilstm = bilstm
        self.top3=top3
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
            prev_observation,url = url_to_feature(url, self.bilstm,self.env)
            start_time = datetime.now()
            if prev_observation == None or url == None:
                continue
            else:
                Step(url,prev_observation, self.multitask,self.bilstm,self.top3,self.env)
        try:
            global_lock.release()
        except:
            pass

def UseThreadRunData(datalist, envs,threshold,multitask,bilstm,top3):
    #global workers
    workers = []
    my_queue = Queue()
    for homepage in list(datalist):
        my_queue.put(homepage)
    for i in range(len(envs)):
        w = Worker(my_queue,envs[i],threshold,multitask,bilstm,top3)
        workers.append(w)
    for i in range(len(envs)):
        workers[i].start()
    for i in range(len(envs)):
        workers[i].join()

if __name__ == '__main__':
    modelname, numofthread,multitask,bilstm,top3 = add_arg()
    # modelname, dataset, numofthread, unit_cost = add_arg()
    bertlayer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", trainable=False)
    global_lock = threading.Lock()
    # traindf = pd.read_csv("dataset_google_search/google_search_rank_train.csv")
    # validation = pd.read_csv("dataset_google_search/google_search_rank_vali.csv")
    traindf = pd.read_csv("dataset_google_search/google_search_rank_train - contain news.csv")
    validation = pd.read_csv("dataset_google_search/google_search_rank_vali - contain_news.csv")
    testdf = pd.read_csv("dataset_google_search/newdata_test - remove error url.csv")
    # testdf = pd.read_csv("dataset_google_search/newdata_test - test.csv")
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
    UseThreadRunData(list(testwebsiteInfo.groups),envs,0,multitask,bilstm,top3)
    clickvalue = []
    for i in global_metric.clickValue:
        tmp = []
        for j in i:
            tmp.append(str(j))
        clickvalue.append(tmp)
    data = {'ClickHomepage':global_metric.clickHomepages,'Ans':global_metric.metricAns,
            'ClickProb':clickvalue,'ClickAT':global_metric.clickATs,'ClickHref':global_metric.clickHrefs}
    # Tool.SaveObj('{}/{}_evaluationdata_unitcost={}'.format('/'.join(modelname.split('/')[0:-1]),
    #                                                       modelname.split('/')[-1],unit_cost),data)
    Tool.SaveObj('{}/{}_evaluationdata_asset_max300_del_click_same'.format('/'.join(modelname.split('/')[0:-1]),
                                                               modelname.split('/')[-1]), data)

    # if dataset == 'mixtestdata' or dataset == 'both':
    #     global_metric = GlobalMetric()
    #     UseThreadRunData(testlist,envs,0,multitask,bilstm,top3)
    #     clickvalue = []
    #     for i in global_metric.clickValue:
    #         tmp = []
    #         for j in i:
    #             tmp.append(str(j))
    #         clickvalue.append(tmp)
    #     data  ={'ClickHomepage':global_metric.clickHomepages,'Ans':global_metric.metricAns
    #         ,'ClickProb':clickvalue,'ClickAT':global_metric.clickATs,'ClickHref':global_metric.clickHrefs}
    #     Tool.SaveObj('{}/{}_mixtestdata'.format('/'.join(modelname.split('/')[0:-1]),
    #                                                        modelname.split('/')[-1]),data)
    Tool.SaveObj(config.get('Main', 'url_feature_file_path_no_coor'),url_featuresMap)
    for env_thread in envs:
       env_thread.webQuit()
