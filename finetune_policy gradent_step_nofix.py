import collections
import random
import copy
import json
import threading
import time
import gym, os
import csv
import logging
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from src import Tool,PolicyDeepthEnvFixCoordinate_singlefile
from Preprocess.Tokenize import Tokenize
from func_timeout import func_timeout, FunctionTimedOut
from datetime import datetime
from multiprocessing import Queue
from sklearn.metrics import precision_score
import requests
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

class Buffer():
    def __init__(self):
        self.bufferFitFeatures = []
        self.bufferRewards = []
        self.bufferActionIndex = []
        self.bufferCoordinate = []
        self.metricAns = []
        self.metricPredict = []
        self.click_step = []

    def GetBuffer(self):
        return self.bufferFitFeatures, self.bufferRewards, self.bufferActionIndex, self.bufferCoordinate

    def SaveBuffer(self, fitfeature, rewawrds, actionIndex, coordinate):
        self.bufferFitFeatures.append(fitfeature)
        self.bufferRewards.append(rewawrds)
        self.bufferActionIndex.append(actionIndex)
        self.bufferCoordinate.append(coordinate)

    def SaveMetric(self, predictvalue,click_step):
        #self.metricAns.append(ans)
        self.metricPredict.append(predictvalue)
        self.click_step.append(click_step)

    def ResetBuffer(self):
        self.bufferFitFeatures = []
        self.bufferRewards = []
        self.bufferActionIndex = []
        self.bufferCoordinate = []

    def ResetMetric(self):
        #self.metricAns = []
        self.metricPredict = []
        self.click_step = []

class AgentModel(tf.keras.Model):
    def __init__(self, tagdimension):
        super(AgentModel, self).__init__()
        self.bilstm_first = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(tagtokenize.GetDictLength(), return_sequences=True), name="tag_path1")
        self.bilstm_two = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(tagdimension // 2, return_sequences=True),
                                                        name="tag_path2")
        self.coordinate_output = tf.keras.layers.Dense(12)
        self.at_embedding_first = tf.keras.layers.Dense(768, activation='relu')
        self.at_embedding_two = tf.keras.layers.Dense(768, activation='relu')
        self.at_embedding_three = tf.keras.layers.Dense(256, activation='relu')
        self.concat_first = tf.keras.layers.Dense(128, activation='relu')
        self.concat_two = tf.keras.layers.Dense(64, activation='relu')
        self.agent_output = tf.keras.layers.Dense(1, activation='sigmoid')

        self.tagdimension = tagdimension

    def call(self, atembedding, tagpath, index):
        tagembedding = self.bilstm_first(tagpath)
        tagembedding = self.bilstm_two(tagembedding)
        tagoutput = self.coordinate_output(tagembedding)
        atembedding = self.at_embedding_first(atembedding)
        atembedding = self.at_embedding_two(atembedding)
        atembedding = self.at_embedding_three(atembedding)
        tag = tf.math.multiply(tagembedding, index)
        tag = tf.math.reduce_sum(tag, axis=1)
        concat = tf.keras.layers.Concatenate()([atembedding, tag])  # tf.gather_nd(tagembedding, full_index)])
        concatlayer = self.concat_first(concat)
        concatlayer = self.concat_two(concatlayer)
        return self.agent_output(concatlayer), tagoutput

    def model(self):
        x = [tf.keras.Input(shape=(768,)), tf.keras.Input(shape=(None, tagtokenize.GetDictLength())),
             tf.keras.Input(shape=(None, self.tagdimension))]
        # self.call(x[0],x[1], x[2])
        return tf.keras.Model(inputs=x, outputs=self.call(x[0], x[1], x[2]))
class Agent():
    def __init__(self, model, buffer):
        #### Evaluate variable
        self.metricAns = []
        #### Fit Variable
        self.buffer = buffer
        self.model = model

    def GetLeaveFeature(self, leaveIndex, trainFeatures):
        leaveFeatures = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for index in leaveIndex:
            i = 0
            tmp = index
            global_lock.acquire()
            while tmp >= len(trainFeatures[i]):
                tmp -= len(trainFeatures[i])
                i += 1
            global_lock.release()
            leaveFeatures[i].append(trainFeatures[i][tmp])
        leaveFeatures = [l for l in leaveFeatures if l != []]
        return leaveFeatures

    def CleanAlreadyClick(self, features, hrefindex, alreadyClick):
        features = copy.deepcopy(features)
        popiIndex = []
        popjIndex = []
        for i in reversed(range(len(features))):
            for j in reversed(range(len(features[i]))):
                if features[i][j][hrefindex] in alreadyClick:
                    popjIndex.append(j)
                    popiIndex.append(i)
        for i, j in zip(popiIndex, popjIndex):
            features[i].pop(j)
        return features
    def GetESPS(self,page):
        API_DOMAIN = f"{config.get('EventSourcePageScoring', 'IP')}:{config.get('EventSourcePageScoring', 'PORT')}"
        url = API_DOMAIN + "/GetPageScore"
        payload = {'url': str(page)}
        try:
            r = requests.get(url, params=payload)
        except:
            return 0
        try:
            data = r.json()
            # 取得分數資料
            score = data.get("score")
            return round(float(score),3)
        except:
            print('error:{}'.format(r.text))
            return 0
    def Step(self, url, url_idx,env, types):  # types = 'train' and 'test'
        batch = 1
        deepth = 3
        for b in range(batch):
            previousTrainFeatures = []
            previousCoordinate = []
            previousActionIndex = []
            predictvalue = []
            leaveTrainFeatures = []
            alreadyClickHrefs = []
            try:
                if url in url_featuresMap:
                    print('in1')
                    AT, HREFS, tagpaths, coordinate = url_featuresMap[url]['attext'], url_featuresMap[url]['hrefs'], \
                                                      url_featuresMap[url][ 'tagpaths'], \
                                                      url_featuresMap[url]['coordinate']
                else:
                    AT, HREFS, tagpaths, coordinate = func_timeout(180, env.reset,args=(url, None, types, True))
                    coordinate = list(coordinate)
                    url_featuresMap[url] = {'attext': AT, 'hrefs': HREFS, 'coordinate': coordinate,
                                            'tagpaths': tagpaths, 'time': str(datetime.now())}
                alreadyClickHrefs.append(env.currentURL)

            except FunctionTimedOut:
                break
            if len(AT) == 0:
                break
            assert len(AT) == len(coordinate) == len(HREFS)
            atembedding, neighborcoordinate, tagid, tagindex = Tool.ProcessFeatures(AT, coordinate, tagpaths,
                                                                                    tagtokenize, bertlayer,
                                                                                    config)  # 處理特徵的fumction
            leaveTrainFeatures.append(list(zip(atembedding, tagid, tagindex, HREFS, AT)))
            rewards = []
            reward = 0.0
            ans = []
            t = 0
            assent = 1
            while assent > 0:
            # for d in range(deepth):
                # logging.info('domain url:{}'.format(url))
                if types == 'train':
                    actionIndex, leaveATIndex, clickhref, originalpredvalue, maxprob = Tool.GetActionIndex(self.model,
                                                                                                           leaveTrainFeatures,
                                                                                                           3, [0, 1, 2],
                                                                                                           alreadyClickHrefs,
                                                                                                           True,
                                                                                                           epison)  # 透過那些處理好的特徵放入這個function就可以得到真正要點擊的action index
                else:
                    actionIndex, leaveATIndex, clickhref, originalpredvalue, maxprob = Tool.GetActionIndex(self.model,
                                                                                                           leaveTrainFeatures,
                                                                                                           3, [0, 1, 2],
                                                                                                           alreadyClickHrefs,
                                                                                                           False)
                predictvalue.append(originalpredvalue)
                global_lock.acquire()
                previousCoordinate.append(neighborcoordinate)
                previousTrainFeatures.append(leaveTrainFeatures)
                previousActionIndex.append(actionIndex)
                global_lock.release()
                firdimension, secdimension = Tool.ConvertToTwoDimensionIndex(actionIndex, leaveTrainFeatures)
                # print(leaveTrainFeatures)
                print('now:{},click AT:{}, toHREFS:{}'.format(alreadyClickHrefs[-1],
                                                              leaveTrainFeatures[firdimension][secdimension][4],
                                                              clickhref))
                alreadyClickHrefs.append(clickhref)
                try:
                    if clickhref in url_featuresMap:
                        tmpAT, tmpHREFS, tmptagpath, tmpcoordinate = url_featuresMap[clickhref]['attext'], \
                                                                     url_featuresMap[clickhref]['hrefs'], \
                                                                     url_featuresMap[clickhref]['tagpaths'], \
                                                                     url_featuresMap[clickhref]['coordinate']
                    else:
                        tmpAT, tmpHREFS, tmptagpath, tmpcoordinate = func_timeout(180, env.step, args=(
                        clickhref, True))  # step to next step 跳轉到現在要點擊的herf
                        tmpcoordinate = list(tmpcoordinate)
                        url_featuresMap[clickhref] = {'attext': tmpAT, 'hrefs': tmpHREFS, 'tagpaths': tmptagpath
                                                      ,'coordinate': tmpcoordinate,'time': str(datetime.now())}

                except FunctionTimedOut:
                    break
                if (len(tmpAT) == 0):
                    leaveTrainFeatures = self.CleanAlreadyClick(leaveTrainFeatures, 3, alreadyClickHrefs)
                    continue
                tmpatembedding, tmpneighborcoordinate, tmptagid, tmptagindex = Tool.ProcessFeatures(tmpAT,
                                                                                                    tmpcoordinate,
                                                                                                    tmptagpath,
                                                                                                    tagtokenize,
                                                                                                    bertlayer, config)
                leaveFeatures = self.GetLeaveFeature(leaveATIndex, leaveTrainFeatures)
                atembedding = tmpatembedding
                neighborcoordinate = tmpneighborcoordinate
                tagid = tmptagid
                tagindex = tmptagindex
                HREFS = tmpHREFS
                AT = tmpAT
                leaveTrainFeatures = leaveFeatures + [list(zip(atembedding, tagid, tagindex, HREFS, AT))]
                leaveTrainFeatures = self.CleanAlreadyClick(leaveTrainFeatures, 3, alreadyClickHrefs)
                #a, r = env.GetAnsAnddReward(clickhref, negativeReward=0)
                try:
                    r = func_timeout(300, self.GetESPS, args=(clickhref,))
                    print('reward',r)
                except FunctionTimedOut:
                    print("程式執行時間超過限制")
                    r = 0
                reward += r
                rewards.append(r)
                #ans.append(a)
                assent = assent - cost + r
                if assent <= 0 or t == 4:  # 回合結束
                    break
                t += 1
            assert len(alreadyClickHrefs[1:]) == len(previousTrainFeatures)
            # ans, rewards = env.GetAnsAndDiscountedReward(alreadyClickHrefs[1:],
            #                                              negativeReward=max([-0.5, float(epison - 1)]))

            print('click:{},rewards:{}'.format(alreadyClickHrefs[1:], rewards))
            global_lock.acquire()
            self.buffer.SaveBuffer(previousTrainFeatures, rewards, previousActionIndex, previousCoordinate)
            self.buffer.SaveMetric(predictvalue, t+1)
            savepreReward(reward,t+1,types)
            global_lock.release()
            break


def GetMetrix(buffer, types):
    #ans = np.array(buffer.metricAns)
    ans = buffer.metricAns
    # p_1 = None
    # for i in range(1):
    #     tmp = []
    #     for a in ans:
    #         for j in range(i + 1):
    #             try:
    #                 tmp.append(a[j])
    #             except:
    #                 continue
    #     p_1 = precision_score(tmp, [1] * len(tmp))
        #print("{}P@{}:{}".format(types, i + 1, precision_score(tmp, [1] * len(tmp))))
    reward = []
    step = 0
    for r in buffer.bufferRewards:
        reward += r
    for t in buffer.click_step:
        step += t
    print('{}TotalReward:{}'.format(types, sum(reward)))
    # print('{}p_1;{}'.format(types,p_1))
    saveReward(sum(reward),step,types)
    return sum(reward)

def saveReward(reward,step,types):
    path = 'precision_result/{}/reward'.format(savemodelname)
    if not os.path.exists(path):
        os.makedirs(path)
        with open('{}/{}/reward/reward_{}.csv'.format('precision_result', savemodelname, types), 'w',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['reward','step'])
    with open('{}/{}/reward/reward_{}.csv'.format('precision_result', savemodelname, types), 'a',
              newline='') as csvfile:
        if types == 'train':
            writer = csv.writer(csvfile)
            writer.writerow([reward,step])
        else:
            writer = csv.writer(csvfile)
            writer.writerow([reward,step])
def savepreReward(reward,step,types):
    path = 'precision_result/{}/prereward'.format(savemodelname)
    if not os.path.exists(path):
        os.makedirs(path)
        with open('{}/{}/prereward/reward_{}.csv'.format('precision_result', savemodelname, types), 'w',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['reward','step'])
    with open('{}/{}/prereward/reward_{}.csv'.format('precision_result', savemodelname, types), 'a',
              newline='') as csvfile:
        if types == 'train':
            writer = csv.writer(csvfile)
            writer.writerow([reward,step])
        else:
            writer = csv.writer(csvfile)
            writer.writerow([reward,step])
def FitModelStep(bufferFitFeatures, bufferActionIndex, bufferRewards, bufferCoordinate, types):
    # 根據收集到的資料，update model
    base = []
    for b in bufferRewards:
        base += b
    with tf.GradientTape() as gen_tape:
        loss = []
        for websitefeature, webactions, webrewards, webcoordinates in zip(copy.deepcopy(bufferFitFeatures),
                                                                          bufferActionIndex, bufferRewards,
                                                                          bufferCoordinate):
            # print("@websitefeature:", websitefeature)
            # print("@webactions", webactions)
            webrewards = np.array(webrewards) - (sum(base) / len(base))
            prob = []
            # shifts = []
            mse = []
            shifts = []
            for page, action, coordinate in zip(websitefeature, webactions, webcoordinates):
                # print("@page:", page)
                # print("@action", action)
                batchPredictValueFlatten = []
                batchPredictValue = []
                shift = []
                batchPredictValueShape = []
                tmp = 0
                for index, b in enumerate(page):
                    features = list(zip(*b))
                    if len(features) == 0:
                        continue
                    features = [np.array(f) for index, f in enumerate(features) if index in [0, 1, 2]]
                    batchvalue, batchcoor = model(features, training=True)
                    # print('@batchvalue:',batchvalue)
                    shift.append(tf.reduce_mean(tf.math.abs(batchvalue[:, 0] - 0.5)))
                    batchvalue = list(batchvalue[:, 0])
                    # print('@list(batchvalue[:,0]):',batchvalue)
                    # print('model pred coor',np.array(batchcoor).shape)
                    batchPredictValueFlatten += batchvalue
                    batchPredictValueShape.append(len(batchPredictValueFlatten))
                    tmp = len(batchvalue)
                    if index == len(page) - 1:
                        batchPredictCoordinate = batchcoor[0]
                start = 0
                # print('@batchPredictValueFlatten',batchPredictValueFlatten)
                batchPredictValueFlatten = tf.nn.softmax(batchPredictValueFlatten)
                # print('@batchPredictValueFlatten_softmax',batchPredictValueFlatten)
                for jumpIndex in batchPredictValueShape:
                    batchPredictValue.append(batchPredictValueFlatten[start:jumpIndex])
                    start = jumpIndex
                # print('shape model coordinate',np.array(coordinate).shape)
                # print('shape save batchPredictCoordinate',np.array(batchPredictCoordinate).shape)
                mse.append(tf.reduce_sum(tf.math.sqrt(tf.keras.losses.MSE(coordinate, batchPredictCoordinate))))
                i, j = Tool.ConvertToTwoDimensionIndex(action, batchPredictValue)
                prob.append(tf.math.log(batchPredictValue[i][j]))
                shifts.append(tf.reduce_sum(np.array(shift)))
            # print('np.array(webrewards).shape',np.array(webrewards).shape)
            # print('np.array(prob).shape',np.array(prob).shape)
            try:
                assert np.array(webrewards).shape == np.array(prob).shape
                prob = tf.math.multiply(prob, webrewards)
                loss.append(
                    (tf.reduce_sum(prob) * (1 - multitask_coor_weight)) - (tf.reduce_sum(mse)) * multitask_coor_weight)
            except AssertionError:
                continue
        if len(loss) != 0:
            loss = sum(loss) / len(loss)
            print('loss',loss)
    gradients_of_model = gen_tape.gradient(loss, model.trainable_variables)
    processed_grads = [-1 * tf.clip_by_value(g, clip_value_min=-1, clip_value_max=1) for g in gradients_of_model]
    model_optimizer = tf.keras.optimizers.Adam(1e-4)
    model_optimizer.apply_gradients((grad, var) for (grad, var) in zip(processed_grads, model.trainable_variables))
    return loss


def FitModelBuffer(buffer, types):
    # assert
    loss = []
    for i in range(len(buffer.bufferFitFeatures) // 5):
        startIndex = i * 5
        endIndex = i * 5 + 5
        try:
            modelloss = FitModelStep(buffer.bufferFitFeatures[startIndex:endIndex],
                                     buffer.bufferActionIndex[startIndex:endIndex],
                                     buffer.bufferRewards[startIndex:endIndex],
                                     buffer.bufferCoordinate[startIndex:endIndex], types)
            loss.append(float(modelloss.numpy()))
        except Exception:
            continue
    print('loss',sum(loss) / len(loss))
    savePerloss(sum(loss) / len(loss))
def savePerloss(loss):
    path = 'precision_result/{}/loss'.format(savemodelname)
    if not os.path.exists(path):
        os.makedirs(path)
        with open('{}/{}/loss/loss.csv'.format('precision_result', savemodelname), 'w',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['loss'])
    with open('{}/{}/loss/loss.csv'.format('precision_result', savemodelname), 'a',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([loss])
timamap = {}

class Worker(threading.Thread):
    def __init__(self, queue, env, agent, url_idx_queue,types):
        threading.Thread.__init__(self)
        self.queue = queue
        self.env = env
        self.agent = agent
        self.types = types
        self.url_idx_queue = url_idx_queue

    def run(self):
        global timamap
        while self.queue.qsize() > 0:
            print('size:{}'.format(self.queue.qsize()))
            msg = self.queue.get()
            url_idx = self.url_idx_queue.get()
            start_time = datetime.now()
            self.agent.Step(msg, url_idx, self.env, self.types)
            finish_time = datetime.now()
            timamap[msg] = (finish_time - start_time).total_seconds()


# %%
def UseThreadRunData(datalist, envs, url_idx_list,types):
    # global workers
    workers = []
    my_queue = Queue()
    queue_idx = Queue()

    # for homepage in list(datalist):
    #     my_queue.put(homepage)
    for (homepage,homepage_idx) in zip(list(datalist), url_idx_list):
        my_queue.put(homepage)
        queue_idx.put(homepage_idx)
    for i in range(len(envs)):
        w = Worker(my_queue, envs[i],Agent(model, global_buffer), queue_idx,types)
        workers.append(w)
    for i in range(len(envs)):
        workers[i].start()
        time.sleep(2)
    for i in range(len(envs)):
        workers[i].join()
def useGPU(usegpu=True):
    if usegpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print('GPU error:{}'.format(e))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
if __name__ == '__main__':
    useGPU(usegpu=True)
    config = Tool.LoadConfig('config.ini')
    tagtokenize = Tokenize()
    tagtokenize.LoadObj('Preprocess/tagtokenizemore')
    bertlayer = hub.KerasLayer(config.get('BertTokenize', 'bert_path'), trainable=False)
    if os.path.isfile(config.get('Main', 'url_feature_file_path_finetune') + '.txt'):
        url_featuresMap = Tool.LoadObj(config.get('Main', 'url_feature_file_path_finetune'))
    else:
        url_featuresMap = {}
    global_lock = threading.Lock()
    traindf = pd.read_csv("./dataset_google_search/google_search_rank_train - contain news.csv")
    validation = pd.read_csv("./dataset_google_search/google_search_rank_vali - contain_news.csv")
    pre_totalDF = pd.concat([traindf, validation])
    totalDF = pre_totalDF.dropna(subset=['Event source page'])
    fintunedf = pd.read_csv("./dataset_google_search/fintunetraindata.csv")
    trainlist = list(fintunedf['url'])
    pretrainmodel = './stepnofixmodel/pretrain_policy_gradient_step_nofix_choose.h5'

    savemodelname_dir = 'finetune_policy_gradient_no_pre_train_nofixstep_eplision_0.3'
    savemodelname = 'finetune_policy_gradient_no_pre_train_nofixstep_eplision_0.3'
    if not os.path.exists(savemodelname_dir):
        os.makedirs(savemodelname_dir)
    multitask_coor_weight = 0.5
    load_pretrain_model = False
    ## Model
    if load_pretrain_model:
        agentmodel = tf.keras.models.load_model('{}'.format(pretrainmodel))
        model = agentmodel
    else:
        tagdimension = 36
        agentmodel = AgentModel(tagdimension)
        model = agentmodel.model()
    numofthread = 2
    global_buffer = Buffer()
    agent = Agent(model, global_buffer)
    envs = []
    for i in range(numofthread):
        envs.append(PolicyDeepthEnvFixCoordinate_singlefile.Env(totalDF, negativeReward=-0.1, rewardweight=0.1))
    # model.summary()
    random.shuffle(trainlist)
    for epoch in range(1):
        # epison = max([0.1, 0.9 - int(epoch // 2) * 0.1])
        cost = 0.4 + 0.05*(epoch//10)
        # train
        for i in range(len(trainlist) // 50):
            #epison = max(0.01, 1 - (0.8 * i) // 2)
            epison = 0.3
            UseThreadRunData(trainlist[50 * i:50 * (i + 1)], envs, list(range(50 * i + 1, 50 * i + 51)), 'train')
            FitModelBuffer(global_buffer, 'train')
            GetMetrix(global_buffer, 'train')
            global_buffer.ResetBuffer()
            global_buffer.ResetMetric()
            model.save('{}/{}_epoch{}.h5'.format(savemodelname_dir, savemodelname, i), save_format="tf")


        Tool.SaveObj(config.get('Main', 'url_feature_file_path_finetune'), url_featuresMap)