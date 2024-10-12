import collections
import random
import copy
import json
import threading
import time
import gym, os
import csv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import requests
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from src import Tool,PolicyDeepthEnvFixCoordinate_singlefile
from Preprocess.Tokenize import Tokenize
from func_timeout import func_timeout, FunctionTimedOut
from datetime import datetime
from multiprocessing import Queue
from sklearn.metrics import precision_score
#from src import ESPS_reward
tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

class ReplayBuffer():
    # Replay Buffer
    def __init__(self):
        # 雙向佇列 (Double-ended queue)
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self,n):
        mini_batch = random.sample(self.buffer, n)
        return mini_batch

    def size(self):
        return len(self.buffer)
class ValiReplayBuffer():
    # Replay Buffer
    def __init__(self):
        # 雙向佇列 (Double-ended queue)
        self.Valibuffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.Valibuffer.append(transition)
    def resetbuffer(self):
        self.Valibuffer.clear()
    def pickVali(self):
        vali = self.Valibuffer
        return vali

class Qnet(keras.Model):
    def __init__(self,tagdimension):
        # 創建Q網路，輸入為狀態向量，輸出為動作的Q值
        super(Qnet, self).__init__()
        self.tagdimension = tagdimension
        self.bilstm_first = layers.Bidirectional(layers.LSTM(tagtokenize.GetDictLength(), return_sequences=True),
                                                 name="tag_path1")
        self.bilstm_two = layers.Bidirectional(layers.LSTM(tagdimension // 2, return_sequences=True), name="tag_path2")
        self.coordinate_output = layers.Dense(12)
        self.bert_bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='bert_bilstm1')
        self.bert_bilstm_two = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='bert_bilstm2')
        self.concat_first = layers.Dense(64, activation='relu')
        self.concat_two = layers.Dense(32, activation='relu')
        self.agent_output = layers.Dense(1, kernel_initializer='he_normal',activation='sigmoid')
        # self.coordinate_output = layers.Dense(12)

    def call(self, x):
        atembedding = x[0]
        tagpath = x[1]
        index = x[2]
        tagembedding = self.bilstm_first(tagpath)
        tagembedding = self.bilstm_two(tagembedding)
        coor_output = self.coordinate_output(tagembedding)
        atembedding = self.bert_bilstm(atembedding)
        atembedding = self.bert_bilstm_two(atembedding)
        atembedding = tf.math.reduce_mean(atembedding, axis=1)
        tag = tf.math.multiply(tagembedding, index)
        tag = tf.math.reduce_sum(tag, axis=1)
        concat = layers.Concatenate()([atembedding, tag])
        concatlayer = self.concat_first(concat)
        concatlayer = self.concat_two(concatlayer)
        anchor_output = self.agent_output(concatlayer)
        # return anchor_output
        return anchor_output, coor_output
    def model(self):
        x = [tf.keras.Input(shape=(None,768)), tf.keras.Input(shape=(None, tagtokenize.GetDictLength())),
             tf.keras.Input(shape=(None, self.tagdimension))]
        return tf.keras.Model(inputs=x, outputs=self.call(x))

class Agent():
    def __init__(self, model):
        #### Fit Variable
        self.model = model
        self.target_model = model

    def sample_action(self, target_url, state_features, current_url, data, actionClickHref, epsilon, step,
                      types, useprestate=False):
        url = []
        AT = []
        null = []
        pred_value = []
        print('len state',len(state_features))
        if not useprestate:
            for batch in state_features[len(state_features) - 1]:
                # for batch in state_features:
                features = list(zip(*batch))
                # print('step features len',len(features))
                # print('feature',features)
                features2 = copy.deepcopy(features)
                for idx, f in enumerate(features2):
                    if idx == 3:
                        url.append(f)
                    elif idx == 4:
                        AT.append(f)
                model_features = [np.array(f) for index, f in enumerate(features) if index in [0, 1, 2]]
                # model_features = [f for index, f in enumerate(features) if index in [0, 1, 2]]
                if len(model_features) == 0:
                    continue
                out,_ = self.model(model_features)
                pred_value += list(out[:, 0].numpy())
                if len(features) == 5:
                    for idx, q_value in enumerate(pred_value):
                        q_value = tf.convert_to_tensor(q_value, dtype=float)
                        state_features[len(state_features) - 1][0][idx] += (q_value,)

            try:
                data['url'].extend(list(url[0]))
                data['q(s,a)'].extend(pred_value)
                data['AT'].extend(list(AT[0]))
                data['step'].extend(list([step + 1] * len(AT[0])))
                # data['flag'].extend(np.zeros(len(list(url[0]))))
                # print(data)
            except Exception:
                data['url'].extend(null)
                data['q(s,a)'].extend(null)
                data['AT'].extend(null)
                data['step'].extend(null)
            # data['flag'].extend(null)
        listlactionIndexs = list(tf.argsort(data['q(s,a)'], axis=-1, direction='DESCENDING').numpy())
        coin = random.random()
        # # 策略改進：e-Greedy方法
        if types == 'train':
            if coin < epsilon:
                # 當前機率小於epsilon門檻值，隨機存取
                action_idx = random.randint(0, len(data['AT']) - 1)
                while data['url'][action_idx] in actionClickHref:
                    action_idx = random.randint(0, len(data['AT']) - 1)

            else:  # 當前機率大於等於epsilon門檻值，選擇Q值最大的動作做
                # action_idx = int(tf.argmax(data['q(s,a)']))
                action_idx = listlactionIndexs[0]
                count_idx = 1
                while data['url'][action_idx] in actionClickHref:
                    action_idx = listlactionIndexs[count_idx]
                    count_idx += 1
        else:
            action_idx = listlactionIndexs[0]
            count_idx = 1
            while data['url'][action_idx] in actionClickHref:
                action_idx = listlactionIndexs[count_idx]
                count_idx += 1

        clickAT = data['AT'][action_idx]
        clickhref = data['url'][action_idx]

        return action_idx,clickhref,clickAT
    def url_to_feature(self,url,url_idx,env,types):
        previousTrainFeatures = []
        try:
            if url in url_featuresMap:
                # print('url in url featureMap:', url)
                AT, HREFS, tagpaths, coordinate = url_featuresMap[url]['attext'], url_featuresMap[url]['hrefs'], \
                                                  url_featuresMap[url]['tagpaths'], url_featuresMap[url][
                                                      'coordinate']
            else:
                AT, HREFS, tagpaths, coordinate = func_timeout(180, env.reset, args=(url, url_idx, types, True))
                coordinate = list(coordinate)
                # print(len(AT),len(HREFS),len(tagpaths),len(coordinate))
                url_featuresMap[url] = {'attext': AT, 'hrefs': HREFS, 'tagpaths': tagpaths,
                                        'coordinate': coordinate, 'time': str(datetime.now())}

        except FunctionTimedOut:
            print('this url cannot get:{}'.format(url))
            return None
            # return None, None
        if len(AT) == 0 or len(AT) == 1:
            print('AT len is 0 or 1, url: {} '.format(url))
            return None
            # return None, None
        assert len(AT) == len(HREFS)

        atembedding, neighborcoordinate, tagid, tagindex = Tool.ProcessFeatures_sequence_output(AT, coordinate, tagpaths,
                                                                                tagtokenize, bertlayer,
                                                                                config)  # 處理特徵的fumction
        if tagindex.shape[0] != tagindex.shape[1]:
            return None
        previousTrainFeatures.append(list(zip(atembedding, tagid, tagindex, HREFS, AT)))
        # return atembedding
        # return previousTrainFeatures
        return previousTrainFeatures, neighborcoordinate
    def saveReward(self,totalreward,unitcost,t,p_1,types):
        path = 'precision_result/{}/reward'.format(savemodelname)
        if not os.path.exists(path):
            os.makedirs(path)
            with open('{}/{}/reward/reward_{}.csv'.format('precision_result', savemodelname, types), 'w',
                      newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['totalreward','unitcost','step','p_1'])
        with open('{}/{}/reward/reward_{}.csv'.format('precision_result', savemodelname, types), 'a',
                  newline='') as csvfile:
            if types == 'train':
                writer = csv.writer(csvfile)
                writer.writerow([totalreward,unitcost,t+1,p_1])
            else:
                writer = csv.writer(csvfile)
                writer.writerow([totalreward,unitcost,t+1,p_1])
    
    def updatetobuffer(self,first_s,url,pre_coor,types,env):
        s = [first_s]
        coor = []
        target_url = url
        current_url = url
        data = {'url': [], 'q(s,a)': [], 'AT': [], 'step': []}
        alreadyclickurl = []
        total_click_url = []
        useprestate = False
        score = 0.0
        precision = []
        t = 0
        assent = 1
        s_prime_count = 0
        while assent > 0 or s_prime_count < 5:
        # for t in range(3):  # 一個回合最大時間戳 (updating N times)
            # 根據當前Q網路提取策略，並改進策略
            if s_prime_count >=4:
                data.clear()
                break
            action_idx, action_url, action_at = self.sample_action(target_url, s, current_url, data,
                                                              alreadyclickurl, epsilon, t,
                                                              types,useprestate)
            alreadyclickurl.append(action_url)
            s_prime, done, r, next_coor = self.step(action_url, s, epsilon, t, env)
            if s_prime == None:
                s = s
                current_url = current_url
                pre_coor = coor
                useprestate = True
                s_prime_count += 1
                continue
            precision.append(done)
            total_click_url.append(action_url)
            print('click_url:{}, reward:{}, AT:{}'.format(action_url, r, action_at))

            done_mask = 0.0 if done else 1.0  # 結束標誌遮罩(mask)
            if types == 'train':
                memory.put((s, action_idx, r, s_prime, done_mask, pre_coor))
            else:
                vali_memory.put((s, action_idx, r, s_prime, done_mask, pre_coor))

            assent = assent - unit_cost + r

            s = s_prime
            current_url = action_url
            pre_coor = next_coor
            coor = pre_coor
            score += r  # 紀錄總回報
            useprestate = False

            if assent <= 0 or t == 9:  # 回合結束
                data.clear()
                break
            t += 1
        print('types:{}, target url:{} ,ans:{}, total step click_url:{}'.format(types,target_url,precision,
                                                                               total_click_url))
        p_1 = precision_score(precision, [1] * len(precision))
        self.saveReward(score, unit_cost,t,p_1, types)
    def step(self,action_url,TrainFeatures,epsilon,step,env):
        nextTrainFeatures = []
        try:
            if action_url in url_featuresMap:
                tmpAT, tmpHREFS,tmptagpath,tmpCoordinate = url_featuresMap[action_url]['attext'], \
                                                  url_featuresMap[action_url]['hrefs'], \
                                                  url_featuresMap[action_url]['tagpaths'], \
                                                  url_featuresMap[action_url][
                                                      'coordinate']
            else:
                tmpAT, tmpHREFS,tmptagpath,tmpCoordinate = func_timeout(240, env.step, args=(action_url, True))  # step to next step 跳轉到現在要點擊的herf
                tmpCoordinate = list(tmpCoordinate)
                # print(len(AT),len(HREFS),len(tagpaths),len(coordinate))
                url_featuresMap[action_url] = {'attext': tmpAT, 'hrefs': tmpHREFS, 'tagpaths': tmptagpath,
                                              'coordinate': tmpCoordinate, 'time': str(datetime.now())}
        except FunctionTimedOut:
            return None,None,None,None
        if len(tmpAT) == 0 or len(tmpAT) == 1:
            print('tmpAT len is 0 or 1, action url: {} '.format(action_url))
            return None,None,None,None
            # return None, None
        if len(tmpAT) >= 500:
            print('tmpAT len over 500, action url: {} '.format(action_url))
            return None,None,None,None
        assert len(tmpAT) == len(tmpHREFS)
        tmpatembedding, tmpCoordinate, tmptagid, tmptagindex = Tool.ProcessFeatures_sequence_output(tmpAT, tmpCoordinate,
                                                                                    tmptagpath, tagtokenize,
                                                                                    bertlayer, config)
        if tmptagindex.shape[0] != tmptagindex.shape[1]:
            return None,None,None,None
        if step == 9:
            q_value = np.zeros(len(tmpHREFS))
            q_value = tf.convert_to_tensor(q_value, dtype=float)
            nextTrainFeatures.append(list(zip(tmpatembedding, tmptagid, tmptagindex, tmpHREFS, tmpAT, q_value)))
        else:
            nextTrainFeatures.append(list(zip(tmpatembedding, tmptagid, tmptagindex, tmpHREFS, tmpAT)))
        newTrainFeatures = [*TrainFeatures,nextTrainFeatures]
        # print('len nextTrainFeatures[0]:{} '.format(len(nextTrainFeatures[0])))
        # ans, reward = env.GetAnsAnddReward(action_url, negativeReward=max([-0.5, float(epsilon - 1)]))
        ans, reward = env.GetAnsAnddReward(action_url, negativeReward=0)

        return newTrainFeatures,ans,reward,tmpCoordinate

def train(q, q_target, memory, optimizer):
    # 通過Q網路和target Q網路來構造貝爾曼(Bellman)方程式的誤差，
    # 並只更新Q網路，target Q網路的更新會滯後Q網路
    huber = losses.Huber()
    loss = []
    for i in range(10):  # 訓練10次
        # 從Replay Buffer採樣
        # s, a, r, s_prime, done_mask = memory.sample(batch_size)
        batch_q_a = []
        batch_target =[]
        mini_batch = memory.sample(batch_size)
        with tf.GradientTape() as tape:  # 在 tf GradientTape 區塊一次為一個批次進行預測
            for transition in mini_batch:
                s, a, r, s_prime, done_mask,coor = transition
            # for each_s,each_s_prime in zip(s,s_prime):
                for pre_page, page in zip(s[len(s)-1],s_prime[len(s_prime)-1]):
                    pre_features = list(zip(*pre_page))
                    features = list(zip(*page))
                    # print('pre_feature len', len(pre_features))
                    # print('features len', len(features))
                    ##  update current Q
                    q_features = [np.array(f) for index, f in enumerate(pre_features) if index in [0,1,2]]
                    target_q_features = [np.array(f) for index, f in enumerate(features) if index in [0,1,2]]
                    value, pred_coor = q(q_features)
                    value2, pred_coor2 = q_target(target_q_features)
                    # print('vlue',value)
                    batchvalue = list(value[:, 0])
                    if len(pre_features)==6:
                        for idx, q_value in enumerate(batchvalue):
                            tuple2list = list(s[len(s) - 1][0][idx])
                            tuple2list[5] = q_value
                            list2tuple = tuple(tuple2list)
                            # print(tuple2list)
                            del s[len(s) - 1][0][idx]
                            s[len(s) - 1][0].insert(idx, list2tuple)
                    else:
                        for idx, q_value in enumerate(batchvalue):
                            q_value = tf.convert_to_tensor(q_value, dtype=float)
                            s[len(s) - 1][0][idx] += (q_value,)
                    batchvalue2= list(value2[:, 0])
                    if len(features) == 6:
                        for idx2, t_value in enumerate(batchvalue2):
                            tuple2list2 = list(s_prime[len(s_prime) - 1][0][idx2])
                            tuple2list2[5] = t_value
                            list2tuple2 = tuple(tuple2list2)
                            # print(tuple2list2)
                            del s_prime[len(s_prime) - 1][0][idx2]
                            s_prime[len(s_prime) - 1][0].insert(idx2, list2tuple2)
                    else:
                        for idx, q_value in enumerate(batchvalue2):
                            q_value = tf.convert_to_tensor(q_value, dtype=float)
                            s_prime[len(s_prime) - 1][0][idx] += (q_value,)

                s_q_value = []
                s_t_value = []
                for pre_webpage in s:
                    for pre_page in pre_webpage:
                        new_pre_features = list(zip(*pre_page))
                        for index1, pre_f in enumerate(new_pre_features):
                            if index1 == 5:
                                s_q_value.extend(pre_f)
                for webpage in s_prime:
                    for page in webpage:
                        new_features = list(zip(*page))
                        for index2, f in enumerate(new_features):
                            if index2 == 5:
                                s_t_value.extend(f)
                q_a = s_q_value[a]
                batch_q_a.append([q_a])
                # print('current_q_value',q_a)
                max_q_prime = max(s_t_value)
                target = r + gamma * max_q_prime * done_mask
                batch_target.append([target])
                # print('target',target)
            modelloss = huber(batch_q_a, batch_target)
            # print('loss', modelloss)
            # loss.append(modelloss)
            # print('pred_coor len',len(pred_coor))
            # print('coor',len(coor))
            try:
                coor_loss = tf.reduce_sum(tf.math.sqrt(tf.keras.losses.MSE(pred_coor, coor)))
                multi_loss = modelloss * 0.5 + tf.cast(coor_loss * 0.5, tf.float32)
                print('loss', multi_loss)
                loss.append(multi_loss)
            except Exception:
                continue
        # # 更新網路，使得Q(s,a_t)估計符合Bellman方程式
        grads = tape.gradient(multi_loss, q.trainable_variables)  # 使用tape 計算loss對各個可訓練變數的梯度
        optimizer.apply_gradients(zip(grads, q.trainable_variables))  # 更新梯度的規則由optimizer決定
    train_loss = sum(loss)/len(loss)
    return float(train_loss.numpy())
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

def useGPU(usegpu=True):
    if usegpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[1], 'GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print('GPU error:{}'.format(e))
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
timamap = {}
class Worker(threading.Thread):
    def __init__(self, queue, env, agent, url_idx_queue, types):
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
            url = self.queue.get()
            url_idx = self.url_idx_queue.get()
            ## collect episode
            print('url:{} ,idx:{}'.format(url, url_idx))
            pre_s,pre_coor = self.agent.url_to_feature(url, url_idx, self.env, self.types)
            start_time = datetime.now()
            if pre_s == None:
                continue
            else:
                self.agent.updatetobuffer(pre_s,url,pre_coor,self.types,self.env)
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
        w = Worker(my_queue, envs[i], Agent(model), queue_idx, types)
        workers.append(w)
    for i in range(len(envs)):
        workers[i].start()
        time.sleep(2)
    for i in range(len(envs)):
        workers[i].join()

if __name__ == '__main__':
    useGPU(usegpu=False)
    config = Tool.LoadConfig('config.ini')
    tagtokenize = Tokenize()
    tagtokenize.LoadObj('Preprocess/tagtokenizemore')
    bertlayer = hub.KerasLayer(config.get('BertTokenize', 'bert_path'), trainable=False)
    if os.path.isfile(config.get('Main', 'url_feature_file_path_new') + '.txt'):
        url_featuresMap = Tool.LoadObj(config.get('Main', 'url_feature_file_path_new'))
    else:
        url_featuresMap = {}

    traindf = pd.read_csv("./dataset_google_search/google_search_rank_train - contain news.csv")
    validation = pd.read_csv("./dataset_google_search/google_search_rank_vali - contain_news.csv")
    pre_totalDF = pd.concat([traindf, validation])
    totalDF = pre_totalDF.dropna(subset=['Event source page'])
    trainwebsiteInfo = traindf.groupby("Homepage")
    validationwebsiteInfo = validation.groupby("Homepage")
    trainlist = list(trainwebsiteInfo.groups)[0:40]
    valilist = list(validationwebsiteInfo.groups)[0:20]
    # Hyperparameters
    learning_rate = 0.0002
    gamma = 0.9
    buffer_limit = 10000
    batch_size = 16
    memory = ReplayBuffer()  # 創建Replay Buffer
    vali_memory = ValiReplayBuffer()

    # Model
    tagdimension = 36
    q = Qnet(tagdimension)  # 創建Q網路

    model = q.model()
    agent = Agent(model)
    target_model = q.model()

    model.summary()

    for src, dest in zip(model.variables, target_model.variables):
        dest.assign(src)  # target Q網路權重来自Q
    print_interval = 5
    numofthread = 2
    savemodelname = "pretrain_dqn_without_top3_fixreward"
    savemodelname_dir = 'pretrain_dqn_without_top3_fixreward'

    envs = []
    for i in range(numofthread):
        envs.append(PolicyDeepthEnvFixCoordinate_singlefile.Env(totalDF, negativeReward=-0.1, rewardweight=0.1))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    total_loss = 0.0
    savelastpre = None
    for n_epi in range(1):  # 訓練次數
        # epsilo機率會從8%到1%衰減，越到後面越使用Q值最大的動作
        # epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 2))
        epsilon = max(0.01, 1 - 0.05 * (n_epi / 2))
        unit_cost = 0.4 + 0.05*(n_epi/10)
        # UseThreadRunData(trainlist[0:1], envs, list(range(1, 2)), 'train')
        # loss = train(model, target_model, memory, optimizer)
        #global_lock = threading.Lock()
        for i in range(len(trainlist) // 10):
            UseThreadRunData(trainlist[10*i:10*i+10], envs, list(range(10*i+1,10*i+11)),'train')
            if memory.size() > 50:  # Replay Buffer只要大於2,000就可以訓練
                loss = train(model, target_model, memory, optimizer)
                total_loss += loss
        epoch_loss = total_loss/4
        total_loss = 0.0
        print('epoch loss:', epoch_loss)
        savePerloss(epoch_loss)
        if n_epi % print_interval == 0 and n_epi != 0:
            for src, dest in zip(model.variables, target_model.variables):
                dest.assign(src)  # target Q網路權重值来自Q網路
            print("# of episode :{},buffer size : {}, " \
                  "epsilon : {:.1f}%" \
                  .format(n_epi,memory.size(), epsilon * 100))
            UseThreadRunData(valilist, envs, list(range(1, 21)), 'validation')
            model.save('{}/{}_epoch{}.h5'.format(savemodelname_dir, savemodelname, n_epi), save_format="tf")
            vali_mini_batch = vali_memory.pickVali()
            done_ans = []
            for transition in vali_mini_batch:
                _, _, _, _, done_mask,_ = transition
                if done_mask == 0:
                    ans = 1
                    done_ans.append(ans)
                else:
                    ans = 0
                    done_ans.append(ans)
            vali_p_1 = precision_score(done_ans,[1]*len(done_ans))
            print('vali_p_1:',vali_p_1)
            if savelastpre == None:
                savelastpre = vali_p_1
                model.save('{}/{}_choose.h5'.format(savemodelname_dir, savemodelname), save_format="tf")
            if vali_p_1 >= savelastpre:
                print('savelast')
                savelastpre = vali_p_1
                model.save('{}/{}_choose.h5'.format(savemodelname_dir, savemodelname), save_format="tf")
            vali_memory.resetbuffer()
        Tool.SaveObj(config.get('Main', 'url_feature_file_path_new'), url_featuresMap)
    for env_thread in envs:
        env_thread.webQuit()

