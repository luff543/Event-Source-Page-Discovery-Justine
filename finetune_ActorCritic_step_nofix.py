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
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from src import Tool,PolicyDeepthEnvFixCoordinate_singlefile
from Preprocess.Tokenize import Tokenize
from func_timeout import func_timeout, FunctionTimedOut
from datetime import datetime, timedelta
from multiprocessing import Queue
from sklearn.metrics import precision_score
import requests

tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

class ReplayBuffer():
    # Replay Buffer
    def __init__(self):
        self.states_actor = []
        self.states_critic = []
        self.actions = []
        self.actions_step = []
        self.rewards = []
        self.next_state_critic = []

    def store(self, states_actor, states_critic, action, actions_step, reward,next_state_critic):
        self.states_actor.append(states_actor)
        self.states_critic.append(states_critic)
        self.actions.append(action)
        self.actions_step.append(actions_step)
        self.rewards.append(reward)
        self.next_state_critic.append(next_state_critic)
    def clear(self):
        self.states_actor = []
        self.states_critic = []
        self.actions = []
        self.actions_step = []
        self.rewards = []
        self.next_state_critic = []
class TotalStep():
    def __init__(self):
        self.click_step = []
    def store(self,click_step):
        self.click_step.append(click_step)
    def clear(self):
        self.click_step = []

class Actor(keras.Model):
    def __init__(self,tagdimension):
        # 創建Q網路，輸入為狀態向量，輸出為動作的Q值
        super(Actor, self).__init__()
        self.tagdimension = tagdimension
        self.bilstm_first = layers.Bidirectional(layers.LSTM(tagtokenize.GetDictLength(), return_sequences=True),
                                                 name="tag_path1")
        self.bilstm_two = layers.Bidirectional(layers.LSTM(tagdimension // 2, return_sequences=True), name="tag_path2")
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
        # coor_output = self.coordinate_output(tagembedding)
        atembedding = self.bert_bilstm(atembedding)
        atembedding = self.bert_bilstm_two(atembedding)
        atembedding = tf.math.reduce_mean(atembedding, axis=1)
        tag = tf.math.multiply(tagembedding, index)
        tag = tf.math.reduce_sum(tag, axis=1)
        concat = layers.Concatenate()([atembedding, tag])
        concatlayer = self.concat_first(concat)
        concatlayer = self.concat_two(concatlayer)
        anchor_output = self.agent_output(concatlayer)
        return anchor_output
        # return anchor_output, coor_output
    def model(self):
        x = [tf.keras.Input(shape=(None,768)), tf.keras.Input(shape=(None, tagtokenize.GetDictLength())),
             tf.keras.Input(shape=(None, self.tagdimension))]
        return tf.keras.Model(inputs=x, outputs=self.call(x))

class Critic(keras.Model):
    def __init__(self,tagdimension):
        # 創建Q網路，輸入為狀態向量，輸出為動作的Q值
        super(Critic, self).__init__()
        self.tagdimension = tagdimension
        self.bert_title = keras.layers.Dense(512, activation='relu')
        self.bert_title_2 = keras.layers.Dense(512, activation='relu')
        self.bert_title_3 = keras.layers.Dense(128, activation='relu')
        self.bert_at = keras.layers.Dense(512, activation='relu')
        self.bert_at_2 = keras.layers.Dense(512, activation='relu')
        self.bert_at_3 = keras.layers.Dense(128, activation='relu')
        # self.coor = keras.layers.Dense(128, activation='relu')
        # self.coor_2 = keras.layers.Dense(128, activation='relu')
        # self.coor_3 = keras.layers.Dense(64, activation='relu')
        self.bilstm_first = layers.Bidirectional(layers.LSTM(tagtokenize.GetDictLength(), return_sequences=True),
                                                 name="tag_path1")
        self.bilstm_two = layers.Bidirectional(layers.LSTM(tagdimension // 2, return_sequences=True), name="tag_path2")
        self.concat_first = layers.Dense(64, activation='relu')
        self.concat_two = layers.Dense(32, activation='relu')
        self.agent_output = layers.Dense(1, kernel_initializer='he_normal',activation='linear')
        # self.coordinate_output = layers.Dense(12)

    def call(self, x):
        titleembedding = x[0]
        atembedding = x[1]
        # coordinate = x[2]
        tagpath = x[2]
        index = x[3]

        titleembedding = self.bert_title(titleembedding)
        titleembedding = self.bert_title_2(titleembedding)
        titleembedding = self.bert_title_3(titleembedding)
        atembedding = self.bert_at(atembedding)
        atembedding = self.bert_at_2(atembedding)
        atembedding = self.bert_title_3(atembedding)
        # coordinate = self.coor(coordinate)
        # coordinate = self.coor_2(coordinate)
        # coordinate = self.coor_3(coordinate)
        # coordinate = tf.math.reduce_mean(coordinate, axis=1)
        # concat = layers.Concatenate()([titleembedding,atembedding, coordinate])
        tagpath = tf.math.reduce_mean(tagpath, axis=0)
        index = tf.math.reduce_mean(index, axis=0)
        tagembedding = self.bilstm_first(tagpath)
        tagembedding = self.bilstm_two(tagembedding)
        tag = tf.math.multiply(tagembedding, index)
        tag = tf.math.reduce_sum(tag, axis=1)
        tag = tf.math.reduce_mean(tag,axis=0)
        tag = tf.reshape(tag,[1,self.tagdimension])
        concat = layers.Concatenate()([titleembedding, atembedding, tag])
        concatlayer = self.concat_first(concat)
        concatlayer = self.concat_two(concatlayer)
        anchor_output = self.agent_output(concatlayer)
        return anchor_output
        # return anchor_output, coor_output
    def model(self):
        x = [tf.keras.Input(shape=(768,)), tf.keras.Input(shape=(768,)),
             tf.keras.Input(shape=(None, None,tagtokenize.GetDictLength())),
             tf.keras.Input(shape=(None, None,self.tagdimension))]
        return tf.keras.Model(inputs=x, outputs=self.call(x))
class Agent():
    def __init__(self, actor,critic):
        #### Fit Variable
        self.actor = actor
        self.critic = critic
    def sample_action(self, target_url,state_features, current_url, data, top3_data,actionClickHref,epsilon,step,
                      types,useprestate=False,random_give=False,save_model_output_log=False):
        url = []
        AT = []
        null = []
        pred_value = []
        pageActionIndexs = []
        # print('len state',len(state_features))
        if not useprestate:
            # for batch in state_features[len(state_features) - 1]:
            for batch in state_features:
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
                out = self.actor(model_features)
                pred_value += list(out[:, 0].numpy())
            pageActionIndexs = list(tf.argsort(pred_value, axis=-1, direction='DESCENDING').numpy())
            try:
                data['url'].extend(list(url[0]))
                data['q(s,a)'].extend(pred_value)
                data['AT'].extend(list(AT[0]))
                data['step'].extend(list([step + 1] * len(AT[0])))
                data['index'].extend(list(range(0, len(AT[0]))))
                # data['flag'].extend(np.zeros(len(list(url[0]))))
                # print(data)
            except Exception:
                data['url'].extend(null)
                data['q(s,a)'].extend(null)
                data['AT'].extend(null)
                data['step'].extend(null)
                data['index'].extend(null)
            try:
                top3_data['url'].extend(list(url[0]))
                top3_data['q(s,a)'].extend(pred_value)
                top3_data['AT'].extend(list(AT[0]))
                if step == 0:
                    top3_data['index'].extend(list(range(0,len(data['url']))))
                else:
                    top3_data['index'].extend(list(range(len(data['url'])-len(pageActionIndexs),len(data['url']))))
                top3_data['step'].extend(list([step + 1] * len(AT[0])))
                # data['flag'].extend(np.zeros(len(list(url[0]))))
                # print(data)
            except Exception:
                top3_data['url'].extend(null)
                top3_data['q(s,a)'].extend(null)
                top3_data['AT'].extend(null)
                top3_data['index'].extend(null)
                top3_data['step'].extend(null)
            # print('top_data_index',top3_data['index'])
            # print('first top_3_data',len(top3_data['q(s,a)']))
        else:
            # for batch in state_features[len(state_features) - 1]:
            for batch in state_features:
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
                out = self.actor(model_features)
                pred_value += list(out[:, 0].numpy())
            pageActionIndexs = list(tf.argsort(pred_value, axis=-1, direction='DESCENDING').numpy())
            try:
                top3_data['url'].extend(list(url[0]))
                top3_data['q(s,a)'].extend(pred_value)
                top3_data['AT'].extend(list(AT[0]))
                if step == 0:
                    top3_data['index'].extend(list(range(0,len(data['url']))))
                else:
                    top3_data['index'].extend(list(range(len(data['url'])-len(pageActionIndexs),len(data['url']))))
                top3_data['step'].extend(list([step + 1] * len(AT[0])))
                # data['flag'].extend(np.zeros(len(list(url[0]))))
                # print(data)
            except Exception:
                top3_data['url'].extend(null)
                top3_data['q(s,a)'].extend(null)
                top3_data['AT'].extend(null)
                top3_data['index'].extend(null)
                top3_data['step'].extend(null)
        # print('first top_3_data', len(top3_data['q(s,a)']))
        if step == 0:
            top3_correct_index = list(pageActionIndexs[:3])
        else:
            top3_correct_index = [idx + (len(data['url']) - len(pageActionIndexs)) for idx in
                                  list(pageActionIndexs[:3])]
        listlactionIndexs = list(tf.argsort(top3_data['q(s,a)'], axis=-1, direction='DESCENDING').numpy())
        coin = random.random()
        # # 策略改進：e-Greedy方法
        last = False
        if random_give == True:
            new_action_index = random.randint(0, len(data['AT']) - 1)
            clickAT = data['AT'][new_action_index]
            clickhref = data['url'][new_action_index]
            original_index = data['index'][new_action_index]
            original_index_step = data['step'][new_action_index]
        else:
            if coin < epsilon:
                # 當前機率小於epsilon門檻值，隨機存取
                count_idx = 1
                action_idx = random.randint(0, len(top3_data['AT']) - 1)
                while top3_data['url'][action_idx] in actionClickHref and last == False:
                    if len(top3_data['url'][action_idx]) - count_idx <= 1:
                        action_idx = random.randint(0, len(top3_data['AT']) - 1)
                        last = True
                    else:
                        action_idx = random.randint(0, len(top3_data['AT']) - 1)
                        last = False
                    count_idx += 1

            else:  # 當前機率大於等於epsilon門檻值，選擇Q值最大的動作做
                # action_idx = int(tf.argmax(data['q(s,a)']))
                action_idx = listlactionIndexs[0]
                count_idx = 1
                while top3_data['url'][action_idx] in actionClickHref and last == False:
                    if len(top3_data['url'][action_idx]) - count_idx <= 1:
                        action_idx = random.randint(0, len(top3_data['AT']) - 1)
                        last = True
                    else:
                        action_idx = listlactionIndexs[count_idx]
                        last = False
                    count_idx += 1
            new_action_index = top3_data['index'][action_idx]
            clickAT = data['AT'][new_action_index]
            clickhref = data['url'][new_action_index]
            original_index = data['index'][new_action_index]
            original_index_step = data['step'][new_action_index]
        # print('now href:{}, actionIndex:{}, clickAT:{}, clickhref:{}'.format(current_url,action_idx, clickAT, clickhref))
        # print('len data:{},new_action_index:{},original_index:{},original_index_step:{}'.format(
        #     len(data['AT']), new_action_index,original_index,original_index_step))
        del top3_data['url'][len(top3_data['url']) - len(pageActionIndexs):]
        del top3_data['q(s,a)'][len(top3_data['q(s,a)']) - len(pageActionIndexs):]
        del top3_data['AT'][len(top3_data['AT']) - len(pageActionIndexs):]
        del top3_data['index'][len(top3_data['index']) - len(pageActionIndexs):]
        del top3_data['step'][len(top3_data['step']) - len(pageActionIndexs):]
        if not useprestate:
            top3_data['url'].extend([url[0][i] for i in pageActionIndexs[:3]])
            top3_data['q(s,a)'].extend([pred_value[i] for i in pageActionIndexs[:3]])
            top3_data['AT'].extend([AT[0][i] for i in pageActionIndexs[:3]])
            top3_data['index'].extend(top3_correct_index)
            top3_data['step'].extend(list([step + 1] * 3))
        # print('second top_3_data', len(top3_data['q(s,a)']))
        # print('top3_data',top3_data)
        # save model output log
        if save_model_output_log:
            candidate_click_anchors = []
            for idx in listlactionIndexs:
                click_anchor = {}
                click_anchor['prob'] = round((data['q(s,a)'][idx]).astype(np.float64),5)
                click_anchor['url'] = data['url'][idx]
                click_anchor['step'] = data['step'][idx]
                candidate_click_anchors.append(click_anchor)
                # logging.info('current url:{},priority queue:{}'.format(clickhref, candidate_click_anchors))
            data = {
                'target url': target_url,
                'current url': current_url,
                'priority queue': candidate_click_anchors
            }
            path = 'log'
            if not os.path.exists(path):
                os.makedirs(path)
            # with open('log/candidate_click_anchors.json', 'a', newline='', encoding='utf-8') as file:
            with open('log/candidate_click_anchors_{}.json'.format('2023-04-11'), 'a', encoding='utf-8') as file:
                file.writelines('\n')
                json.dump(data, file, ensure_ascii=False)
        return original_index,original_index_step,clickhref,clickAT

    def saveReward(self,stepreward,cost,t,types):
        path = 'precision_result/{}/reward'.format(savemodelname)
        if not os.path.exists(path):
            os.makedirs(path)
            with open('{}/{}/reward/reward_{}.csv'.format('precision_result', savemodelname, types), 'w',
                      newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['stepreward', 'cost', 'step'])
        with open('{}/{}/reward/reward_{}.csv'.format('precision_result', savemodelname, types), 'a',
                          newline='') as csvfile:
            if types == 'train':
                writer = csv.writer(csvfile)
                writer.writerow([stepreward, cost, t])
            else:
                writer = csv.writer(csvfile)
                writer.writerow([stepreward, cost, t])
    def url_to_feature(self,url,url_idx,env,types):
        previousTrainFeatures = []
        critic_state = []
        try:
            if url in url_featuresMap:
                # print('url in url featureMap:', url)
                page_title, AT, HREFS, tagpaths, coordinate = url_featuresMap[url]['pagetitle'], \
                                                              url_featuresMap[url]['attext'], \
                                                              url_featuresMap[url]['hrefs'], \
                                                              url_featuresMap[url]['tagpaths'], \
                                                              url_featuresMap[url]['coordinate']
            else:
                AT, HREFS, tagpaths, coordinate = func_timeout(180, env.reset, args=(url, None, types, True))
                coordinate = list(coordinate)
                page_title = env.GetPageTitle()
                # print(len(AT),len(HREFS),len(tagpaths),len(coordinate))
                url_featuresMap[url] = {'pagetitle': page_title, 'attext': AT, 'hrefs': HREFS, 'tagpaths': tagpaths,
                                        'coordinate': coordinate, 'time': str(datetime.now())}
        except FunctionTimedOut:
            print('this url cannot get:{}'.format(url))
            # return None
            return None, None
        if len(AT) == 0 or len(AT) == 1:
            print('AT len is 0 or 1, url: {} '.format(url))
            # return None
            return None, None
        assert len(AT) == len(HREFS)

        atembedding, neighborcoordinate, tagid, tagindex = Tool.ProcessFeatures_sequence_output(AT, coordinate, tagpaths,
                                                                                tagtokenize, bertlayer,
                                                                                config)  # 處理特徵的fumction
        pagetitle_emb, clickAT_emb = Tool.ProcessFeatures_critic([page_title], [page_title], bertlayer, config)
        if tagindex.shape[0] != tagindex.shape[1]:
            return None, None
        previousTrainFeatures.append(list(zip(atembedding, tagid, tagindex, HREFS, AT)))
        # return atembedding
        # newcoordinate = np.array(coordinate)[:, [0,3]]
        critic_state.extend([np.array(pagetitle_emb[0]), np.array(clickAT_emb[0]), np.array(tagid), np.array(tagindex)])
        return previousTrainFeatures,critic_state
    def updatetobuffer(self,url,url_idx,types,env):
        for batch in range(1):
            previousTrainFeatures,critic_state = self.url_to_feature(url,url_idx,env,types)
            if previousTrainFeatures == None:
                break
            # s = [previousTrainFeatures]
            s = previousTrainFeatures
            s_critic = critic_state
            target_url = url
            current_url = url
            log_model_output = False
            data = {'url': [], 'q(s,a)': [], 'AT': [], 'step': [], 'index':[]}
            top3_data = {'url': [], 'q(s,a)': [], 'AT': [], 'index':[],'step': []}
            alreadyclickurl = []
            total_click_url = []
            useprestate = False
            score = 0.0
            # score_2 = 0.0
            total_action_idx = []
            s_prime_count = 0
            t = 0
            # for t in range(3):  # 一個回合最大時間戳 (updating N times)
            # do_step = True
            random_give = False
            # while do_step:
            assent = 1
            s_prime_count = 0
            while assent > 0:
                # 根據當前Q網路提取策略，並改進策略
                if s_prime_count >= 2:
                    random_give = True

                action_idx, action_idx_step,action_url, action_at = self.sample_action(target_url, s, current_url, data,
                                                                  top3_data,alreadyclickurl, epsilon, t,
                                                                  types,useprestate, random_give,log_model_output)
                alreadyclickurl.append(action_url)
                s_prime, s_critic_prime, r = self.step(action_url, action_at, env)
                if s_prime == None:
                    s = s
                    s_critic = s_critic
                    current_url = current_url
                    useprestate = True
                    s_prime_count += 1
                    continue
                total_click_url.append(action_url)
                total_action_idx.append(action_idx)
                print('click_url:{}, reward:{}, AT:{}'.format(action_url, r, action_at))
                memory.store(s, s_critic, action_idx,action_idx_step, r, s_critic_prime)
                assent = assent - cost + r

                s = s_prime
                current_url = action_url
                score += r  # 紀錄總回報
                # score_2 += done
                s_critic = s_critic_prime
                useprestate = False
                random_give = False
                if assent <= 0 or t == 4:  # 回合結束
                    data.clear()
                    top3_data.clear()
                    break
                t += 1
            print('types:{}, target url:{} ,,total step click_url:{}'.format(types,target_url
                                                                     ,total_click_url))
            self.saveReward(score, cost,t+1, types)
            memory_step.store(t+1)
        # self.saveReward(score, score_2, p_1, types)
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
    def step(self,action_url,clickAT,env):
        nextTrainFeatures = []
        nextCriticFeatures = []
        try:
            if action_url in url_featuresMap:
                tmpPagetitle, tmpAT, tmpHREFS, tmptagpath, tmpCoordinate = url_featuresMap[action_url]['pagetitle'], \
                                                                           url_featuresMap[action_url]['attext'], \
                                                                           url_featuresMap[action_url]['hrefs'], \
                                                                           url_featuresMap[action_url]['tagpaths'], \
                                                                           url_featuresMap[action_url][
                                                                               'coordinate']
            else:
                tmpAT, tmpHREFS, tmptagpath, tmpCoordinate = func_timeout(240, env.step, args=(action_url, True))
                # step to next step 跳轉到現在要點擊的herf
                tmpCoordinate = list(tmpCoordinate)
                tmpPagetitle = env.GetPageTitle()
                # print(len(AT),len(HREFS),len(tagpaths),len(coordinate))
                url_featuresMap[action_url] = {'pagetitle': tmpPagetitle, 'attext': tmpAT, 'hrefs': tmpHREFS,
                                               'tagpaths': tmptagpath, 'coordinate': tmpCoordinate,
                                               'time': str(datetime.now())}
        except FunctionTimedOut:
            return None,None,None
        if len(tmpAT) == 0 or len(tmpAT) == 1:
            print('tmpAT len is 0 or 1, action url: {} '.format(action_url))
            return None,None,None
            # return None, None
        if len(tmpAT) >= 500:
            print('tmpAT len over 500, action url: {} '.format(action_url))
            return None,None,None
        assert len(tmpAT) == len(tmpHREFS)
        tmpatembedding, tmpCoordinate, tmptagid, tmptagindex = Tool.ProcessFeatures_sequence_output(tmpAT, tmpCoordinate,
                                                                                    tmptagpath, tagtokenize,
                                                                                    bertlayer, config)
        if tmptagindex.shape[0] != tmptagindex.shape[1]:
            return None,None,None
        else:
            nextTrainFeatures.append(list(zip(tmpatembedding, tmptagid, tmptagindex, tmpHREFS, tmpAT)))

        tmpPagetitle_emb, tmpAT_emb = Tool.ProcessFeatures_critic([tmpPagetitle], [clickAT], bertlayer, config)
        # newtmpCoordinate = np.array(tmpCoordinate)[:, [0,3]]
        nextCriticFeatures.extend([np.array(tmpPagetitle_emb[0]), np.array(tmpAT_emb[0]), np.array(tmptagid),
                                   np.array(tmptagindex)])

        # newTrainFeatures = [*TrainFeatures,nextTrainFeatures]
        # newCriticFeatures = [TrainCriticFeatures,nextCriticFeatures]
        # print('len nextTrainFeatures[0]:{} '.format(len(nextTrainFeatures[0])))
        # ans, reward = env.GetAnsAnddReward(action_url, negativeReward=max([-0.5, float(epsilon - 1)]))
        # ans, reward = env.GetAnsAnddReward(action_url, negativeReward=0)
        # ans, reward = env.GetAnsAndDiscountedReward(action_url, pre_rewards,negativeReward=max([-0.5, float(epsilon - 1)]))
        try:
            reward = func_timeout(300, self.GetESPS, args=(action_url,))
        except FunctionTimedOut:
            print("程式執行時間超過限制")
            reward = 0
        return nextTrainFeatures,nextCriticFeatures,reward

def actor_output(actor,s):
    for batch in s:
        features = list(zip(*batch))
        # print('actor output feature len',len(features))
        model_features = [np.array(f) for index, f in enumerate(features) if index in [0, 1, 2]]
        out = actor(model_features)
        # print('actor out', out)
        return out
def critic_output(critic,s):
    titleembedding, atembedding,tagpath,tagid = zip(s)
    # out = critic([[np.array(titleembedding), np.array(atembedding),np.array(tagpath),np.array(tagid)]])
    # print('critic out', out)
    out = critic([[np.array(titleembedding), np.array(atembedding),np.array(tagpath),np.array(tagid)]])
    return out
def epoch_compute_loss(actor,critic,
                 memory,memory_step,
                 gamma=0.99):
    total_loss = []
    print('len(memory_step.click_step)',len(memory_step.click_step))
    #print('memory.actions',memory.actions)

    change_step_memory = []
    newstep = 0
    for click_step in memory_step.click_step:
        newstep += click_step
        change_step_memory.append(newstep)
    for i,click_step in enumerate(change_step_memory):
        if i == 0:
            start = 0
        else:
            start = change_step_memory[i-1]
        buffer_actor_state = memory.states_actor[start:click_step]
        buffer_critic_state = memory.states_critic[start:click_step]
        buffer_reward = memory.rewards[start:click_step]
        buffer_action = memory.actions[start:click_step]
        buffer_action_step = memory.actions_step[start:click_step]
        try:
            buffer_next_critic_state = memory.next_state_critic[click_step - 1]
            buffer_done = memory.rewards[click_step - 1]
        except:
            continue
        try:

            # with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                if buffer_done > 0.6:
                    reward_sum = 0.  # 终止状态的v(终止)=0
                else:
                    reward_sum = critic_output(critic, buffer_next_critic_state)[0]
                # print('reward_sum',reward_sum)
                # 統計折扣reward
                discounted_rewards = []
                for reward in buffer_reward[::-1]:  # reverse buffer r, memory.rewards[::-1] == 倒序
                    reward_sum = reward + gamma * reward_sum
                    discounted_rewards.append(reward_sum)
                discounted_rewards.reverse()
                # 獲取狀態的Pi(a|s)和v(s)
                logits = []
                values = []
                for i,state_actor in enumerate(buffer_actor_state):
                    logits.append(actor_output(actor, state_actor))
                for i,state_critic in enumerate(buffer_critic_state):
                    values.append(critic_output(critic,state_critic)[0])
                # values = tf.reshape(values,[len(values),1])
                advantage = tf.constant(np.array(discounted_rewards),
                                        dtype=tf.float32) - values
                # print('advantage',advantage)
                # # Critic loss
                value_loss = advantage ** 2
                # # Policy loss
                policy_loss = []
                for i, (action, action_step, each_advantage, each_logits) in enumerate(
                        zip(buffer_action, buffer_action_step,
                            advantage, logits)):
                    if action_step == i+1:
                        each_logits = tf.reshape(each_logits, [len(each_logits)])
                        policy = tf.nn.softmax(each_logits)
                    else:
                        each_logits = tf.reshape(logits[action_step-1], [len(logits[action_step-1])])
                        policy = tf.nn.softmax(each_logits)
                    each_policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=action, logits=policy)
                    # # 計算policy loss，這邊不會算 critic
                    each_policy_loss = each_policy_loss * tf.stop_gradient(each_advantage)
                    # # Entropy Bonus
                    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy,
                                                                      logits=each_logits)
                    each_policy_loss = each_policy_loss - 0.01 * entropy
                    #if np.isnan(each_policy_loss):
                        #continue
                    #else:
                    policy_loss.extend(each_policy_loss)
                # 合併loss
                policy_loss = tf.reshape(policy_loss, [len(policy_loss),1])
                value_loss = value_loss * 0.5
                loss = tf.reduce_mean((value_loss + policy_loss))
                #print('loss:{}, value loss:{}, policy_loss:{}'.format(loss,value_loss,policy_loss))
                total_loss.append(loss)
            actor_grads = actor_tape.gradient(policy_loss,actor.trainable_variables)
            # actor_clip_grads, _ = tf.clip_by_global_norm(actor_grads, 5.0)
            optimizer_actor = optimizers.Adam(learning_rate=2e-4)
            optimizer_actor.apply_gradients(zip(actor_grads,actor.trainable_variables))
            # optimizer_actor.apply_gradients((grad, var) for (grad, var) in zip(actor_clip_grads,
            #                                                        actor.trainable_variables) if grad is not None)
            critic_grads = critic_tape.gradient(value_loss, critic.trainable_variables)
            optimizer_critic = optimizers.Adam(learning_rate=1e-3)
            optimizer_critic.apply_gradients(zip(critic_grads, critic.trainable_variables))
        except Exception:
            continue
    final_loss = sum(total_loss)/len(total_loss)
    print('pre 10 url loss',float(final_loss.numpy()))
    # savePerloss(float(final_loss.numpy()))
    return float(final_loss.numpy())

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
            start_time = datetime.now()
            self.agent.updatetobuffer(url, url_idx, self.types, self.env)
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
        w = Worker(my_queue, envs[i], Agent(actor,critic), queue_idx, types)
        workers.append(w)
    for i in range(len(envs)):
        workers[i].start()
        time.sleep(2)
    for i in range(len(envs)):
        workers[i].join()
def GetLeaveTime():
    now_time = datetime.now()
    leave_time = now_time - timedelta(days=7)
    return leave_time
def delOldData():
    if os.path.isfile(config.get('Main', 'url_feature_file_path_title')+'.txt'):
        url_featuresMap = Tool.LoadObj(config.get('Main', 'url_feature_file_path_title'))
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
if __name__ == '__main__':
    useGPU(usegpu=True)
    config = Tool.LoadConfig('config.ini')
    tagtokenize = Tokenize()
    tagtokenize.LoadObj('Preprocess/tagtokenizemore')
    bertlayer = hub.KerasLayer(config.get('BertTokenize', 'bert_path'), trainable=False)
    if os.path.isfile(config.get('Main', 'url_feature_file_path_title_finetune') + '.txt'):
        url_featuresMap = Tool.LoadObj(config.get('Main', 'url_feature_file_path_title_finetune'))
    else:
        url_featuresMap = {}
    #url_featuresMap = delOldData()
    global_lock = threading.Lock()
    traindf = pd.read_csv("./dataset_google_search/google_search_rank_train - contain news.csv")
    validation = pd.read_csv("./dataset_google_search/google_search_rank_vali - contain_news.csv")
    pre_totalDF = pd.concat([traindf, validation])
    totalDF = pre_totalDF.dropna(subset=['Event source page'])
    fintunedf = pd.read_csv("./dataset_google_search/fintunetraindata.csv")
    trainlist = list(fintunedf['url'])
    preactormodel = './stepnofixmodel/Actor_pretrain_A2C_tagpath_step_nofix_top3_cost0.4_critic0.5_fixreward_choose.h5'
    precriticmodel = './stepnofixmodel/Critic_pretrain_A2C_tagpath_step_nofix_top3_cost0.4_critic0.5_fixreward_choose.h5'

    # Hyperparameters
    memory = ReplayBuffer()  # 創建Replay Buffer
    memory_step = TotalStep()
    load_pretrain_model = True
    # Model
    if load_pretrain_model:
        actortmodel = tf.keras.models.load_model('{}'.format(preactormodel))
        criticmodel = tf.keras.models.load_model('{}'.format(precriticmodel))
        actor = actortmodel
        critic = criticmodel
    else:
        tagdimension = 36
        Actor = Actor(tagdimension)
        Critic = Critic(tagdimension)
        actor = Actor.model()
        critic = Critic.model()
    agent = Agent(actor,critic)
    actor.summary()
    critic.summary()


    print_interval = 5
    numofthread = 1
    savemodelname = "finetune_A2C_nofixstep_eplision_0.7"
    savemodelname_dir = 'finetune_A2C_nofixstep_eplision_0.7'

    envs = []
    for i in range(numofthread):
        envs.append(PolicyDeepthEnvFixCoordinate_singlefile.Env(totalDF, negativeReward=-0.1, rewardweight=0.1))
    total_loss = 0.0
    for n_epi in range(1):  # 訓練次數
        # global_lock = threading.Lock()
        cost = 0.4 + 0.05 * (n_epi / 10)
        #UseThreadRunData(trainlist[0:5], envs, list(range(1, 6)), 'train')
        #epoch_compute_loss(actor, critic, memory, memory_step)

        for i in range(len(trainlist) // 50):
            # epsilon = max(0.01, 1 - (0.8*i)//2)
            epsilon = 0.3
            UseThreadRunData(trainlist[50 * i:50 * (i + 1)], envs, list(range(50 * i + 1, 50 * i + 51)), 'train')
            loss = epoch_compute_loss(actor,critic,memory,memory_step)
            savePerloss(loss)
            memory.clear()
            memory_step.clear()
            actor.set_weights(actor.get_weights())
            critic.set_weights(critic.get_weights())
            actor.save('{}/Actor_{}_epoch{}.h5'.format(savemodelname_dir, savemodelname, i), save_format="tf")
            critic.save('{}/Critic_{}_epoch{}.h5'.format(savemodelname_dir, savemodelname, i), save_format="tf")

        Tool.SaveObj(config.get('Main', 'url_feature_file_path_title_finetune'), url_featuresMap)
    for env_thread in envs:
        env_thread.webQuit()

