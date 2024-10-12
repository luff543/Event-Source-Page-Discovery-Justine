import configparser
import pickle
import json
import copy
import numpy as np
import tensorflow as tf
try:
    from src import StringTool
except:
    import StringTool
import random
def LoadConfig(confilefile):
    config = configparser.ConfigParser()
    config.read(confilefile)
    return config

def SaveList(filename,listinfo):
    with open(f"{filename}.txt", "wb") as fp:  
        pickle.dump(listinfo, fp)
        
def LoadList(filename):
    with open(f"{filename}.txt", "rb") as fp:
        return pickle.load(fp)
    
    
def SaveObj(filename, data):
    # f = open(filename + '.txt', 'w',encoding='utf-8')
    # print("filename:", filename)
    # print("saveOj_data:", data)
    # json.dump(data, f, ensure_ascii=False)
    with open(filename + '.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n')
        json.dump(data, f, ensure_ascii=False)
    
def LoadObj(filename):
    f = open(filename + '.txt', 'r', encoding="utf-8")
    # print(type(f))
    return json.load(f)
    # return f.read()

def ConvertToTwoDimensionIndex(index, convertedlist):
        i = 0
        while index >= len(list(convertedlist[i])):
            index -= len(list(convertedlist[i]))
            i += 1
        return i, index
    
def GetActionIndex(model,allfeatures, hrefindex, trainfeatureindexs, alreadyClicks, randomjump,epison=0):
    if randomjump:
        if int(epison) == int(0):
            assert IndexError("Epison is zero, but random jump")
    predvalue = []
    # print('len(allfeatures),len(allfeatures[0])',len(allfeatures),len(allfeatures[0]))
    for batch in copy.deepcopy(allfeatures):
        features = zip(*batch)
        features = [np.array(f) for index, f in enumerate(features) if index in trainfeatureindexs]
        if len(features)==0:
            continue
        # atpredd , _ = model(features) # [[0.xx],[0.xx],...]
        atpredd, _ = model(features)  # [[0.xx],[0.xx],...]
        predvalue += list(atpredd[:,0])  # [<tf.Tensor: shape=(), dtype=float32, numpy=0.xxx>,<>,...]
    originalactionIndexs = list(tf.argsort(predvalue,axis=-1,direction='DESCENDING').numpy())
    # print('originalactionIndexs',originalactionIndexs)
    leaveATIndex = list(originalactionIndexs[1:11])
    if(len(originalactionIndexs)>1):
        if(predvalue[originalactionIndexs[0]] == predvalue[originalactionIndexs[1]]):
            sameindex = []
            maxnumber = predvalue[originalactionIndexs[0]]
            for i in originalactionIndexs:
                if(maxnumber == predvalue[i]):
                    sameindex.append(i)
                else:
                    break
            originalactionIndex = originalactionIndexs[random.sample(sameindex,1)[0]]
        else:
            originalactionIndex = originalactionIndexs[0]
    else:
        originalactionIndex = originalactionIndexs[0]

    actionIndex = originalactionIndex#np.random.choice(np.arange(0, len(predvalue)), p=p)#originalactionIndex
    if(randomjump):
        if(random.randint(0,10)/10<=epison):
            actionIndex = random.sample(list(range(0,len(predvalue))),1)[0]
    firstdi, secdi = ConvertToTwoDimensionIndex(actionIndex, allfeatures)
    if(actionIndex in leaveATIndex):
        leaveATIndex.remove(actionIndex)
    for index in leaveATIndex:
        if(allfeatures[firstdi][secdi][hrefindex] in alreadyClicks):
            leaveATIndex.remove(index)
    for i in range(10-len(leaveATIndex)):
        try:
            leaveATIndex.append(originalactionIndexs[10+i])
        except:
            continue
    clickhref = allfeatures[firstdi][secdi][hrefindex]
    clickvalue = predvalue[actionIndex]
    return actionIndex, leaveATIndex, clickhref,clickvalue.numpy(), predvalue[originalactionIndex].numpy()

def get_action_index(model,state_features, randomjump, epison, data, step, FeatureNotNull,dynamicstep=False):
    predvalue = []
    null =[]
    if FeatureNotNull:
        for batch in copy.deepcopy(state_features[len(state_features)-1]):
            features = list(zip(*batch))
            # print('step features len',len(features))
            features2 = copy.deepcopy(features)
            url = []
            AT = []
            for idx, f in enumerate(features2):
               if idx == 3:
                   url.append(f)
               if idx == 4:
                   AT.extend(f)
            model_features = [np.array(f) for index, f in enumerate(features) if index in [0,1,2]]
            if len(model_features) == 0:
                continue
            atpredd = model(model_features)  # [[0.xx],[0.xx],...]
            predvalue += list(atpredd[:, 0].numpy())  # [<tf.Tensor: shape=(), dtype=float32, numpy=0.xxx>,<>,...]
            # print('predvalue',predvalue)
        try:
            data['url'].extend(list(url[0]))
            data['AT'].extend(AT)
            data['q(s,a)'].extend(predvalue)
            data['flag'].extend(np.zeros(len(list(url[0]))))
            # print(data)
        except Exception:
            data['url'].extend(null)
            data['AT'].extend(null)
            data['q(s,a)'].extend(null)
            data['flag'].extend(null)
            print(data)
        print('data len',len(data['url']))
    originalactionIndexs = list(tf.argsort(data['q(s,a)'], axis=-1, direction='DESCENDING').numpy())
    if randomjump:
        # for i in range(1):
        if (random.randint(0, 10) / 10 <= epison):
            actionIndex = random.sample(list(range(0, len(data['q(s,a)']))), 1)[0]
            print('actionIndex',actionIndex)
            if data['flag'][actionIndex] == 1:
                actionIndex = random.sample(list(range(0, len(data['q(s,a)']))), 1)[0]
                if data['flag'][actionIndex] == 1:
                    actionIndex = random.sample(list(range(0, len(data['q(s,a)']))), 1)[0]

        else:
            actionIndex = originalactionIndexs[0]
            print('actionIndex pick max', actionIndex)
            if data['flag'][actionIndex] == 1:
                actionIndex = originalactionIndexs[1]
                print('actionIndex second',actionIndex)
    else:
        actionIndex = originalactionIndexs[0]
        # if data['flag'][actionIndex] == 1:
        #     actionIndex = originalactionIndexs[1]
        #     print('actionIndex second', actionIndex)
    clickhref = data['url'][actionIndex]
    predictvalue=data['q(s,a)'][actionIndex]
    anchortext = data['AT'][actionIndex]
    maxvalue = data['q(s,a)'][originalactionIndexs[0]]
    data['flag'][actionIndex] = 1
    if not dynamicstep and step == 2:
        data.clear()
    elif dynamicstep and step == 4:
        data.clear()
    return actionIndex, clickhref, predictvalue,anchortext,maxvalue
def ProcessFeatures_beautifulsoup(at,tagpath, tagtokenize,bertlayer,config):
    # print('ProcessFeatures coordinate',coordinate)
    # coo = GetCoordinateNeighbor(coordinate)
    tagid_1 = GetTokenTagPath(tagpath,tagtokenize)
    tagid = tf.repeat(np.array([tagid_1]),len(tagid_1),axis=0)
    return StringTool.GetTextListEmbedding(at,bertlayer,config), list(tagid.numpy()), GetIndexMetrix(tagid.shape)

def ProcessFeatures(at, coordinate,tagpath, tagtokenize,bertlayer,config):
    # print('ProcessFeatures coordinate',coordinate)
    coo = GetCoordinateNeighbor(coordinate)
    tagid_1 = GetTokenTagPath(tagpath,tagtokenize)
    tagid = tf.repeat(np.array([tagid_1]),len(coo),axis=0)
    # print('tagid_1 len:{},tagid_1:{}'.format(len(tagid_1),tagid_1))  # AT len , [[x x x x][x x x ]][x x x x]]
    # print('tagid len:{},tagid:{}'.format(len(tagid),tagid))  # AT len , [[[x x x x][x x x ]]...[[x x x x][x x x x]]]
    # return StringTool.GetTextListEmbedding(at,bertlayer,config),  list(coo), list(tagid.numpy()),GetIndexMetrix(tagid.shape),GetwidthMetrix(ATlenweight)
    return StringTool.GetTextListEmbedding(at,bertlayer,config),  list(coo), list(tagid.numpy()),GetIndexMetrix(tagid.shape)

def ProcessFeatures_sequence_output(at, coordinate,tagpath, tagtokenize,bertlayer,config):
    # print('ProcessFeatures coordinate',coordinate)
    coo = GetCoordinateNeighbor(coordinate)
    tagid_1 = GetTokenTagPath(tagpath,tagtokenize)
    tagid = tf.repeat(np.array([tagid_1]),len(coo),axis=0)
    # print('tagid',tagid.shape)
    # print('tagid_1 len:{},tagid_1:{}'.format(len(tagid_1),tagid_1))  # AT len , [[x x x x][x x x ]][x x x x]]
    # print('tagid len:{},tagid:{}'.format(len(tagid),tagid))  # AT len , [[[x x x x][x x x ]]...[[x x x x][x x x x]]]
    # return StringTool.GetTextListEmbedding(at,bertlayer,config),  list(coo), list(tagid.numpy()),GetIndexMetrix(tagid.shape),GetwidthMetrix(ATlenweight)
    return StringTool.GetTextListEmbedding_sequence_output(at,bertlayer,config),  list(coo), list(tagid.numpy()),GetIndexMetrix(tagid.shape)
def ProcessFeatures_critic(title,criticAT,bertlayer,config):
    return StringTool.GetTextListEmbedding(title,bertlayer,config),StringTool.GetTextListEmbedding(criticAT,bertlayer,config)
def ProcessFeatures_critic_sequence_output(title,criticAT,bertlayer,config):
    return StringTool.GetTextListEmbedding_sequence_output(title,bertlayer,config),StringTool.GetTextListEmbedding_sequence_output(criticAT,bertlayer,config)

def GetwidthMetrix(ATlenweight):
    # print('ATlenweight',ATlenweight)
    shape = (len(ATlenweight),4)
    weight = np.ones(shape)
    for idx, i in enumerate(ATlenweight):
        # if 0 < i < 0.5:
        #     # ATlenweight[idx] = round(i * 1.5, 2)
        #     ATlenweight[idx] = random.randint(1, 10)/100
        if i==0:
            ATlenweight[idx] = random.randint(1, 6)/10
        elif i > 0.25:
            # ATlenweight[idx] = round(i * 0.5, 2)
            ATlenweight[idx] = random.randint(45, 60)/100
    weight[:, 2] = ATlenweight
    weight = tf.constant(weight, dtype='float32')
    return weight
def GetTokenTagPath(path,tagtokenize):
    ids = [tagtokenize.GetIDs(tmp) for tmp in path]
    tagoutput = np.zeros((len(ids),tagtokenize.GetDictLength()))
    for i in range(len(ids)):
        tagoutput[i] = np.bincount(ids[i],minlength=tagtokenize.GetDictLength())
    return tagoutput
def GetIndexMetrix(tagshape):
    shape = (tagshape[0],tagshape[1],36)
    index = np.zeros(shape)
    print('shape:{}'.format(shape))
    if tagshape[0] == tagshape[1]:
        for i in range(shape[0]):
            index[i][i] = 1
    return index
def GetIndexMetrix_pre(tagshape):
    shape = (tagshape[0],tagshape[1],36)
    index = np.zeros(shape)
    print('shape:{}'.format(shape))
    for i in range(shape[0]):
        index[i][i] = 1
    return index
def GetCoordinateNeighbor(coordinate):
    neighborCoordinate = []
    # print('coordinate',coordinate)
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