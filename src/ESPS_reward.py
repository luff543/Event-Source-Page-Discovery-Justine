import configparser
from official import nlp
import official.nlp.optimization
import tensorflow_hub as hub
import tensorflow as tf
from src import DataPreprocessing_dev_new
import numpy as np
def getESPSreward(url,config):
    optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=1, num_warmup_steps=1)
    l_bert = hub.KerasLayer(config.get('BertTokenize', 'bert_path_pre'),
                            trainable=False)
    model = tf.keras.models.load_model(config.get('PaperModelSettings', 'ESPS_path'),
                                       {"KerasLayer": l_bert, "AdamWeightDecay": optimizer})
    # model.summary()
    featuresFunctionMap = {"maxrepeat": DataPreprocessing_dev_new.GetMaxRepeat,
                           "secondrepeat": DataPreprocessing_dev_new.GetSecondRepeat,
                           "pagetitle": DataPreprocessing_dev_new.GetTitleEmbedding,
                           "middlesegment": DataPreprocessing_dev_new.GetMiddleSegment,
                           }
    inputs = []
    for feature in config.get('PaperModelSettings', 'model_path').split('/')[-1].split('_'):
        if feature not in featuresFunctionMap:
            continue
        # print(feature)
        if feature == 'maxrepeat' or feature == 'secondrepeat':
            tmp = featuresFunctionMap[feature](url, max_sequence=128, getTokenID=True)
        elif feature == 'pagetitle':
            tmp = featuresFunctionMap[feature](url, max_sequence=64, getTokenID=True)
        else:
            tmp = featuresFunctionMap[feature](url, max_sequence=256, getTokenID=True)
        inputs += [np.array([tmp[0]]), np.array([tmp[2]]), np.array([tmp[1]])]

    result = model.predict(inputs)[0][0]
    print('type result',type(result))
    # print("score:{}".format(str(result)))
    reward = round(result,3)
    return reward

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./config.ini')
    # url = 'https://www.tndais.gov.tw/theme_list.php?theme=news&sub_theme=activity'
    url = 'https://kktix.com/events'
    reward=getESPSreward(url,config)
    print(reward)