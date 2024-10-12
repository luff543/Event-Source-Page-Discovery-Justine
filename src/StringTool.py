from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

try:
    from bert.tokenization.bert_tokenization import FullTokenizer
except:
    from bert.tokenization import FullTokenizer


def CleanString(string):
    cleantext = string.replace('\n', '')
    cleantext = cleantext.replace('\t', '')
    cleantext = cleantext.replace('\r', '')
    cleantext = cleantext.replace(' ', '')
    return cleantext


def ConvertListToID(data, config):
    tokenizer = FullTokenizer(vocab_file=config.get('BertTokenize', 'vocab_file'), do_lower_case=True)
    sep_id = int(np.array([tokenizer.convert_tokens_to_ids(["[SEP]"])])[0])
    cls_id = int(np.array([tokenizer.convert_tokens_to_ids(["[CLS]"])])[0])
    output = []
    for d in data:
        hi = tokenizer.tokenize(d)
        tmp = [cls_id] + list(tokenizer.convert_tokens_to_ids(hi)) + [sep_id]
        output.append(tmp)
    return output


def GetAnchorTextToken(data, config):
    return ConvertListToID(data, config)


def GetTextListEmbedding(anchorTexts, bertlayer, config):
    text = GetAnchorTextToken(anchorTexts, config)
    # print('text len',len(text))
    text = pad_sequences(text, maxlen=10, dtype=np.int32, padding='post', truncating='post')
    featureEmbedding = bertlayer({"input_word_ids": text, "input_mask": np.ones(shape=text.shape, dtype=np.int32),
                                   "input_type_ids": np.zeros(shape=text.shape, dtype=np.int32)})["pooled_output"].numpy()
    # featureEmbedding = \
    # bertlayer([text, np.zeros(shape=text.shape, dtype=np.int32), np.zeros(shape=text.shape, dtype=np.int32)])[
    #     0].numpy()  # textMap[str(anchorTexts)]
    return featureEmbedding

def GetTextListEmbedding_sequence_output(anchorTexts, bertlayer, config):
    text = GetAnchorTextToken(anchorTexts, config)
    # print('text len',len(text))
    text = pad_sequences(text, maxlen=10, dtype=np.int32, padding='post', truncating='post')
    featureEmbedding = bertlayer({"input_word_ids": text, "input_mask": np.ones(shape=text.shape, dtype=np.int32),
                                   "input_type_ids": np.zeros(shape=text.shape, dtype=np.int32)})["sequence_output"].numpy()

    return featureEmbedding
