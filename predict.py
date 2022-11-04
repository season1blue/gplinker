import json,pipetool
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import sparse_multilabel_categorical_crossentropy
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from tqdm import tqdm
from pathlib  import Path 

def extract_spoes(text, threshold=0):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = {} 
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                psh = mapping[sh][0]
                pst = mapping[st][-1] + 1
                poh = mapping[oh][0]
                pot = mapping[ot][-1] + 1
                h = text[psh:pst]
                t = text[poh:pot]
                p = id2predicate[p]
                key = '_'.join([h,p,t])
                if not spoes.get(key):
                    p_spo = (
                        [h,psh,pst],p,[t,poh,pot]
                    )
                    spoes[key] = p_spo
    return spoes.values()

config_path = 'pretrain_models/chinese_wobert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pretrain_models/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pretrain_models/chinese_wobert_L-12_H-768_A-12/vocab.txt'

    
# 加载预训练模型
base = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False
)

# 预测结果
entity_output = GlobalPointer(heads=2, head_size=64)(base.model.output)
head_output = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
tail_output = GlobalPointer(
    heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
)(base.model.output)
outputs = [entity_output, head_output, tail_output]

# 构建模型
model = keras.models.Model(base.model.inputs, outputs)
# model.compile(loss=globalpointer_crossentropy, optimizer=Adam(2e-5))
# model.summary()


import tensorflow as tf
opt = tf.keras.optimizers.Adam(1e-5) 
#add a line  混合精度训练
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(
            opt,
            loss_scale='dynamic')

model.compile(loss=globalpointer_crossentropy, optimizer=opt)
 
model.summary() 

model_n = 'best_model.weights'
def do_predict(): 
    print("predicting.....")
    model.load_weights(model_n)
    with open(Path('output','spo.json'),'w', encoding='utf-8') as f:
        for d in tqdm(test_data):
            rs = {'ID':d['ID'],'text':d['text'],'spo_list':[]}
            R = extract_spoes(d['text'])
            for spo in R:
                spo_dict = {
                    'h':{
                        'name':spo[0][0],
                        'pos':[spo[0][1],spo[0][2]]
                        },
                    't':{
                        'name':spo[2][0],
                        'pos':[spo[2][1],spo[2][2]]
                        },
                    'relation':spo[1]
                    }
                rs['spo_list'].append(spo_dict)
            s = json.dumps(rs,ensure_ascii=False)
            f.write(s + '\n') 

do_predict()