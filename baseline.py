#! -*- coding:utf-8 -*-
# 关系抽取任务，基于GPLinker
# 文章介绍：https://kexue.fm/archives/8888
import json, pipetool
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
from pathlib import Path

import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))


def normalize(text):
    """简单的文本格式化函数
    """
    return ' '.join(text.split())


# 100个有space的
def clear_space(line_data):
    arr = {}
    arr['ID'] = line_data['ID']

    text = line_data['text']
    arr['text'] = text.replace(" ", "")
    spo_list = line_data['spo_list']
    new_spo_list = []
    for spo in spo_list:
        h, t, r = spo['h'], spo['t'], spo['relation']

        head_str = str(text[:h['pos'][0]])
        new_h = {'name': h['name'], 'pos': [h['pos'][0] - head_str.count(" "), h['pos'][1] - head_str.count(" ")]}
        tail_str = str(text[:t['pos'][0]])
        new_t = {'name': t['name'], 'pos': [t['pos'][0] - tail_str.count(" "), t['pos'][1] - tail_str.count(" ")]}

        new_spo = {'h': new_h, 't': new_t, 'relation': r}
        new_spo_list.append(new_spo)

    arr['spo_list'] = new_spo_list

    return arr


def format_new_data(line_data):
    arr = {}
    text = line_data['text']
    arr['text'] = line_data['text']

    h, t, r = line_data['h'], line_data['t'], line_data['relation']

    head_str = str(text[:h['pos'][0]])
    new_h = {'name': h['name'], 'pos': [h['pos'][0] - head_str.count(" "), h['pos'][1] - head_str.count(" ")]}
    tail_str = str(text[:t['pos'][0]])
    new_t = {'name': t['name'], 'pos': [t['pos'][0] - tail_str.count(" "), t['pos'][1] - tail_str.count(" ")]}
    new_spo = {'h': new_h, 't': new_t, 'relation': r}

    arr['spo_list'] = [new_spo]

    return arr


def delete_dup(data_list):
    text_set, spo_set = [], []
    new_data_list = []
    # 构建text_set and spo_set
    for line_data in data_list:
        text_permit, spo_permit = False, False
        if line_data['text'] not in text_set:
            text_set.append(line_data['text'])
            text_permit = True

        spo_list = line_data['spo_list']
        line_spo_set = set()
        for spo in spo_list:
            line_spo_set.add(spo['h']['name'])
            line_spo_set.add(spo['t']['name'])

        if line_spo_set not in spo_set:
            spo_set.append(line_spo_set)
            spo_permit = True

        # # 检测只出现一次的data
        # for line_data in data_list:
        #     text_permit, spo_permit = False, False
        #     if line_data['text'] not in text_set:
        #         text_permit = True
        #     spo_list = line_data['spo_list']
        #     line_spo_set = set()
        #     for spo in spo_list:
        #         line_spo_set.add(spo['h']['name'])
        #         line_spo_set.add(spo['t']['name'])
        #
        #     if line_spo_set not in spo_set:
        #         spo_permit = True
        #     if text_permit and spo_permit:
        #         once_data_list.append(line_data)
        if text_permit and spo_permit:
            new_data_list.append(line_data)

    return new_data_list


# 202号汽车故障报告故障现象空调工作状态指示、室外温度显示及行车数据均在显示屏上无显示，Ａ/Ｃ开关、内外循环开关及后风窗加热开关指示灯也不亮 1159 2898
def pure_data(filename_old, filename_new):
    f_old = open(filename_old, "r", encoding='utf8').readlines()
    f_old2 = open('ouside_data/train_maybeEval.json', "r", encoding='utf8').readlines()
    f_new = open(filename_new, "r", encoding='utf8').readlines()

    f_train = open("data/train_pure.json", 'w', encoding='utf8')
    f_dev = open("data/dev.json", 'w', encoding='utf8')

    data_list = []
    # 5956 2953
    for i in f_old2:
        i = i.strip()
        line_data = json.loads(i)
        if " " in line_data['text']:
            line_data = clear_space(line_data)
        data_list.append(line_data)

    for index, i in enumerate(f_old):
        i = i.strip()
        line_data = json.loads(i)
        # 62 过长的
        if " " in line_data['text']:
            line_data = clear_space(line_data)
        data_list.append(line_data)

    for i in f_new:
        i = i.strip()
        line_data = json.loads(i)
        line_data = format_new_data(line_data)
        data_list.append(line_data)

    data_list = delete_dup(data_list)  # TDOO 是否要删掉duplicate 还待商榷
    f_train.write(json.dumps(data_list, ensure_ascii=False, indent=1))


def load_data(filename):
    f = open(filename, "r", encoding='utf8').readlines()
    data = []
    for index, i in enumerate(f):
        i = i.strip()
        i_data = json.loads(i)
        data.append(i_data)
    return data


def json_load_data(filename):
    f = open(filename, "r", encoding='utf8')
    return json.load(f)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        for is_end, d in self.sample(random):
            text = d['text'].lower()
            tokens = tokenizer.tokenize(text, maxlen=maxlen)
            mapping = tokenizer.rematch(text, tokens)
            head_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            tail_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            # 整理三元组 {(s, o, p)}
            spoes = set()
            for spo in d['spo_list']:
                s = spo['h']['name']
                o = spo['t']['name']
                p = spo['relation']
                s, o = s.lower(), o.lower()
                if s not in text or o not in text:
                    continue

                sh = spo['h']['pos'][0]
                oh = spo['t']['pos'][0]
                # sh1, oh1 = text.index(s), text.index(o)
                st, ot = sh + len(s) - 1, oh + len(o) - 1
                if sh in head_mapping and st in tail_mapping:
                    if oh in head_mapping and ot in tail_mapping:
                        sh, st = head_mapping[sh], tail_mapping[st]
                        oh, ot = head_mapping[oh], tail_mapping[ot]
                        if sh <= st and oh <= ot:
                            spoes.add((sh, st, predicate2id[p], oh, ot))
            # 构建标签
            entity_labels = [set() for _ in range(2)]
            head_labels = [set() for _ in range(len(predicate2id))]
            tail_labels = [set() for _ in range(len(predicate2id))]
            for sh, st, p, oh, ot in spoes:
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))
                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
            for label in entity_labels + head_labels + tail_labels:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_entity_labels = sequence_padding(
                    batch_entity_labels, seq_dims=2
                )
                batch_head_labels = sequence_padding(
                    batch_head_labels, seq_dims=2
                )
                batch_tail_labels = sequence_padding(
                    batch_tail_labels, seq_dims=2
                )
                yield [batch_token_ids, batch_segment_ids], [
                    batch_entity_labels, batch_head_labels, batch_tail_labels
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = K.shape(y_pred)
    y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
    return K.mean(K.sum(loss, axis=1))


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
                key = '_'.join([h, p, t])
                if not spoes.get(key):
                    p_spo = (
                        [h, psh, pst], p, [t, poh, pot]
                    )
                    spoes[key] = p_spo
    return spoes.values()


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO((spo[0][0], spo[1], spo[2][0])) for spo in extract_spoes(d['text'])])
        T = set([SPO((spo['h']['name'], spo['relation'], spo['t']['name'])) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
            ensure_ascii=False,
            indent=4)
        if list(R - T):
            f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


def do_predict():
    model.load_weights(model_n)
    with open(Path('output', 'spo.json'), 'w', encoding='utf-8') as f:
        for d in tqdm(test_data):
            rs = {'ID': d['ID'], 'text': d['text'], 'spo_list': []}
            R = extract_spoes(d['text'])
            for spo in R:
                spo_dict = {
                    'h': {
                        'name': spo[0][0],
                        'pos': [spo[0][1], spo[0][2]]
                    },
                    't': {
                        'name': spo[2][0],
                        'pos': [spo[2][1], spo[2][2]]
                    },
                    'relation': spo[1]
                }
                rs['spo_list'].append(spo_dict)
            s = json.dumps(rs, ensure_ascii=False)
            f.write(s + '\n')


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(model_n)
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


class ModelCheckpoint(keras.callbacks.Callback):
    """自动保存最新模型。"""

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(checkpoint_path, overwrite=True)


# =====================================================正文开始================================================

maxlen = 512
batch_size = 2
epochs = 150

config_path = 'pretrain_models/NEZHA-Base-WWM/bert_config.json'
checkpoint_path = 'pretrain_models/NEZHA-Base-WWM/model.ckpt'
dict_path = 'pretrain_models/NEZHA-Base-WWM/vocab.txt'

# 数据集预处理
pure_data(filename_old='data/train_concat.json', filename_new='ouside_data/new_train_bdci.json')

# 加载数据集
data = json_load_data('data/train_pure.json')  # 5956
test_data = load_data('data/test.json')

train_data = [j for i, j in enumerate(data) if i % 9 != 0]
valid_data = [j for i, j in enumerate(data) if i % 9 == 0]
predicate2id, id2predicate = {}, {}

with open('data/schemas.json', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载预训练模型
base = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    model='nezha'
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

opt = tf.compat.v1.train.AdamOptimizer(1e-5)
# add a line  混合精度训练
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale='dynamic')

model.compile(loss=globalpointer_crossentropy, optimizer=opt)

model.summary()

# from tools.adversarial_training import *
# adversarial_training(model, 'Embedding-Token', 0.5)

model_n = 'best_model.weights'


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
else:
    do_predict()
# do_predict()
