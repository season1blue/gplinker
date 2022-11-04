from


maxlen = 512
batch_size = 2
epochs = 70
k_num = 5
train = True

config_path = 'pretrain_models/NEZHA-Base-WWM/bert_config.json'
checkpoint_path = 'pretrain_models/NEZHA-Base-WWM/model.ckpt'
dict_path = 'pretrain_models/NEZHA-Base-WWM/vocab.txt'
model_n = 'best_model.weights'

if train:
    # 数据集预处理
    pure_data(filename_old='data/train_concat.json', filename_new='ouside_data/new_train_bdci.json')
    # 加载数据集
    data = json_load_data('data/train_pure.json')  # 5956
    data = np.array(data)
    # 构建train_data和valid_data
    kf = KFold(n_splits=k_num, shuffle=True, random_state=42)
    fold = 0
    for train_index, valid_index in kf.split(data):
        fold += 1
        print(f"Current fold is {fold}")
        train_data = data[train_index]
        valid_data = data[valid_index]

        # train_data = [j for i, j in enumerate(data) if i % 9 != 0]
        # valid_data = [j for i, j in enumerate(data) if i % 9 == 0]
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
        head_output = GlobalPointer(heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False)(
            base.model.output)
        tail_output = GlobalPointer(heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False)(
            base.model.output)
        outputs = [entity_output, head_output, tail_output]

        # 构建模型
        model = keras.models.Model(base.model.inputs, outputs)

        opt = tf.compat.v1.train.AdamOptimizer(1e-5)
        # add a line  混合精度训练
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale='dynamic')

        model.compile(loss=globalpointer_crossentropy, optimizer=opt)
        model.summary()

        # from tools.adversarial_training import *
        # adversarial_training(model, 'Embedding-Token', 0.5)

        # Run main program
        train_generator = data_generator(train_data, batch_size)
        evaluator = Evaluator()
        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
else:
    test_data = load_data('data/test.json')
    do_predict(model_name=model_n, t_data=test_data)