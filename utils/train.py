# Zhu Zhi, @Fairy Devices Inc., 2021
# ==============================================================================
import os
import pickle
import multiprocessing

import wandb
import numpy as np
# import pandas as pd

from utils import corpus, evaluation


class Train_mono():

    def __init__(self,
                 modelsPath,
                 df_train,
                 df_val,
                 df_test,
                 emotions,
                 sp_test,
                 sample_weight_mode='segment_class',
                 lr=1e-5,
                 batch_size=64,
                 epochs=1000,
                 monitor='val_wa',
                 es_patience=100,
                 lr_patience=30,
                 lr_decay=0.8,
                 lr_limit=1e-8,
                 use_wandb=False,
                 project_name=None,
                 group_name=None,
                 job_type_name=None,
                 wandb_configs=None,
                 **kwargs):
        print('Initializing training object')
        # path to model logs
        self.modelsPath = modelsPath
        self.modelfile = '{}/BestModel_{}.h5'.format(modelsPath, sp_test)
        self.history_file = '{}/history_{}.pickle'.format(modelsPath, sp_test)
        # corpus information
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.emotions = emotions
        self.sp_test = sp_test
        # data parameters
        self.sample_weight_mode = sample_weight_mode
        # training process parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        # call parameters
        self.monitor = monitor
        self.es_patience = es_patience
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_limit = lr_limit
        # wandb
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.group_name = group_name
        self.job_type_name = job_type_name
        self.wandb_configs = wandb_configs
        # other parameters
        self.kwargs = kwargs


    def make_training_datasets(self):
        from utils import prepare_data
        # training data
        data_pipline = prepare_data.Melspectrogram_pipline(
            self.df_train.filepath.to_list(), **self.kwargs
        )
        feature_ds, num_segments_list = data_pipline()
        self.train_dict, train_class_weights = prepare_data.label_weight(
            np.stack(self.df_train.label), num_segments_list, **self.kwargs
        )
        self.train_dict['melspectrogram'] = feature_ds
        self.train_dict['num_segments_list'] = num_segments_list
        # training validation data
        data_pipline = prepare_data.Melspectrogram_pipline(
            self.df_val.filepath.to_list(), **self.kwargs
        )
        feature_ds, num_segments_list = data_pipline()
        self.train_val_dict, _ = prepare_data.label_weight(
            np.stack(self.df_val.label), num_segments_list,
            class_weights=train_class_weights, **self.kwargs
        )
        self.train_val_dict['melspectrogram'] = feature_ds
        self.train_val_dict['num_segments_list'] = num_segments_list

    def make_evaluation_datasets(self):
        from utils import prepare_data
        # validation data
        data_pipline = prepare_data.Melspectrogram_pipline(
            self.df_val.filepath.to_list(), **self.kwargs
        )
        feature_ds, num_segments_list = data_pipline()
        self.val_dict, _ = prepare_data.label_weight(
            np.stack(self.df_val.label), num_segments_list, **self.kwargs
        )
        self.val_dict['melspectrogram'] = feature_ds
        self.val_dict['num_segments_list'] = num_segments_list
        # test data
        data_pipline = prepare_data.Melspectrogram_pipline(
            self.df_test.filepath.to_list(), **self.kwargs
        )
        feature_ds, num_segments_list = data_pipline()
        self.test_dict, _ = prepare_data.label_weight(
            np.stack(self.df_test.label), num_segments_list, **self.kwargs
        )
        self.test_dict['melspectrogram'] = feature_ds
        self.test_dict['num_segments_list'] = num_segments_list

    def make_model(self):
        import tensorflow as tf
        from utils import prepare_model

        def _make_model():
            # normalization layer to normalize melspectrogram input
            norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
            norm_layer.adapt(self.train_dict['melspectrogram'])
            # make model
            input_shape = next(iter(self.train_dict['melspectrogram'])).shape
            self.model = prepare_model.make_model(
                input_shape, len(self.emotions), norm_layer, **self.kwargs)
            # compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.lr),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name='wa')]
            )

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.backend.clear_session()
        if self.distribute:
            with self.strategy.scope():
                _make_model()
        else:
            _make_model()
        # make callbacks
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=self.monitor, patience=self.es_patience,
                verbose=1, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.monitor, factor=self.lr_decay,
                patience=self.lr_patience, verbose=1,
                mode='max', min_lr=self.lr_limit),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.modelfile, monitor=self.monitor,
                verbose=1, save_best_only=True, mode='max')
        ]


    def wandbcallback(self):
        from utils import keras_callbacks
        if self.distribute:
            train_ds_distribute = self.strategy.experimental_distribute_dataset(self.train_ds)
            self.callbacks.append(
                keras_callbacks.WandbCallback_distribute(
                    wandb_run=self.wandb_run,
                    batch_size=self.batch_size,
                    train_data=train_ds_distribute,
                    strategy=self.strategy)
            )
        else:
            self.callbacks.append(keras_callbacks.WandbCallback(
                wandb_run=self.wandb_run, train_data=self.train_ds))


    def train_process(self, history):
        import tensorflow as tf
        # multi-gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # distributed training
        if len(gpus) > 1:
            self.distribute = True
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.distribute = False
        # prepare data
        print('Prepare training datasets')
        self.make_training_datasets()
        # train datasets
        train_size = sum(self.train_dict['num_segments_list'])
        # for multi gpus
        if len(gpus) > 1:
            gpus_num = len(gpus)
            take_count = int(train_size // gpus_num * gpus_num)
        else:
            take_count = train_size
        self.train_ds = tf.data.Dataset.zip((
            self.train_dict['melspectrogram'],
            self.train_dict['label'],
            self.train_dict[self.sample_weight_mode]
        )).shuffle(train_size).take(take_count).batch(
            self.batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
        # val set
        val_size = sum(self.train_val_dict['num_segments_list'])
        # for multi gpus
        if len(gpus) > 1:
            gpus_num = len(gpus)
            take_count = int(val_size // gpus_num * gpus_num)
        else:
            take_count = val_size
        self.train_val_ds = tf.data.Dataset.zip((
            self.train_val_dict['melspectrogram'],
            self.train_val_dict['label'],
            self.train_val_dict[self.sample_weight_mode]
        )).take(take_count).batch(self.batch_size).cache().prefetch(
            tf.data.experimental.AUTOTUNE)
        # prepare model
        print('Make model')
        self.make_model()
        # wandb
        if self.use_wandb:
            try:
                self.wandb_run = self.kwargs['wandb_run']
            except:
                # wandb init
                self.wandb_run = wandb.init(
                    name=self.group_name + '_' + str(self.sp_test),
                    project=self.project_name,
                    group=self.group_name,
                    job_type=self.job_type_name,
                    config=self.wandb_configs,
                    save_code=False,
                    reinit=True
                )
            # wandb callback
            self.wandbcallback()
        # train
        print('Start training')
        train_history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            verbose=2,
            callbacks=self.callbacks,
            validation_data=self.train_val_ds
        )
        print('Training finished')
        history.update(train_history.history)
        # evaluation
        print('Evaluating')
        self.make_evaluation_datasets()
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model(self.modelfile)
        cmV, y_pV = evaluation.confusion_matrix(
            np.array(list(iter(self.val_dict['label']))),
            self.model.predict(self.val_dict['melspectrogram'].batch(32)),
            self.val_dict['num_segments_list'],
            len(self.emotions)
        )
        cmT, y_pT = evaluation.confusion_matrix(
            np.array(list(iter(self.test_dict['label']))),
            self.model.predict(self.test_dict['melspectrogram'].batch(32)),
            self.test_dict['num_segments_list'],
            len(self.emotions)
        )
        history['cmV'], history['cmT'] = cmV, cmT
        history['y_p_V'], history['y_p_T'] = y_pV, y_pT
        if self.use_wandb:
            # log results
            waV, uaV, f1V = evaluation.accuracy(cmV)
            waT, uaT, f1T = evaluation.accuracy(cmT)
            metrics_names = ['waV', 'uaV', 'f1V', 'waT', 'uaT', 'f1T']
            for metric in metrics_names:
                self.wandb_run.summary[metric] = eval(metric)
            evaluation.log_confusion_matrix(
                cmV, 'cmV', self.emotions, self.wandb_run)
            evaluation.log_confusion_matrix(
                cmT, 'cmT', self.emotions, self.wandb_run)
            self.wandb_run.finish()
        

    def __call__(self):
        if os.path.isfile(self.history_file):
            with open(self.history_file, 'rb') as f:
                history = pickle.load(f)
            cmV = history['cmV']
            cmT = history['cmT']
            return cmV, cmT
        else:
            manager = multiprocessing.Manager()
            history = manager.dict()
            train_p = multiprocessing.Process(
                target=self.train_process, args=(history,))
            train_p.start()
            train_p.join()
            train_p.kill()
            print('Training subprocess finished')
            # save history file
            history_save = {}
            history_save.update(history)
            with open(self.history_file, 'wb') as f:
                pickle.dump(history_save, f)
            cmV = history['cmV']
            cmT = history['cmT']
            return cmV, cmT
