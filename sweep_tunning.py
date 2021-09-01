import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import traceback
import time

import wandb
import slackweb

from utils import corpus, train, evaluation

hostname = os.uname()[1]
slack_url = 'https://hooks.slack.com/services/T02CY73AU/B01LBU7R3HD/' \
    'XRBvWpD6fVSWkAjR1sacqEfa'
slack = slackweb.Slack(url=slack_url)


def make_folder(foldername):
    '''Make folder
    Args:
      foldername: folder name
    Make a new folder if not existed.
    '''
    try:
        os.mkdir(foldername)
    except FileExistsError:
        pass


def train_process(tunning_parameters):
    default_parameters = {
        # corpus
        'corpus': 'MELD',
        'emotions': [
            'neutral', 'happiness', 'sadness', 'anger',
            'disgust', 'fear', 'surprise'
        ],
        # data
        'sr': 16000, 'seg_len': 3, 'seg_hop': 1, 'seg_norm': 1,
        'win_len': 0.025, 'win_hop': 0.01, 'n_mels': 40, 'd_dd': False,
        # model
        'init_mode': 'custom', 'norm': 'batch',
        'conv1k': 128, 'conv1f1': 3, 'conv1f2': 3, 'maxp1f1': 2, 'maxp1f2': 4,
        'nConv2': 1, 'conv2k': 128, 'conv2f1': 3, 'conv2f2': 3,
        'maxp2f1': 1, 'maxp2f2': 2,
        'lin': 768, 'rnn': 128, 'rnn_drop': 0.3,
        'att_layer': 'keras0', 'num_heads': 8, 'key_dim': 128,
        'nfc_layers': 2, 'nfc': 128, 'fc_drop': 0.3,
        # training
        'lr': 1e-5, 'batch_size': 256, 'epochs': 1000,
        'sample_weight_mode': 'segment_class',
        'es_patience': 300, 'lr_patience': 30, 'lr_decay': 0.8,
        'lr_limit': 1e-8, 'monitor': 'val_wa',
        # other
        'use_wandb': True
    }
    # tunning parameters
    parameters = default_parameters
    # make folders
    time_now = '{}_{}'.format(
        hostname, time.strftime("%m%d%H%M%S", time.localtime()))
    modelsPath = 'data/model_logs/{}'.format(time_now)
    make_folder(modelsPath)
    # corpus
    if parameters['corpus'] == 'MELD':
        df_train, df_val, df_test = corpus.load_MELD()
        sp_test = 0
    # wandb
    # wandb_sweep_run = wandb.init()
    parameters.update(tunning_parameters)
    project_name = 'MELD'
    group_name = 'data_model_parameters'
    train_obj = train.Train_mono(
        modelsPath=modelsPath,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        sp_test=sp_test,
        project_name=project_name,
        group_name=group_name,
        job_type_name='training',
        run_name=time_now,
        wandb_configs=parameters,
        **parameters
    )
    cmV, cmT = train_obj()
    return cmV, cmT


def tunning_process():
    # wandb_sweep_run = wandb.init()
    with wandb.init() as run:
        parameters = wandb.config
        parameters['use_wandb'] = False
        cmV, cmT = train_process(parameters)
        waV, uaV, f1V = evaluation.accuracy(cmV)
        waT, uaT, f1T = evaluation.accuracy(cmT)
        metrics_names = ['waV', 'uaV', 'f1V', 'waT', 'uaT', 'f1T']
        for metric in metrics_names:
            wandb.summary[metric] = eval(metric)


def main():
    import wandb
    sweep_config = {
        'name': 'MELD_tunning',
        "method" : "random",
        'metric' : {'name': 'f1T', 'goal': 'maximize'},
        "parameters" : {
            'n_mels': {
                'values': [40, 80]
            },
            "d_dd" : {
                "values" : [True, False]
            },
            "norm" : {
                "values" : ['batch', 'layer']
            },
            'lr': {
                'values': [1e-3, 1e-4, 1e-5]
            },
            'nConv2': {
                'values': [1, 2, 4]
            },
            'nfc_layers': {
                'values': [2, 3]
            },
            'num_head': {
                'values': [8, 32]
            },
            'key_dim': {
                'values': [128, 512]
            },
            'batch_size': {
                'values': [128, 256]
            },
            'lr_patience': {
                'values': [30, 80]
            }
        }
    }
    project_name = 'MELD_hyperparameters'
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    count = 100
    wandb.agent(sweep_id, function=tunning_process, count=count)
    slack.notify(text='Finished.{}{}'.format(hostname, sys.argv[1]))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        t, v, tb = sys.exc_info()
        for message in traceback.format_exception(t, v, tb):
            print(message)
        slack.notify(text='Crashed.{}'.format(hostname))
