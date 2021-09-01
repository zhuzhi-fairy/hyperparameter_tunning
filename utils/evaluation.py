# Utils of evaluation for speech emotion recognition.
# Zhu Zhi, @Fairy Devices Inc., 2020
# ==============================================================================
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def unsegment(y_seg, num_segments):
    _, num_class = y_seg.shape
    nUttrances = len(num_segments)
    y_raw = np.zeros((nUttrances, num_class))
    nsub = 0
    for nU in range(nUttrances):
        num_subs = int(num_segments[nU])
        y_raw[nU] = np.mean(y_seg[nsub:nsub+num_subs], 0)
        nsub += num_subs
    return y_raw


def unsegment_dataset(ds_num_seg):
    n, num_seg_list, num_segs = 0, [], list(iter(ds_num_seg))
    while n < len(num_segs):
        num_seg_list.append(num_segs[n].numpy())
        n += int(num_segs[n].numpy())
    num_seg_list = np.array(num_seg_list)
    return num_seg_list


def confusion_matrix(y, y_p, num_segments, nclass):
    """calculate confusion matrix of segmented data
    Args:
        y: true labels
        y_p: predicted labels
        nseg: list of the number of segments of each input speech file
        nclass: number of classes
    Returns:
        cm: confusion matrix
    """
    y_raw = unsegment(y, num_segments).argmax(axis=-1)
    y_raw_p = unsegment(y_p, num_segments).argmax(axis=-1)
    # confusion matrix
    cm = np.zeros((nclass, nclass))
    for n in range(y_raw.shape[0]):
      cm[y_raw[n], y_raw_p[n]] += 1
    return cm, y_raw_p


def accuracy(cm):
    """calculate metrics
    Args:
      cm: confusion matrix
    
    Returns:
      wa: weighted accuracy
      ua: unweighted accuracy
      f1: macro f1 score
    """
    num_classes = cm.shape[0]
    cm2 = np.array(cm, dtype="float64") + 1e-20
    # weighted accuracy
    wa = np.sum(cm2*np.eye(num_classes))/np.sum(cm2)
    # unweighted accuracy
    ua = np.mean(np.sum(cm2*np.eye(num_classes), 1)/np.sum(cm2, 1))
    # precision
    pre = np.sum(cm2*np.eye(num_classes)/np.sum(cm2, 0), 0)
    # recall
    rec = np.sum(cm2*np.eye(num_classes)/np.sum(cm2, 1), 0)
    # macro f1
    f1 = np.mean(pre*rec/(pre+rec)*2)
    # return wa, ua, f1
    return np.around((wa, ua, f1), decimals=4)


def log_confusion_matrix(cm, cmname, labels, run):
    '''log confuison matrix for wandb
    Args:
      cm: confusion matrix
      cmname: title of the confusion matrix
      labels: list of the names of labels
    Plot confusion matrix with seaborn.heatmap and log it with wandb.
    '''
    df_cm = pd.DataFrame(
        cm, index=labels, columns=labels
    )
    plt.figure()
    sns.heatmap(df_cm.astype('int'), annot=True, fmt='d')
    plt.title(cmname)
    run.log({cmname: [wandb.Image(plt, caption=cmname)]})
    plt.close()
