# Zhu Zhi, @Fairy Devices Inc., 2021
# ==============================================================================
import numpy as np
import pandas as pd
from glob import glob


def separate_sets(df, speakers, sp_test):
    '''Separate speakers into train, validation, and test sets
    Args:
        df: dataframe of corpus
        speakers: list of speakers
        sp_test: test speaker
    Returns:
        df_train: dataframe of train set
        df_val: dataframe of validation set
        df_test: dataframe of test set
    '''
    # separate speakers
    speakers = speakers.copy()
    nspeakers = len(speakers)
    speaker_test = speakers.pop(sp_test)
    speaker_val = speakers.pop(sp_test//2*2 % (nspeakers-1))
    speaker_train = sum(speakers, [])  # flatten the list
    # data frame of train, val and test sets
    df_train = df.where(df.speaker.isin(speaker_train)).dropna()
    df_val = df.where(df.speaker.isin(speaker_val)).dropna()
    df_test = df.where(df.speaker.isin(speaker_test)).dropna()
    return df_train, df_val, df_test


def load_IEMOCAP(dataPath='../Database/IEMOCAP_full_release/',
                 mode=0,
                 sp_test=0):
    emotion_dict = {
        "neu": "neutral",
        "sad": "sadness",
        "fea": "fear",
        "xxx": "xxx",
        "hap": "happiness",
        "exc": "excited",
        "dis": "disgust",
        "fru": "frustration",
        "sur": "surprise",
        "ang": "anger",
        "oth": "other"
    }
    emotions = ["neutral", "happiness", "sadness", "anger"]
    I = np.eye(len(emotions))
    if mode == 0:
        actTypeTest = ['impro']
    elif mode == 1:
        actTypeTest = ['impro', 'script']
        emotion_dict['exc'] = 'happiness'
    dataList = []
    for nSes in range(1, 6):
        txtfiles = (glob(
            '{}Session{}/dialog/EmoEvaluation/*.txt'.format(
                dataPath, nSes)))
        for nEmoEva in range(len(txtfiles)):
            with open(txtfiles[nEmoEva]) as txtfile:
                for line in txtfile:
                    if line[0] == '[':
                        line = line.split()
                        filename = line[3]
                        filename_split = filename.split("_")
                        scenario = filename_split[1][:7]
                        # informations about the sound file
                        soundPath = (
                            "{}Session{}/sentences/wav/{}"
                            .format(dataPath, nSes,
                                    filename_split[0]))
                        for m in range(1, len(filename_split)-1):
                            soundPath += "_" + filename_split[m]
                        soundPath += "/{}.wav".format(filename)
                        session = nSes
                        speaker = (filename_split[0][:5] + "_"
                                    + filename_split[-1][0])
                        if filename_split[1][0] == "i":
                            actingType = "impro"
                        else:
                            actingType = "script"
                        emotion = emotion_dict[line[4]]
                        dataList.append([
                            soundPath, filename, scenario,
                            session, speaker,
                            actingType, emotion
                        ])
    df = pd.DataFrame(
        dataList,
        columns=[
            "filepath", "filename", "scenario",
            "session", "speaker",
            "actType", "emotion"])
    df.where(df.actType.isin(actTypeTest), inplace=True)
    df.where(df.emotion.isin(emotions), inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['label'] = df.emotion.apply(lambda emo: I[emotions.index(emo)])
    # train develop test
    speakers = [
        ['Ses01_F'], ['Ses01_M'],
        ['Ses02_F'], ['Ses02_M'],
        ['Ses03_F'], ['Ses03_M'],
        ['Ses04_F'], ['Ses04_M'],
        ['Ses05_F'], ['Ses05_M']
    ]
    df_train, df_val, df_test = separate_sets(df, speakers, sp_test)
    return df_train, df_val, df_test


def load_MELD():
    emotions = [
        'neutral', 'joy', 'sadness', 'anger',
        'disgust', 'fear', 'surprise'
    ]
    I = np.eye(len(emotions))
    dataPath = '../Database/MELD/'

    def _load_df(setname):
        df = pd.read_csv(dataPath+setname+'_sent_emo.csv', index_col=[0])
        df['filepath'] = df.apply(
            lambda row: '{}audio/{}/dia{}_utt{}.wav'.format(
                dataPath, setname, row['Dialogue_ID'], row['Utterance_ID']),
                axis=1)
        df['label'] = df.Emotion.apply(lambda emo: I[emotions.index(emo)])
        wrongfiles = [
            '{}audio/train/dia125_utt3.wav'.format(dataPath),
            '{}audio/dev/dia110_utt7.wav'.format(dataPath),
            '{}audio/test/dia23_utt0.wav'.format(dataPath)
        ]
        df = df.where(~df.filepath.isin(wrongfiles)).dropna()
        return df

    df_train = _load_df('train')
    df_val = _load_df('dev')
    df_test = _load_df('test')
    return df_train, df_val, df_test