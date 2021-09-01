# Zhu Zhi, @Fairy Devices Inc., 2021
# ==============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_addons as tfa


class Melspectrogram_pipline():
    '''Pipline for preparing melspectrogram data'''

    def __init__(self,
                 wavfiles,
                 sr=16000,
                 seg_len=3,
                 seg_hop=1,
                 win_len=0.025,
                 win_hop=0.01,
                 n_mels=40,
                 d_dd=False,
                 **kwargs):
        '''Prepare data \
        Args: \
            wavfiles: list of file paths of .wav files \
            sr: sampling rate \
            seg_len: length of segment in seconds \
            seg_hop: hop of segment in seconds \
            win_len: spectrogram window length \
            win_hop: spectrogram window hop \
            n_mels: dimensions of mel-filters \
            d_dd: option of delta and delta-delta features \
        '''
        self.wavfiles = wavfiles
        self.sr = sr
        self.seg_len = seg_len
        self.seg_hop = seg_hop
        self.win_len = win_len
        self.win_hop = win_hop
        self.n_mels = n_mels
        self.d_dd = d_dd
        self.kwargs = kwargs
    
    def load_audio(self, filepath, ch=0):
        '''load the audio tensor \
        Args: \
            filepath: path of .wav file \
        Return: \
            audio_tensor: tf.tensor of the audio signal \
        '''
        # load audio file
        audioIOtensor = tfio.audio.AudioIOTensor(filepath, dtype=tf.int16)
        if audioIOtensor.to_tensor().shape[-1] == 6:
            ch = 2  # choose the 3rd channel for 5.1ch
        # int16 -> float32
        audio_tensor = tf.cast(audioIOtensor.to_tensor()[:, ch], tf.float32)
        # range -1, 1
        audio_tensor = audio_tensor / audioIOtensor.dtype.max
        # resample into target sample rate
        if audioIOtensor.rate != self.sr:
            audio_tensor = tfio.audio.resample(
                input=audio_tensor,
                rate_in=tf.cast(audioIOtensor.rate, tf.int64),
                rate_out=tf.constant(self.sr, tf.int64)
            )
        return audio_tensor

    def audio_data_prepare(self):
        '''prepare the list of audio signals'''
        audio_tensor_list = []
        for n in range(len(self.wavfiles)):
            audio_tensor = self.load_audio(self.wavfiles[n])
            audio_tensor_list.append(audio_tensor)
        return audio_tensor_list

    def amp_norm(self, audio_tensor):
        '''Amplitude normalization
        Args:
            audio_tensor: input audio tensor
        Return:
            normalized_audio_tensor
        '''
        # amplitude normalization
        audio_mean = tf.math.reduce_mean(audio_tensor)
        audio_std = tf.math.reduce_std(audio_tensor)
        normalized_audio_tensor = (audio_tensor - audio_mean) / audio_std
        return normalized_audio_tensor

    def segmenting(self, audio_tensor_list):
        '''Segmenting audio signals'''
        seg_len_p = int(self.seg_len * self.sr)
        seg_hop_p = int(self.seg_hop * self.sr)
        segments_list, num_segments_list = [], []
        for n in range(len(self.wavfiles)):
            audio_tensor = audio_tensor_list[n]
            audio_length = int(audio_tensor.shape[0])
            # cropping and padding signal
            start, end, num_segments = 0, seg_len_p, 0
            while(end < audio_length):
                # cropping
                segments_list.append(self.amp_norm(audio_tensor[start:end]))
                start += seg_hop_p
                end += seg_hop_p
                num_segments += 1
            if end > audio_length:
                # padding the final segment
                padding = tf.zeros(int(end-audio_length), tf.float32)
                audio_tensor_end = tf.concat((
                    self.amp_norm(audio_tensor[start:end]), padding), axis=0)
                num_segments += 1
            else:
                audio_tensor_end = self.amp_norm(audio_tensor[start:end])
                num_segments += 1
            segments_list.append(audio_tensor_end)
            num_segments_list.append(num_segments)
        return segments_list, num_segments_list

    def calculate_melspectrogram(self, audio_tensor):
        '''Log-Mel-Spectrogram calculation based on tensorflow io
        Args:
            audio_tensor: input audio tensor
        Returns:
            output: tf.Tensor of mel-spectrogram feature
        '''

        def _comput_mel_delta(input, win_length=9):
            # input: [time, n_mels]
            tensor = tf.expand_dims(tf.transpose(input), -1)
            n = (win_length - 1) // 2
            denom = n * (n + 1) * (2 * n + 1) / 3
            kernel = tf.range(-n, n+1, dtype=tf.float32)
            kernel = tf.reshape(kernel, (win_length, 1, 1))
            output = tf.nn.conv1d(tensor, kernel, 1, 'SAME') / denom
            output = tf.transpose(output[:, :, 0])
            return output

        win_len_p = int(self.win_len * self.sr)
        win_hop_p = int(self.win_hop * self.sr)
        # spectrogram
        nfft = int(2**np.ceil(np.log(win_len_p) / np.log(2.0)))
        # nfft = win_len_p
        spectrogram = tfio.experimental.audio.spectrogram(
            input=audio_tensor,
            nfft=nfft,
            window=win_len_p,
            stride=win_hop_p
        )
        # melspectrogram
        mel_spectrogram = tfio.experimental.audio.melscale(
            input=spectrogram,
            rate=self.sr,
            mels=self.n_mels,
            fmin=0,
            fmax=self.sr/2
        )
        # dB scale
        db_mel_spectrogram = tfio.experimental.audio.dbscale(
            input=mel_spectrogram, top_db=90)
        if self.d_dd:
            # delta and delta-delta features
            mel_delta1 = _comput_mel_delta(db_mel_spectrogram)
            mel_delta2 = _comput_mel_delta(mel_delta1)
            # output
            output = tf.concat([
                tf.expand_dims(db_mel_spectrogram, -1),
                tf.expand_dims(mel_delta1, -1),
                tf.expand_dims(mel_delta2, -1)], axis=-1)
        else:
            # output
            output = tf.expand_dims(db_mel_spectrogram, -1)
        return output

    def __call__(self):
        with tf.device('/CPU:0'):
            # load the original audio signals
            audio_list = self.audio_data_prepare()
            # segmenting
            segments_list, num_segments_list = self.segmenting(audio_list)
            # make dataset
            audio_ds = tf.data.Dataset.from_tensor_slices(segments_list)
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = (tf.data
                .experimental.AutoShardPolicy.DATA)
            audio_ds = audio_ds.with_options(options)
            # calculate melspectrogram
            melspectrogram_ds = audio_ds.map(
                self.calculate_melspectrogram,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return melspectrogram_ds, num_segments_list


class Melspectrogram_augment_pipline(Melspectrogram_pipline):
    '''Prepare augmented melspectrogram data'''

    def __init__(self,
                 augment_mode,
                 specaugment='10,15,60',
                 **kwargs):
        '''Data augmentation \
        Args: \
            augment_mode: list of augment options \
                noise: add noise \
                reverberation: add reverberation \
                specaugment: SpecAugment \
        '''
        self.augment_mode = augment_mode
        saw, saf, sat = specaugment.split(',')
        self.saw, self.saf, self.sat = int(saw), int(saf), int(sat)
        super().__init__(**kwargs)
        self.seg_len_frame = int(self.seg_len / self.win_hop)
    
    def add_noise(self, audio_tensor):
        '''Randomly add noise with a SNR distributed from 0 to 30 dB'''
        # load noise file from DEMAND database
        folders = [
            'NRIVER', 'PRESTO', 'NPARK', 'SCAFE', 'OMEETING', 'NFIELD',
            'OOFFICE', 'PCAFETER', 'DWASHING', 'TMETRO', 'TCAR', 'DLIVING',
            'PSTATION', 'STRAFFIC', 'SPSQUARE', 'TBUS', 'OHALLWAY', 'DKITCHEN'
        ]
        noise_file = '../Database/DEMAND/{}/ch{:02d}.wav'.format(
            folders[np.random.randint(0, 18)], np.random.randint(1, 16))
        noise_tensor = self.load_audio(noise_file)
        # cut noise
        audio_length = tf.shape(audio_tensor)[0]
        noise_length = tf.shape(noise_tensor)[0]
        start = tf.random.uniform(
            (1,),
            maxval=noise_length-audio_length,
            dtype=tf.int32)[0]
        noise_tensor = noise_tensor[start:start+audio_length]
        # Root mean square
        rms = lambda tensor: tf.math.sqrt(
            tf.math.reduce_mean(tf.math.square(tensor)))
        audio_rms, noise_rms = rms(audio_tensor), rms(noise_tensor)
        # add noise
        snr = np.random.uniform(0, 30)  # SNR ~ uniform(0, 30)
        add_noise_rms = audio_rms / (10**(snr/20))
        add_noise_tensor = noise_tensor / noise_rms * add_noise_rms
        noisy_audio_tensor = audio_tensor + add_noise_tensor
        return noisy_audio_tensor

    def add_reverberation(self, audio_tensor):
        '''Randomly add reverberation on speech'''
        # load RIR
        num_rir_file = np.random.randint(0, self.rir_df.shape[0])
        rir_file = self.rir_df.iloc[num_rir_file].filepath
        rir_tensor = self.load_audio(rir_file)
        # convolution with tensorflow tf.nn.conv1d
        audio_tensor = tf.expand_dims(tf.expand_dims(audio_tensor, 0), -1)
        rir_tensor = tf.expand_dims(tf.expand_dims(rir_tensor, -1), -1)
        rir_length = 16000
        zero_padding_length = rir_length-1
        zero_padding0 = tf.zeros((1, int(np.ceil(zero_padding_length/2)), 1))
        zero_padding1 = tf.zeros((1, int(np.floor(zero_padding_length/2)), 1))
        audio_tensor = tf.concat(
            [zero_padding0, audio_tensor, zero_padding1], axis=1)
        reverberation_audio_tensor = tf.squeeze(
            tf.nn.conv1d(audio_tensor, rir_tensor[::-1], 1, 'SAME'))
        return reverberation_audio_tensor

    def time_warp(self, audio_tensor):
            
        def _point_transform(time_pt, freq_pt):
            pt = tf.ones((self.n_mels,), dtype=tf.int32) * time_pt
            pt = tf.expand_dims(tf.stack([pt, freq_pt], -1), 0)
            pt = tf.cast(pt, tf.float32)
            return pt

        freq_pt = tf.range(0, self.n_mels, dtype=tf.int32)
        source_pt = tf.random.uniform(
            (), self.saw+1, self.seg_len_frame-self.saw-1, tf.int32)
        target_pt = source_pt + tf.random.uniform(
            (), -self.saw, self.saw, tf.int32)
        # transform points
        source_pt = _point_transform(source_pt, freq_pt)
        target_pt = _point_transform(target_pt, freq_pt)
        warped_tensor, _ = tfa.image.sparse_image_warp(
            tf.expand_dims(audio_tensor, 0),
            source_pt, target_pt,
            num_boundary_points=3)
        return warped_tensor[0]

    def freq_mask(self, audio_tensor):
        f = tf.random.uniform((), 0, self.saf, tf.int32)
        f0 = tf.random.uniform((), 0, self.n_mels-f, tf.int32)
        m0 = audio_tensor[:, :f0, :]
        mask = tf.math.reduce_mean(audio_tensor, axis=(0, 1), keepdims=True)
        mask = tf.repeat(mask, repeats=self.seg_len_frame, axis=0)
        mask = tf.repeat(mask, repeats=f, axis=1)
        m1 = audio_tensor[:, f0+f:, :]
        freq_masked_tensor = tf.concat((m0, mask, m1), axis=1)
        return freq_masked_tensor

    def time_mask(self, audio_tensor):
        t = tf.random.uniform((), 0, self.sat, tf.int32)
        t0 = tf.random.uniform((), 0, self.seg_len_frame-t, tf.int32)
        m0 = audio_tensor[:t0, :, :]
        mask = tf.math.reduce_mean(audio_tensor, axis=(0, 1), keepdims=True)
        mask = tf.repeat(mask, repeats=t, axis=0)
        mask = tf.repeat(mask, repeats=self.n_mels, axis=1)
        m1 = audio_tensor[t0+t:, :, :]
        time_masked_tensor = tf.concat((m0, mask, m1), axis=0)
        return time_masked_tensor

    def audio_data_prepare(self):
        '''prepare the audio signals with augmentation'''
        audio_tensor_list = super().audio_data_prepare()
        audio_ds = tf.data.Dataset.from_generator(
            lambda: audio_tensor_list, output_types=tf.float32)
        if 'noise' in self.augment_mode:
            audio_ds = audio_ds.map(
                self.add_noise,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        if 'reverberation' in self.augment_mode:
            # BIRD RIR database
            rir_path = '../Database/BIRD/'
            self.rir_df = pd.read_csv(rir_path+'information.csv', index_col=[0])
            self.rir_df['filepath'] = self.rir_df.apply(
                lambda row: rir_path + row.filepath, axis=1)
            audio_ds = audio_ds.map(
                self.add_reverberation,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        audio_tensor_list = list(iter(audio_ds))
        return audio_tensor_list

    def __call__(self):
        melspectrogram_ds, num_segments_list = super().__call__()
        if 'specaugment' in self.augment_mode:
            with tf.device('/CPU:0'):
                # SpecAugment
                print('SpecAugment')
                melspectrogram_ds = melspectrogram_ds.map(
                    self.time_warp,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                melspectrogram_ds = melspectrogram_ds.map(
                    self.freq_mask,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                melspectrogram_ds = melspectrogram_ds.map(
                    self.freq_mask,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                melspectrogram_ds = melspectrogram_ds.map(
                    self.time_mask,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                melspectrogram_ds = melspectrogram_ds.map(
                    self.time_mask,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                return melspectrogram_ds, num_segments_list
        else:
            return melspectrogram_ds, num_segments_list


def label_weight(labels,
                 num_segments_list,
                 class_weights=None,
                 label_smooth=0,
                 **kwargs):
    '''generate labels and sample weights
    Args:
        labels: np.array of one-hot labels
        num_segments_list: list of the number of segments
        class_weights: class weight
            None: for training data the class weight will be calculated based \
                on the labels of training data
            list of class weights: for validation and test data the list class \
                weights calculated by from training data should be given
    Returns:
        outputs: dict of the labels and sample weights
            label: one-hot labels
            noweights: tf.ones tensor for training without sample weights
            segment: segmentation sample weights
            class: class sample weights
            segment_class: segmentation * calss sample weights
        class_weights: list of class weights
    '''
    labels = labels.astype('float32')
    num_data, num_class = labels.shape
    # label smooth
    alpha, beta = (1-label_smooth), (label_smooth/num_class)
    labels = labels * alpha + beta
    # class weight
    if not type(class_weights)==np.ndarray:
        class_weights = labels.sum() / labels.sum(0) / num_class
        print('class weight: ', class_weights)
    # outputs
    outputs = {
        'label': [],
        'noweight': [],
        'segment': [],
        'class': [],
        'segment_class': []
    }
    tfcon = lambda num: tf.constant(num, tf.float32, (1,))
    for n in range(num_data):
        # weights
        segment_weight = 1 / num_segments_list[n]
        class_weight = class_weights[labels[n].argmax()]
        segment_class_weight = segment_weight * class_weight
        for _ in range(int(num_segments_list[n])):
            # output
            outputs['label'].append(labels[n])
            outputs['noweight'].append(tfcon(1.0))
            outputs['segment'].append(tfcon(segment_weight))
            outputs['class'].append(tfcon(class_weight))
            outputs['segment_class'].append(tfcon(segment_class_weight))
    for key in list(outputs.keys()):
        outputs[key] = tf.data.Dataset.from_tensor_slices(outputs[key])
    return outputs, class_weights
