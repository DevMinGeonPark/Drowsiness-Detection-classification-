# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 08:27:44 2018

@ author: Chang-Hee Han
@ Comments: Cropped Decoding
"""



""" (1) Enable logging """
''' 로그 정보를 저장하는 파트 '''
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)



""" (2) Load data """
''' nme 데이터를 로드해서 발췌하는 부분'''
import mne #nme 튜토리얼 데이터 셋
from mne.io import concatenate_raws

# 5,6,7,10,13,14 are codes for executed and imagined hands/feet
#실행 및 상상된 손과 발에 대한 코드번호가 5 6 ...이다.
subject_id = 22
event_codes = [5,6,9,10,13,14]
#event_codes = [3,4,5,6,7,8,9,10,11,12,13,14]

# This will download the files if you don't have them yet,
# and then return the paths to the files.
#data load file

physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

# Load each of the files
#data load file edf
parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto', verbose='WARNING')
         for path in physionet_paths]

# Concatenate them
raw = concatenate_raws(parts)

# Find the events in this dataset
events, _ = mne.events_from_annotations(raw)

# Use only EEG channels
eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Extract trials, only using EEG channels
epoched = mne.Epochs(raw, events, dict(hands_or_left=2, feet_or_right=3), 
                     tmin=1, tmax=4.1, proj=False, picks=eeg_channel_inds,
                baseline=None, preload=True)



""" (3) Convert data to Braindecode format """
import numpy as np
# Convert data from volt to millivolt
# Pytorch expects float32 for input and int64 for labels.
X = (epoched.get_data() * 1e6).astype(np.float32)
y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1

from braindecode.datautil.signal_target import SignalAndTarget
train_set = SignalAndTarget(X[:40], y=y[:40])
valid_set = SignalAndTarget(X[40:70], y=y[40:70])
#test_set = SignalAndTarget(X[70:], y=y[70:])



""" (4) Creat model """
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from torch import nn
from braindecode.torch_ext.util import set_random_seeds

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
# torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(), 
# torch.cuda.get_device_name(0), torch.cuda.device(0)
cuda = True #gpu mode
set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 2
in_chans = train_set.X.shape[1]

model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=None,
                        final_conv_length=12)
if cuda:
    model.cuda()

#최적화 관련
from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F
#optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
model.compile(loss=F.nll_loss, optimizer=optimizer,  iterator_seed=1, cropped=True)



""" (5) Run the training """
input_time_length = 450
model.fit(train_set.X, train_set.y, epochs=30, batch_size=64, scheduler='cosine',
          input_time_length=input_time_length,
          validation_data=(valid_set.X, valid_set.y),)
        
model.epochs_df



""" (6) Evaluation """
#TestPerform = model.evaluate(test_set.X, test_set.y)
#Acc = 1 - TestPerform['misclass']
#model.predict_classes(test_set.X)
#model.predict_outs(test_set.X)

"""원래코드"""
test_set = SignalAndTarget(X[70:],y=y[70:])
model.evaluate(test_set.X,test_set.y)
model.predict_classes(test_set.X)
model.predict_outs(test_set.X)



""" (7) Accuracies per timestep """
"""그림을 그려보는 것"""
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
%matplotlib inline
matplotlib.style.use('seaborn')
labels_per_trial_per_crop = model.predict_classes(test_set.X, individual_crops=True)
accs_per_crop = [l == y for l,y in zip(labels_per_trial_per_crop, test_set.y)]
accs_per_crop = np.mean(accs_per_crop, axis=0)
plt.figure(figsize=(8,3))
plt.plot(accs_per_crop * 100)
plt.title("Accuracies per timestep", fontsize=16)
plt.xlabel('Timestep in trial', fontsize=14)
plt.ylabel('Accuracy [%]', fontsize=14)