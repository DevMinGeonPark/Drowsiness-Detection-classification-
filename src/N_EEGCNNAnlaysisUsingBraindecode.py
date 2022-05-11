# -*- coding: utf-8 -*-

"""
튜토리얼 코드
Created on Thu Aug  9 14:04:27 2018

@ Name: Chang-Hee Han (Ph.D)
@ Position: Assistant Professor
@ Affiliation: Dept. of Software, Dongseo University
@ Email: changheehan@dongseo.ac.kr
@ HP: +82 10 8074 4974
@ Google Scholar: https://goo.gl/Bu7vde
@ Homepage: https://zeros8706.wixsite.com/changheehan/

"""


###############################################################################
### (1) Enable logging ########################################################
###############################################################################
import warnings
warnings.simplefilter('ignore') #워닝 메세지 무시에 관련된 코

#
Timesegment = 10
Fs = 200


###############################################################################
### (2) Load data #############################################################
###############################################################################
# set up data path
import os
default_path = 'D:/Works/2021/업무/3_한양대학교/210507_KHBM_Morning Workshop/examples'
os.chdir(default_path)
del default_path

# load raw data and labels
import numpy as np
Final_Acc = np.zeros(2)
for temp_subNum in range(1, 3):
    print(temp_subNum)

    from scipy import io
    raw_data = io.loadmat('sub_' + str(temp_subNum) + 'Day2_200508.mat')
    raw_labels = io.loadmat('sub_' + str(temp_subNum) + 'Day2_200508_ylabel.mat')
    FeatVect = raw_data['rawData']
    # 3-dimentiona array = [trials x channels x time-series]
    Timesegment = 10
    Fs = 200
    FeatVect = FeatVect[:, :, 0:(Timesegment*Fs)]
    y_labels = raw_labels['y_label']
    del raw_data, raw_labels
    
    
    ###########################################################################
    ### (3) Preprocessing #####################################################
    ###########################################################################
    from braindecode.datautil.signalproc import lowpass_cnt, highpass_cnt, exponential_running_standardize
    for ii in range(0, 60):
        # 1. Data reconstruction
        temp_data = FeatVect[ii, :, :]
        temp_data = temp_data.transpose()
        # 2. Lowpass filtering
        lowpassed_data = lowpass_cnt(temp_data, 13, 200, filt_order=3)
        # 3. Highpass filtering
        bandpassed_data = highpass_cnt(lowpassed_data, 8, 200, filt_order=3)
        # 4. Exponential running standardization
        ExpRunStand_data = exponential_running_standardize(bandpassed_data, 
                                                           factor_new=0.001, 
                                                           init_block_size=None, 
                                                           eps=0.0001)
        # 5. Renewal preprocessed data
        ExpRunStand_data = ExpRunStand_data.transpose()
        FeatVect[ii, :, :] = ExpRunStand_data
        #del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data
        del temp_data, ExpRunStand_data
    
    
    ###########################################################################
    ### (3) Convert data to braindecode format ################################
    ###########################################################################
    X = (FeatVect).astype(np.float32)
    y = (y_labels).astype(np.int64)
    y = y[:, 0]
    del FeatVect, y_labels
    
    from braindecode.datautil.signal_target import SignalAndTarget
    from braindecode.datautil.splitters import split_into_two_sets
    
    ### 10-flod Cross validation ###
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=36851234)
    CV_cnt = 0
    CV_Acc = np.zeros(4)
    for train_index, test_index in rskf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        train_set = SignalAndTarget(X_train, y_train)
        train_set, valid_set = split_into_two_sets(train_set, first_set_fraction=0.8)
        test_set = SignalAndTarget(X_test, y_test)
        
        
        #######################################################################
        ### (4) Create the model ##############################################
        #######################################################################
        from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
        #from braindecode.models.shallow_fbcsp_forNIRS import ShallowFBCSPNet_forNIRS
        from braindecode.torch_ext.util import set_random_seeds
        
        # Set if you want to use GPU
        # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
        cuda = True
        set_random_seeds(seed=20170629, cuda=cuda)
        n_classes = 2
        in_chans = train_set.X.shape[1]
        
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, 
                                input_time_length=None, n_filters_time=40,
                                filter_time_length=25, n_filters_spat=40, 
                                pool_time_length=30,
                                pool_time_stride=4, final_conv_length=12,
                                pool_mode='mean', split_first_layer=True,
                                batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5)
        
        
        if cuda:
            model.cuda()
            
            
        #######################################################################
        ### (5) Create cropped iterator #######################################
        #######################################################################
        from braindecode.torch_ext.optimizers import AdamW
        import torch.nn.functional as F
        #optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
        optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
        model.compile(loss=F.nll_loss, optimizer=optimizer,  iterator_seed=1, cropped=True)
        
        
        #######################################################################
        ### (6) Run the training ##############################################
        #######################################################################
        input_time_length = Timesegment*Fs
        model.fit(train_set.X, train_set.y, epochs=30, batch_size=8, scheduler='cosine',
                  input_time_length=input_time_length, remember_best_column='valid_misclass',
                  validation_data=(valid_set.X, valid_set.y),)
        
        
        #######################################################################
        ### (7) Evaluation ####################################################
        #######################################################################
        scores = model.evaluate(test_set.X, test_set.y)
        
        CV_Acc[CV_cnt] = scores['misclass']
        CV_cnt = CV_cnt + 1
        
        del scores, optimizer, model, input_time_length, in_chans, n_classes, X_train, X_test, y_train, y_test, train_set, valid_set, test_set
    
    
    Final_Acc[temp_subNum - 1] = 1 - sum(CV_Acc) / len(CV_Acc)
     
    del CV_Acc, y, X
    
    print(Final_Acc[temp_subNum - 1])

print(sum(Final_Acc)/len(Final_Acc))

Final_Acc = Final_Acc*100