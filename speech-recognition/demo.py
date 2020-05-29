import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from sklearn.model_selection import train_test_split

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1,mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2,mode="nearest")
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix


def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav")]
    return mfcc




class_names = ['khong', 'toi', 'co', 'amtinh', 'nguoi']
dataset = {}
dataset_train = {}
dataset_test = {}

for cname in class_names:
    print(f"Load {cname} dataset")
    dataset[cname] = get_class_data(os.path.join("data", cname))
    train_size = int(0.8*len(dataset[cname]))
    dataset_train[cname] = dataset[cname][:train_size]
    dataset_test[cname] = dataset[cname][train_size:]




models = {}

for cname in class_names:
    
    hmm = hmmlearn.hmm.GMMHMM(
        n_components=6, n_mix = 2, random_state=42, n_iter=1000, verbose=True,
        params='mctw',
        init_params='mst'
    )
    hmm.startprob_ = np.array([1.0,0.0,0.0,0.0,0.0, 0.0,0.0])
    hmm.transmat_ = np.array([
            [0.7,0.2,0.1,0.0,0.0,0.0,0.0],
            [0.0,0.7,0.2,0.1,0.0,0.0,0.0],
            [0.0,0.0,0.7,0.2,0.1,0.0,0.0],
            [0.0,0.0,0.0,0.7,0.2,0.1,0.0],
            [0.0,0.0,0.0,0.0,0.7,0.2,0.1],
            [0.0,0.0,0.0,0.0,0.0,0.7,0.3],
            [0.0,0.0,0.0,0.0,0.0,0.0,1.0],
        ])

    X = np.concatenate(dataset_train[cname])
    lengths = list([len(x) for x in dataset_train[cname]])
    hmm.fit(X)
    models[cname] = hmm
print("Training done")


print("Testing")

#class_names = ['khong', 'toi', 'trong', 'amtinh',"test_toi","test_trong","test_khong","test_amtinh"]
for true_cname in class_names:
    #if true_cname[:4] == 'test':
    true_predict = 0
#     for O in dataset[true_cname]:
    for O in dataset_test[true_cname]:
        score = {cname : model.score(O, [len(O)]) for cname, model in models.items() if cname[:4] != 'test' }
        print(true_cname, score)
        predict = max(score, key=score.get)
        if predict == true_cname or predict == true_cname[5:]:
            true_predict += 1
#         print(true_cname, score, predict)
    print(true_cname)
#     change dataset_test to dataset to test in full dataset
    print(f'TRUE PREDICT: {true_predict}/{len(dataset_test[true_cname])}')
    print('ACCURACY:', true_predict/len(dataset_test[true_cname]))                             


