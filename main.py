import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import sys
sys.path.append('Liverpool-Ion-Switching')

import processing, hmm


train = pd.read_csv('data/train_clean.csv')
test = pd.read_csv('data/test_clean.csv')
signal = np.concatenate((train['signal'].values, test['signal'].values))
signal = processing.align(signal)

good_id = list(range(3600000))+list(range(3900000,5000000))
f1 = f1_score(train['open_channels'].values[good_id], 
              np.round(train['signal'].values[good_id]), 
              average='macro')
Ys = None
converged = False
TOL = 1e-5


for _ in range(5):

    # Hyper-parameters
    Kexp = [.103, .120, .1307, .138, .267, .105] 
    Kexpp =  [1.8,  1.8,  1.8,   1.83, 1.807, 1.8]
    N_PROCESSES = [1, 1, 3, 5, 10, 1]
    COEFS_BACK = [1, .9192, .9192, .8792, .9022, .9192]
    COEFS_FOR = [1, .8869, .8869, .8869, .8849, .8869]
    COEFS_FIN = [.618, 0.50, 0.50,  0.49, 0.509, 0.50]
    COEFS_FIN3 = [0.3,  0.3,  0.3, 0.35, 0.335, 0.3]

    # Batches (per 100K) and corresponding categories:
    # category 1 = open-channels 0/1 (mostly 0) (not important)
    # category 2 = open-channels 0/1 (mostly 1) (not important)
    # category 3 = open-channels 0-3
    # category 4 = open-channels 0-5
    # category 5 = open-channels 0-10 (most important)
    # category 6 = open-channels 0-4 (not many 2/3/4)
    BATCHES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51, 52, 
                        53, 54, 55, 56, 57, 58, 59, 60, 65, 70])
    CATEGORIES = np.array([1, 1, 2, 3, 5, 4, 2, 3, 4, 5, 6, 3, 4, 6,  
                           2, 5, 4, 5, 6, 3, 6, 6])

    # Our transition matrices (and the corresponding open channels 
    # of each hidden state)
    all_Ptran = [
        np.array([
            [0     , 0.1713   , 0   , 0      ],
            [0.3297, 0        , 0   , 0.01381],
            [0     , 1        , 0   , 0      ],
            [0     , 0.0002686, 0   , 0      ]
        ]),
        np.array([
            [0     , 0.0121, 0     , 0     ],
            [0.0424, 0     , 0.2766, 0.0101],
            [0     , 0.2588, 0     , 0     ],
            [0     , 0.0239, 0     , 0     ]
        ]),
        np.array([
            [0     , 0.0067, 0     , 0     ],
            [0.0373, 0     , 0.2762, 0.0230],
            [0     , 0.1991, 0     , 0     ],
            [0     , 0.0050, 0     , 0     ]
        ]),
        np.array([
            [0     , 0.0067, 0     , 0     ],
            [0.0373, 0     , 0.2762, 0.0230],
            [0     , 0.1991, 0     , 0     ],
            [0     , 0.0050, 0     , 0     ]
        ]),
        np.array([
            [0     , 0.0067, 0     , 0     ],
            [0.0373, 0     , 0.2762, 0.0230],
            [0     , 0.1991, 0     , 0     ],
            [0     , 0.0050, 0     , 0     ]
        ]),
        np.array([
            [0.        , 0         , 0.34493706, 0.00287762, 0.00006045, 0         ],
            [0         , 0.        , 0.00040108, 0         , 0         , 0         ],
            [0.16435428, 0.00438756, 0.        , 0.01714043, 0.00023227, 0         ],
            [0.02920171, 0.00080145, 0.27065939, 0.        , 0.01805161, 0.00108684],
            [0.00268151, 0.00000064, 0.06197474, 0.30666751, 0.        , 0.06625158],
            [0         , 0         , 0.00000136, 0.13616454, 0.51059444, 0         ]
        ])
    ]

    all_States = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 2, 3, 4]
    ]
    blocks = [
        list(range(1000000)),
        list(range(1000000, 1500000)) + list(range(3000000, 3500000)),
        list(range(1500000, 2000000)) + list(range(3500000, 3600000)) + list(range(3900000, 4000000)),
        list(range(2500000, 3000000)) + list(range(4000000, 4500000)),
        list(range(2000000, 2500000)) + list(range(4500000, 5000000)),
    ]
    sub = pd.read_csv('data/sample_submission.csv')

    train = pd.read_csv('data/train_clean.csv')
    test = pd.read_csv('data/test_clean.csv')
    signal = np.concatenate((train['signal'].values, test['signal'].values))
    signal = processing.align(signal)

    # If we have predictions, than remove sine from noise
    cleaned_signal = signal
    if Ys is not None:
        cleaned_signal = processing.remove_power_line(cleaned_signal, Ys)

    full_pred = np.zeros(7_000_000)

    for c in [0,1,2,3,4,5]:

        print("Training cat", c)
        kexp = Kexp[c]
        kexpp = Kexpp[c]
        coefback = COEFS_BACK[c]
        coeffor = COEFS_FOR[c]
        coef_fin = COEFS_FIN[c]
        coef_fin3 = COEFS_FIN3[c]
        Ptran, States = hmm.calculate_matrix(all_Ptran[c], all_States[c], 
                                             N_PROCESSES[c])

        for jb, b in enumerate(BATCHES):
            if b == 70 or CATEGORIES[jb]!=c+1: continue
            end_b = BATCHES[jb+1] if b!=65 else 70
            sig = cleaned_signal[100_000*b:100_000*end_b]
            nstates = Ptran.shape[0]
            
            for k in range(len(sig) // 100_000):
                sub_sig = sig[100_000*k:(k+1)*100_000]
                
                Psig = hmm.get_Psig(sub_sig, States, kexp)
                alpha0, etat0 = hmm.forward(Psig, Ptran, normalize=False)
                alpha1, etat1 = hmm.forward(Psig[::-1], np.transpose(Ptran), 
                                        etat_in=etat0[::-1], coef=coefback)
                alpha2, etat2 = hmm.forward(Psig, Ptran, etat_in=etat1[::-1], 
                                        coef=coeffor)

                alpha3 = etat1[::-1]*etat2*Psig**kexpp
                for j, alp in enumerate(alpha3): 
                    alpha3[j] /= alp.sum()

                pred = coef_fin*(alpha1[::-1]) + (1-coef_fin-coef_fin3)*alpha2 + coef_fin3*alpha3

                full_pred[(b + k)*100_000:(b + k + 1)*100_000] = pred @ States

    Ys = full_pred.copy()
    Yopt, Thres = processing.optimize_thres_unsupervised(Ys.copy(), 
                                                         cleaned_signal)
    
    # Calculate our new F1
    new_f1 = f1_score(train['open_channels'].values[good_id], Yopt[good_id], 
                      average='macro')
    if new_f1 < f1 or new_f1 - f1 < TOL:
        converged = True
    else:
        f1 = new_f1
        sub['open_channels'] = np.round(Yopt[5_000_000:]).astype(int)
        sub.to_csv('output/submission_{}.csv'.format(np.round(f1, 5)), 
                   index=False, float_format='%.4f')

    for i, block in enumerate(blocks):
        Y_block = Yopt[block]
        chan = train['open_channels'].values[block]

        labels = list(range(0, max(chan) + 1))
        if i == 4:
            labels = labels[1:]

        print('Category #{} F1 = {}'.format(
            i + 1, 
            f1_score(chan, np.round(Y_block), average='macro', labels=labels)
        ))

    print('Total F1 = {}'.format(
        f1_score(train['open_channels'].values[good_id], Yopt[good_id], 
                 average='macro')
    ))