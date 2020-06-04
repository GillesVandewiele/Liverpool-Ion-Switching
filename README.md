# Liverpool-Ion-Switching
Third place solution to the Kaggle "Liverpool Ion Switching" competition.

By team [Gilles](https://www.kaggle.com/group16) & [Kha Vo](https://www.kaggle.com/khahuras/) & [Zidmie](https://www.kaggle.com/zidmie).

## Getting the data

We will be using [Chris Deotte](https://www.kaggle.com/cdeotte) his [excellent dataset](https://www.kaggle.com/cdeotte/data-without-drift). Download these from Kaggle and put the `train_clean.csv` and `test_clean.csv` in a `data/` folder. Moreover, we also require a `sample_submission.csv` file which can be downloaded from the Kaggle competition page. Your directory structure should look as follows:
```
.
├── data
│   ├── sample_submission.csv
│   ├── test_clean.csv
│   └── train_clean.csv
├── hmm.py
├── main.py
├── processing.py
├── notebooks
│   ├── 1 - Align Channels and Signal.ipynb
│   ├── 2 - Remove Power Line Interference.ipynb
│   ├── 3 - Fit 4-state HMM (Cat 3).ipynb
│   ├── 4 - Setting the Transition Matrix.ipynb
│   ├── 5 - Fit 20-state HMM (Cat 3).ipynb
│   ├── 6 - Custom Forward-Backward (Cat 3).ipynb
│   └── 7 - Prediction Post-Processing (Cat 3).ipynb
├── output
├── LICENSE
├── README.md
└── requirements.txt

```

## Required hardware
This code ran perfectly in Kaggle Notebooks, which has:
* Ubuntu 18.04.4 LTS (Bionic Beaver)
* 4 cores: Intel(R) Xeon(R) CPU @ 2.30GHz
* around 16 GB available RAM 

## Installing requirements
We provide a `requirements.txt` file to install the dependencies through pip. Only `pandas`, `numpy`, `scipy` and `scikit-learn` (the final one only for its f1_score function) are the minimal requirements. All others are needed for the [notebooks](#notebooks).

## Producing our submission

We provide a `main.py` script. It will iteratively improve the submission and write away the results to the `output/` directory. In each iteration, it removes the power line interference (for which it uses "out-of-fold" predictions) and fits a HMM on the train and test set (batches of 100K). In the first iteration, the power line interference is skipped.

The following can be copy-pasted to a Kaggle notebook (or Google Colab):
```
!git clone https://github.com/GillesVandewiele/Liverpool-Ion-Switching.git
!mkdir Liverpool-Ion-Switching/data
!cp ../input/data-without-drift/train_clean.csv Liverpool-Ion-Switching/data/train_clean.csv
!cp ../input/data-without-drift/test_clean.csv Liverpool-Ion-Switching/data/test_clean.csv
!cp ../input/liverpool-ion-switching/sample_submission.csv Liverpool-Ion-Switching/data/sample_submission.csv
!cd Liverpool-Ion-Switching; python3 main.py
```

## Notebooks

We provide notebooks that elaborate upon each of the 7 significant steps in our approach:

* [1. Aligning the signal and channels with linear regression](notebooks/1%20-%20Align%20Channels%20and%20Signal.html)

We align the signal and channels. A simple baseline which just rounds the signal values scores an F1 of `0.9211271639823664`

* [2. Removing power line interference](notebooks/2%20-%20Remove%20Power%20Line%20Interference.html)

We remove 50 Hz power line interference. This slightly improves the F1 of our baseline: `0.9250468415867467`

* [3. Hidden Markov Models: a naive approach](notebooks/3%20-%20Fit%204-state%20HMM%20(Cat%203).html)

We show how a naive approach of a Hidden Markov Model already increased the F1 significantly. The category 3 F1 score for our baseline approach is `0.9738199736256037` while a Hidden Markov Model with 4 hidden states scores an F1 of `0.9840563515575094`.

* [4. Setting the Ptran variables](notebooks/4%20-%20Setting%20the%20Transition%20Matrix.html)

We show how you could go about tuning the Ptran which is needed for further steps

* [5. K Independent Hidden Markov Models](notebooks/5%20-%20Fit%2020-state%20HMM%20(Cat%203).html)

We show that, by assumining K independent binary Markov Processes, for data that goes up to K open channels, that we can significantly increase the F1. We show this for category 3 of our data, where we expand a 4x4 transition matrix used to model category 2 to a 20x20 matrix. The achieved F1 score, using only category 3 data, is `0.9866748988341756`.

* [6. A custom forward-backward (inference) algorithm](notebooks/6%20-%20Custom%20Forward-Backward%20(Cat%203).html)

We adapt the forward-backward algorithm to work both faster and slightly more accurate. The F1 score on category 3 data is `0.986794704167445`. The impact, in terms of F1 score, is more significant for category 4 and 5 of the data (which were the most important ones).

* [7. Post-processing the posterior probabilities](notebooks/7%20-%20Prediction%20Post-Processing%20(Cat%203).html)

We convert the posterior probabilities (more probabilities than the number of classes) to a continuous value by taking the dot product between the probabilities and the open channels to which each respective hidden state corresponds to. We then learn thresholds, again in an unsupervised manner, to convert these continuous values to a discrete number of open channels. This increases the F1 for our category 3 data to `0.9869704508621362`.

## References & Pointers

* [The kaggle competition](https://www.kaggle.com/c/liverpool-ion-switching)

* [A detailed blog post on our solution](https://towardsdatascience.com/identifying-the-number-of-open-ion-channels-with-hidden-markov-models-334fab86fc85)

* [A notebook by Kha that uses an already processed signal & strong oofs](https://www.kaggle.com/khahuras/1st-place-non-leak-solution)

* [A leak in the private data that can get you a up to 0.04+ boost (LB ~ 0.985)](https://www.kaggle.com/group16/private-0-9688-a-better-but-useless-solution)

* [A paper by the organizers](https://www.nature.com/articles/s42003-019-0729-3)

## Contributing

We welcome any kind of contributions. Whether that be cleaning up some of the code, extra documentation, or anything else. Please feel free to open a pull request!
