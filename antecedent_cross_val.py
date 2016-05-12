# coding=utf-8
__author__ = 'kian'

import numpy as np
from detect_antecedents import AntecedentClassifier
from sklearn.cross_validation import KFold

# antecedent classifier hyper parameters
K = 5
C = 5.0
LR = 0.01
EPOCHS = 2

def cross_validate(k_fold=5):
    ac = AntecedentClassifier(0, 14, 15, 19, 20, 24)
    ac.load_imported_data()
    ac.initialize_weights()

    ac.C = C
    ac.learn_rate = lambda x: LR

    all_trigs = np.array(ac.train_triggers + ac.val_triggers + ac.test_triggers)

    kf = KFold(len(all_trigs), n_folds=k_fold, shuffle=True, random_state=111)

    accs = []
    baseline_accs = []
    for train_idxs, test_idxs in kf:
        train = all_trigs[train_idxs]
        test = all_trigs[test_idxs]

        # need to create validation set to know how to decide on the best weights
        # we will take 1 / k_fold of the training set as the test set.
        val_size = (1.0 / k_fold) * len(train)
        val = train[:val_size]
        train = train[val_size:]

        ac.train_triggers = list(train)
        ac.val_triggers = list(val)
        ac.test_triggers = list(test)

        val_acc, test_acc = ac.fit(epochs=EPOCHS, k=K)
        accs.append((val_acc, test_acc))

        bval_acc, btest_acc = ac.baseline_prediction()
        baseline_accs.append((bval_acc, btest_acc))
        ac.reset()
        ac.initialize_weights()

    for lst in (baseline_accs, accs):
        print 'MIRA RESULTS' if lst == accs else 'BASLINE RESULTS'
        print 'Average val accuracy: ',np.mean([t[0] for t in lst])
        print 'Average test accuracy: ',np.mean([t[1] for t in lst])
        for tup in lst:
            print '\t',tup
        print

if __name__ == '__main__':
    cross_validate()
