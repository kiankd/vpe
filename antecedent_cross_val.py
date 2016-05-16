# coding=utf-8
__author__ = 'kian'

import numpy as np
import word_characteristics as wc
from detect_antecedents import AntecedentClassifier
from sklearn.cross_validation import KFold

AUTO_PARSE_NPY_DATA = 'antecedent_auto_parse_data.npy'

# antecedent classifier hyper parameters
K = 5
C = 5.0
LR = 0.01
EPOCHS = 2

def cross_validate(k_fold=5, type_=None, auto_parse=False):

    if auto_parse:
        ac = load_imported_data_for_antecedent()
    else:
        ac = AntecedentClassifier(0, 14, 15, 19, 20, 24)
        ac.load_imported_data()

    ac.initialize_weights()

    ac.C = C
    ac.learn_rate = lambda x: LR

    if type_:
        ac.set_trigger_type(type_, alter_train=True)

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

    results = []
    for lst in (baseline_accs, accs):
        end = '\n'
        s = ''
        if type_:
            s+= 'TRIGGER TYPE: ' + type_ + end
        s += 'MIRA RESULTS' if lst == accs else 'BASLINE RESULTS'
        s += end
        s += 'Average val accuracy: ' + str(np.mean([t[0] for t in lst])) + end
        s += 'Average test accuracy: ' + str(np.mean([t[1] for t in lst])) + end
        for tup in lst:
            s += '\t' + str(tup) + end
        s += end
        results.append(s)
        print s
    results.append('------------------------------------------------')
    return results

def log_results(results_lst):
    with open('ANT_CROSS_VALIDATION_RESULTS.txt', 'a') as f:
        for result_str in results_lst:
            f.write(result_str)

def save_imported_data_for_antecedent(classifier):
    """
    @type classifier: AntecedentClassifier
    """

    # for some reason we need to do this for numpy to not get in infinite loop...
    sent_words = []
    for sent in classifier.sentences:
        sent_words.append(sent.words)
        sent.words = None

    classifier.sentence_words = sent_words

    data = [classifier.sentences, classifier.train_triggers,
            classifier.val_triggers, classifier.test_triggers,
            classifier.sentence_words]

    np.save(AUTO_PARSE_NPY_DATA, np.array(data))

def load_imported_data_for_antecedent():
    ac = AntecedentClassifier(0, 14, 15, 19, 20, 24)

    data = np.load(AUTO_PARSE_NPY_DATA)

    ac.sentences = data[0]
    ac.train_triggers = data[1]
    ac.val_triggers = data[2]
    ac.test_triggers = data[3]

    for i,sentwords in enumerate(data[4]):
        ac.sentences[i].words = sentwords

    return ac

if __name__ == '__main__':

    # ac = AntecedentClassifier(0,14,15,19,20,24)
    # ac.import_data(get_mrg=False)
    # save_imported_data_for_antecedent(ac)
    # ac.generate_possible_ants(['VP', wc.is_predicative, wc.is_adjective, wc.is_verb])
    # ac.build_feature_vectors()
    # ac.normalize()
    # save_imported_data_for_antecedent(ac)
    #
    # exit(0)

    for type_ in [None,'do','be','to','modal','have','so']:
        results_lst = cross_validate(type_=type_, auto_parse=True)
        log_results(results_lst)
