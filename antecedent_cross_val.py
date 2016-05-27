# coding=utf-8
__author__ = 'kian'

import numpy as np
import word_characteristics as wc
from detect_antecedents import AntecedentClassifier
from sklearn.cross_validation import KFold
from sys import platform, argv
from random import shuffle
from load_data import find_section


AUTO_PARSE_NPY_DATA      = 'antecedent_auto_parse_data_FULL_DATASET.npy'
GOLD_PARSE_FULL_NPY_DATA = 'antecedent_GOLD_parse_data_FULL_DATASET.npy'
AUTO_PARSE_ALL_ANTS_NPY  = 'antecedent_auto_parse_ALL_ANTS.npy'

if platform == 'linux2':
    AUTO_PARSE_NPY_DATA      = '../npy_data/' + AUTO_PARSE_NPY_DATA
    GOLD_PARSE_FULL_NPY_DATA = '../npy_data/' + GOLD_PARSE_FULL_NPY_DATA
    AUTO_PARSE_ALL_ANTS_NPY  = '../npy_data/' + AUTO_PARSE_ALL_ANTS_NPY

# antecedent classifier hyper parameters
K = 5
C = 5.0
LR = 0.01
EPOCHS = 4
seed = 347890

for arg in argv:
    if arg.startswith('seed='):
        seed = int(arg.split('seed=')[1])
        break

def init_classifier(auto_parse=True):
    ac = load_classifier(auto_parse=auto_parse, fname=AUTO_PARSE_ALL_ANTS_NPY if auto_parse else GOLD_PARSE_FULL_NPY_DATA)
    ac.initialize_weights(seed=seed)
    ac.C = C
    ac.learn_rate = lambda x: LR
    return ac

def cross_validate(k_fold=5, type_=None, auto_parse=False, classifier=None):
    if classifier is not None:
        ac = classifier
    else:
        ac = load_classifier(auto_parse=auto_parse, fname=AUTO_PARSE_ALL_ANTS_NPY)

    ac.initialize_weights()

    ac.C = C
    ac.learn_rate = lambda x: LR

    if type_:
        ac.set_trigger_type(type_, alter_train=True)

    all_trigs = np.array(ac.train_triggers + ac.val_triggers + ac.test_triggers)

    kf = KFold(len(all_trigs), n_folds=k_fold, shuffle=True, random_state=848613439)

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

        val_acc, test_acc, val_preds, test_preds = ac.fit(epochs=EPOCHS, k=K)
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

def bos_compare():
    ac = init_classifier()

    train_secs = [0,1,2,3,4,5,6,7,8,10,12,14]
    test_secs = [9,11,13,15]
    section_ends = {0: 964, 1: 1538, 2: 2461, 3: 2966, 4: 3661, 5: 4402, 6: 4843, 7: 5721, 8: 6137, 9: 6915, 10: 7606,
                    11: 8251, 12: 8851, 13: 9621, 14: 10348, 15: 11206, 16: 12337, 17: 12906, 18: 13510, 19: 14152,
                    20: 14704, 21: 15416, 22: 16419, 23: 17112, 24: 17925}
    # section_ends = {item:key for key,item in section_ends.iteritems()}

    temp_train = [] # 81 trigs
    temp_test = [] # 39 trigs
    for trig in ac.itertrigs():
        if trig.type == 'do':
            section = None

            for sec in range(0, 25):
                if trig.sentnum < section_ends[sec]:
                    section = sec
                    break

            if section in train_secs:
                temp_train.append(trig)

            if section in test_secs:
                temp_test.append(trig)

    shuffle(temp_train)
    div = int(0.8 * len(temp_train))
    temp_val = temp_train[div:]
    temp_train = temp_train[:div]

    ac.train_triggers = temp_train
    ac.val_triggers = temp_val
    ac.test_triggers = temp_test

    val_acc, test_acc = ac.fit(epochs=EPOCHS, k=K)
    bval_acc, btest_acc = ac.baseline_prediction()

    print 'MIRA RESULTS:'
    print val_acc, test_acc
    print '\nBASELINE:'
    print bval_acc, btest_acc

    return

def bos_spen_split():
    ac = init_classifier()

    if 'hardt' in argv:
        ac = set_classifier_features_to_hardt(ac)
        ac.initialize_weights(seed=seed)

    ac.debug_ant_selection()

    # train_secs = range(0,15)
    # val_secs = range(15,20)
    # test_secs = range(20,25)
    # section_ends = {0: 964, 1: 1538, 2: 2461, 3: 2966, 4: 3661, 5: 4402, 6: 4843, 7: 5721, 8: 6137, 9: 6915, 10: 7606,
    #                 11: 8251, 12: 8851, 13: 9621, 14: 10348, 15: 11206, 16: 12337, 17: 12906, 18: 13510, 19: 14152,
    #                 20: 14704, 21: 15416, 22: 16419, 23: 17112, 24: 17925}
    #
    # train, val, test = [], [], []
    #
    # for trig in ac.itertrigs():
    #     section = find_section(trig.sentnum, section_ends)
    #
    #     if section in train_secs:
    #         train.append(trig)
    #
    #     if section in val_secs:
    #         val.append(trig)
    #
    #     if section in test_secs:
    #         test.append(trig)
    #
    # ac.train_triggers = train
    # ac.val_triggers = val
    # ac.test_triggers = test

    val_acc, test_acc, val_preds, test_preds = ac.fit(epochs=EPOCHS, k=K)
    bval_acc, btest_acc = ac.baseline_prediction()

    print 'MIRA RESULTS:'
    print val_acc, test_acc
    print '\nBASELINE:'
    print bval_acc, btest_acc

    return

def ablation_study(auto_parse=False, exclude=True):
    # This is the division of features by their class:
    # first excludes the alignment features,
    # next exclude relational features
    # next exclude ant_trig relation features
    # last exclude hardt/nielsen feats
    feat_dict = {(139,404):'alignment',
                 (1,139,143,404):'relational',
                 (1,143,201,404):'ant_trig',
                 (1,201):'hardt'}

    for tup in feat_dict.iterkeys():
        ac = load_classifier(auto_parse=auto_parse, fname=AUTO_PARSE_ALL_ANTS_NPY)

        print 'Current excluded feature:',feat_dict[tup]
        print 'Using tuple: ',tup
        for trig in ac.itertrigs():
            for ant in trig.possible_ants + [trig.gold_ant]:
                l = list(ant.x)

                if exclude:
                    ant.x = l[tup[0]:tup[1]]
                    if len(tup) == 4:
                        ant.x += l[tup[2]:tup[3]]
                else:
                    if len(tup) == 2:
                        ant.x = l[1:tup[0]] + l[tup[1]:]
                    else:
                        ant.x = l[tup[1]:tup[2]]

                ant.x = np.array([1] + ant.x)

        results = ['----\nFeature: %s\n' % feat_dict[tup]] + ['EXCLUDED' if exclude else 'INLCUDED', '\n'] \
                  + cross_validate(auto_parse=auto_parse, classifier=ac)
        log_results(results, fname='feature_ablation_ant_laxloss_%s.txt'%('EXCLUDED' if exclude else 'INCLUDED'))

def log_results(results_lst, fname='ANT_CROSS_VALIDATION_RESULTS.txt'):
    with open(fname, 'a') as f:
        for result_str in results_lst:
            f.write(result_str)

def load_classifier(auto_parse=False, fname=None):
    if auto_parse:
        ac = load_imported_data_for_antecedent(fname=fname)
    else:
        ac = AntecedentClassifier(0, 14, 15, 19, 20, 24)
        ac.load_imported_data(fname=(AUTO_PARSE_NPY_DATA if auto_parse else GOLD_PARSE_FULL_NPY_DATA))
    return ac

def save_imported_data_for_antecedent(classifier, fname=AUTO_PARSE_NPY_DATA):
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

    np.save(fname, np.array(data))

def load_imported_data_for_antecedent(fname=AUTO_PARSE_NPY_DATA):
    ac = AntecedentClassifier(0, 14, 15, 19, 20, 24)

    print 'Loading NPY data from this file:',fname

    data = np.load(fname)

    ac.sentences = data[0]
    ac.train_triggers = data[1]
    ac.val_triggers = data[2]
    ac.test_triggers = data[3]

    for i,sentwords in enumerate(data[4]):
        ac.sentences[i].words = sentwords

    return ac

if __name__ == '__main__':
    mrg = 'mrg' in argv
    save_file = GOLD_PARSE_FULL_NPY_DATA if mrg else AUTO_PARSE_ALL_ANTS_NPY
    results_save = 'final_results_ALL_TYPES_ALL_ANTS.txt'

    if 'gen2' in argv:
        AUTO_PARSE_ALL_ANTS_NPY = AUTO_PARSE_ALL_ANTS_NPY[:-4]+'_antgen2.npy'
        results_save = results_save[:-4]+'_antgen2.txt'

    if 'build' in argv:
        # ac = AntecedentClassifier(0,14,15,19,20,24)
        # ac.import_data(get_mrg=mrg)
        # save_imported_data_for_antecedent(ac, fname=save_file)
        # exit(0)

        ac = load_imported_data_for_antecedent(fname=save_file)
        ac.generate_possible_ants2()
        # ac.generate_possible_ants(['VP', wc.is_predicative, wc.is_adjective, wc.is_verb], filter=True)
        ac.debug_ant_selection()
        # exit(0)
        ac.build_feature_vectors()
        ac.normalize()
        save_imported_data_for_antecedent(ac, fname=save_file[:-4]+'_antgen2.npy')

    if 'types' in argv:
        for type_ in [None,'do','be','to','modal','have','so']:
            ac = None

            if mrg:
                ac = load_imported_data_for_antecedent(fname=GOLD_PARSE_FULL_NPY_DATA)

            results_lst = cross_validate(type_=type_, auto_parse=not mrg, classifier=ac)

            log_results(results_lst, fname=results_save)

    if 'ablate' in argv:
        ablation_study(auto_parse=not mrg, exclude=True)

    if 'bos' in argv:
        bos_compare()

    if 'bos_spen' in argv:
        bos_spen_split()

    if 'debug' in argv:
        ac = load_imported_data_for_antecedent(fname=save_file)
        ac.train_triggers = ac.train_triggers[0:2]
        ac.val_triggers = ac.val_triggers[0:2]
        ac.test_triggers = ac.test_triggers[0:2]
        ac.generate_possible_ants(['VP', wc.is_predicative, wc.is_adjective, wc.is_verb])
        ac.build_feature_vectors(debug=True)

