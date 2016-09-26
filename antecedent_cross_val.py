# coding=utf-8
__author__ = 'kian'

import numpy as np
import word_characteristics as wc
from detect_antecedents import AntecedentClassifier
from sklearn.cross_validation import KFold
from sys import platform, argv
from random import shuffle
from load_data import find_section
from sklearn.linear_model import LogisticRegressionCV

print argv

AUTO_PARSE_NPY_DATA      = 'antecedent_auto_parse_data_FULL_DATASET.npy'
GOLD_PARSE_FULL_NPY_DATA = 'antecedent_GOLD_parse_data_FULL_DATASET.npy'
AUTO_PARSE_ALL_ANTS_NPY  = 'antecedent_auto_parse_ALL_ANTS_with_liu.npy'
END_TO_END = 'END_TO_END_PREDICTIONS_FINAL_ABSOLUTE.npy'

if platform == 'linux2':
    AUTO_PARSE_NPY_DATA      = '../npy_data/' + AUTO_PARSE_NPY_DATA
    GOLD_PARSE_FULL_NPY_DATA = '../npy_data/' + GOLD_PARSE_FULL_NPY_DATA
    AUTO_PARSE_ALL_ANTS_NPY  = '../npy_data/' + AUTO_PARSE_ALL_ANTS_NPY

# antecedent classifier hyper parameters
K = 5
C = 2.65
LR = 0.0075
dropout = 0.0
EPOCHS = 2
seed = 1611944

print 'We used parameters:'
print 'k=%d, c=%0.2f, epochs=%d, lr=%0.07f, seed=%d' % (K, C, EPOCHS, LR, seed)

for arg in argv:
    if arg.startswith('seed='):
        seed = int(arg.split('seed=')[1])
    if arg.startswith('c='):
        C = float(arg.split('c=')[1])
    if arg.startswith('lr='):
        LR = float(arg.split('lr=')[1])
    if arg.startswith('drop='):
        dropout = float(arg.split('drop=')[1])
    if arg.startswith('epochs='):
        EPOCHS = int(arg.split('epochs=')[1])


def init_classifier(auto_parse=True):
    ac = load_classifier(auto_parse=auto_parse, fname=AUTO_PARSE_ALL_ANTS_NPY if auto_parse else GOLD_PARSE_FULL_NPY_DATA)
    ac.initialize_weights(seed=seed)
    ac.C = C
    ac.learn_rate = lambda x: LR
    return ac

def add_end_to_end(classifier):
    print 'Adding end to end results...'
    gold_and_predicted = np.load(END_TO_END)[0]
    false_positives = 0
    i = 0
    for gold,predicted in gold_and_predicted:
        if gold == 1:
            if predicted == 1:
                classifier.test_triggers[i].was_automatically_detected = True
            else:
                classifier.test_triggers[i].was_automatically_detected = False
            i += 1
        elif gold==0 and predicted==1:
            false_positives += 1
    classifier.false_positives = false_positives
    print '%d false positives. %0.2f percent of the data are FPs.'\
          %(classifier.false_positives, false_positives/(float(false_positives + i)))

def cross_validate(k_fold=5, type_=None, auto_parse=False, classifier=None, baseline=False, get_res_str=False):
    prediction_cv_list = []

    if classifier is not None:
        ac = classifier
    else:
        ac = load_classifier(auto_parse=auto_parse, fname=AUTO_PARSE_ALL_ANTS_NPY)
        if 'some_liu' in argv:
            for ant in ac.iterants():
                ant.x = ant.x[range(404) + range(len(ant.x)-15, len(ant.x))]
        ac.initialize_weights(seed=seed)

    if 'debug2' in argv:
        ac.train_triggers = ac.train_triggers[:5]
        ac.val_triggers = ac.val_triggers[:5]
        ac.test_triggers = ac.test_triggers[:5]

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

        if not baseline:
            val_acc, test_acc, val_preds, test_preds = ac.fit(epochs=EPOCHS, k=K, dropout=dropout)
            accs.append((val_acc, test_acc))
            prediction_cv_list.append(test_preds)
        else:
            bval_acc, btest_acc, bval_pred, btest_pred = ac.baseline_prediction()
            baseline_accs.append((bval_acc, btest_acc))
            prediction_cv_list.append(btest_pred)

        ac.reset()
        ac.initialize_weights()

    results = []
    lst = baseline_accs if baseline else accs
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
    if get_res_str:
        return results
    else:
        return ac, prediction_cv_list

def results_by_type(ac, prediction_cv):
    results_dict = {key:[] for key in ['do','be','to','modal','have','so']}

    for index in xrange(len(prediction_cv)):
        preds = prediction_cv[index]
        type_dict = {key:[] for key in ['do','be','to','modal','have','so']}

        for i,ant in enumerate(preds):
            type_dict[ant.trigger.type].append(preds[i])

        for type_ in type_dict:
            results_dict[type_].append(1.0-ac.accuracy(type_dict[type_]))

    for type_ in results_dict:
        avg_f1 = np.mean(results_dict[type_])
        print type_,'gets this antecedent identification accuracy:',avg_f1,'\n'

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

            for sec in xrange(0, 25):
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

    val_acc, test_acc = ac.fit(epochs=EPOCHS, k=K, dropout=dropout)
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

    if 'only_liu' in argv:
        for ant in ac.iterants():
            ant.x = ant.x[404:]
        ac.initialize_weights(seed=seed)

    if 'no_liu' in argv:
        for ant in ac.iterants():
            ant.x = ant.x[:404]
        ac.initialize_weights(seed=seed)

    if 'some_liu' in argv:
        for ant in ac.iterants():
            ant.x = ant.x[range(404) + range(len(ant.x)-15, len(ant.x))]
        ac.initialize_weights(seed=seed)

    if 'remove' in argv:
        print 'Removing to-VPE...'
        remove = [i for i in range(len(ac.train_triggers)) if ac.train_triggers[i].type=='to']
        for idx in remove:
            ac.train_triggers[idx] = None
        ac.train_triggers = [t for t in ac.train_triggers if t is not None]

        remove = [i for i in range(len(ac.val_triggers)) if ac.val_triggers[i].type=='to']
        for idx in remove:
            ac.val_triggers[idx] = None
        ac.val_triggers = [t for t in ac.val_triggers if t is not None]       

        remove = [i for i in range(len(ac.test_triggers)) if ac.test_triggers[i].type=='to']
        for idx in remove:
            ac.test_triggers[idx] = None
        ac.test_triggers = [t for t in ac.test_triggers if t is not None]       

    if 'end_to_end' in argv:
        print 'Performing end-to-end evaluation...'
        ac.use_auto_triggers = True
        add_end_to_end(ac)

    ac.debug_ant_selection()

    val_acc, test_acc, val_preds, test_preds = ac.fit(epochs=EPOCHS, k=K, dropout=dropout)
    bval_acc, btest_acc = None,None #ac.baseline_prediction()

    print 'MIRA RESULTS:'
    print val_acc, test_acc
    print '\nBASELINE:'
    print bval_acc, btest_acc

    prediction_results(test_preds, [trig.gold_ant for trig in ac.test_triggers])

    # results with end to end
    if 'end_to_end' in argv:
        print 'RESULTS WITH END TO END:'
        precision_results_by_trig = [0.]*ac.false_positives # add 0 for each false positives
        recall_results_by_trig = []
        for i,trig in enumerate(ac.test_triggers):
            if not trig.was_automatically_detected:
                recall_results_by_trig.append(0.)
            else:
                tp_score = 1.0-ac.loss_function(trig.gold_ant, test_preds[i])
                precision_results_by_trig.append(tp_score)
                recall_results_by_trig.append(tp_score)

        precision, recall = np.mean(precision_results_by_trig), np.mean(recall_results_by_trig)
        f1 = (2*precision*recall)/(precision+recall)
        print 'Precision: %0.4f, Recall: %0.4f, F1: %0.4f'%(precision,recall,f1)

    return

def prediction_results(proposed, actual):
    # calculating liu et al. antecedent head resolution:
    assert len(proposed) == len(actual)
    tp, fp, fn = 0, 0, 0
    for i in range(len(proposed)):
        proposed_head = proposed[i].get_head()

        if proposed_head in actual[i].sub_sentdict.words:
            tp += 1
        else:
            fp += 1

    print '\nLiu head resolution', float(tp)/(tp+fp)

def ablation_study(auto_parse=False, exclude=True):
    # This is the division of features by their class:
    # first excludes the alignment features,
    # next exclude relational features
    # next exclude ant_trig relation features
    # last exclude hardt/nielsen feats
    feat_dict = {(1,419):'NO FEATURES EXCLUDED',
                 (139,419):'alignment',
                 (1,143,201,419):'ant_trig',
                 (1,201,404,419):'hardt',
                 (1,404):'liu'}

    for tup in feat_dict.iterkeys():
        ac = load_classifier(auto_parse=auto_parse, fname=AUTO_PARSE_ALL_ANTS_NPY)

        for ant in ac.iterants():
            ant.x = ant.x[range(404) + range(len(ant.x)-15, len(ant.x))]

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

        ac.initialize_weights(seed=seed)

        results = ['----\nFeature: %s\n' % feat_dict[tup]] + ['EXCLUDED' if exclude else 'INLCUDED', '\n'] \
                  + cross_validate(auto_parse=auto_parse, classifier=ac, get_res_str=True)
        log_results(results, fname='LAST_feature_ablation_ant_LIU_features_c%s_lr%s_LAST.txt'%(str(C), str(LR)))

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

    print 'Saving data...'
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
    results_save = 'final_results_ALL_TYPES_ALL_ANTS_WITH_LIU_c%s_lr%s.txt' %(str(C), str(LR))

    if 'gen2' in argv:
        AUTO_PARSE_ALL_ANTS_NPY = AUTO_PARSE_ALL_ANTS_NPY[:-4]+'_antgen2.npy'
        results_save = results_save[:-4]+'_antgen2.txt'

    if 'build' in argv:
        ac = AntecedentClassifier(0,14,15,19,20,24)
        ac.import_data(get_mrg=mrg)
        # save_imported_data_for_antecedent(ac, fname=save_file)
        # exit(0)

        ac = load_imported_data_for_antecedent(fname=save_file)
        ac.generate_possible_ants(['VP', wc.is_predicative, wc.is_adjective, wc.is_verb], filter=True)
        ac.debug_ant_selection()
        # save_imported_data_for_antecedent(ac, fname=save_file)
        # exit(0)

        ac.build_feature_vectors()
        ac.normalize()
        save_imported_data_for_antecedent(ac, fname=save_file)

    if 'types' in argv:
        ac = None
        if mrg:
            ac = load_imported_data_for_antecedent(fname=GOLD_PARSE_FULL_NPY_DATA)

        ac, prediction_list = cross_validate(auto_parse=not mrg, classifier=ac, baseline='baseline' in argv)
        results_by_type(ac, prediction_list)
        # log_results(results_lst, fname=results_save)

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



