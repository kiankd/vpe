import matplotlib
matplotlib.use('Agg')

import vpe_objects as vpe
import numpy as np
import word_characteristics as wc
import antecedent_vector_creation as avc
import optimize_mira_dual as mira
import truth
import time
import sys
from heapq import nlargest, nsmallest
from os import listdir
from pyprind import ProgBar
from random import randrange,sample
from matplotlib import pyplot as plt
from alignment import alignment_matrix
from copy import copy
from sklearn.preprocessing import StandardScaler
from subprocess import call
call('clear')

AUTO_PARSE_XML_DIR = '/Users/kian/Documents/HONOR/xml_annotations/raw_auto_parse/'
AUTO_PARSE_NPY_DATA = '../npy_data/antecedent_auto_parse_data.npy'

def shuffle(list_):
    np.random.shuffle(list_)
    return list_


class AntecedentClassifier(object):
    SCHEDULE_FREQUENCY = 10

    def __init__(self, train_start, train_end, val_start, val_end, test_start, test_end,
                 learn_rate=lambda x: 1.0 / (x + 1), C=1.0):
        self.sentences = vpe.AllSentences()
        self.annotations = vpe.Annotations()
        self.file_names = vpe.Files()

        self.train_ants = []
        self.val_ants = []
        self.test_ants = []
        self.train_triggers = []
        self.val_triggers = []
        self.test_triggers = []

        self.start_train = train_start
        self.end_train = train_end
        self.start_val = val_start
        self.end_val = val_end
        self.start_test = test_start
        self.end_test = test_end

        self.W_avg = None
        self.W_old = None
        self.learn_rate = learn_rate
        self.C = C

        self.missed_vpe = 0
        self.num_features = 0

        self.norms = []
        self.feature_vals = []
        self.feature_names = []
        self.random_features = []
        self.train_err = []
        self.val_err = []
        self.test_err = []
        self.train_results, self.val_results, self.test_results = [], [], []
        self.diffs = []

        self.sentence_words = []

    def reset(self):
        self.W_avg = None
        self.W_old = None
        self.norms = []
        self.feature_vals = []
        self.feature_names = []
        self.random_features = []
        self.train_err = []
        self.val_err = []
        self.test_err = []
        self.diffs = []
        self.train_results, self.val_results, self.test_results = [], [], []

    def initialize(self, pos_tests, seed=1917, W=None, test=0, test_specific=(None, None), save=False, load=False,
                   update=False, verbose=False):
        if not load:
            self.import_data()
            self.save_imported_data()
            print 'SAVED IMPORTED DATA'
            self.generate_possible_ants(pos_tests, test=test)
            self.debug_ant_selection()
            self.build_feature_vectors()
            self.normalize()
        else:
            self.load_imported_data()
            # self.generate_possible_ants(pos_tests, only_filter=False)
            # self.debug_ant_selection()
            # self.debug_ant_selection(check_list=self.val_triggers)
            # if update:
            #     self.build_feature_vectors(test_specific=test_specific)
            #     self.normalize()

        if update:
            # self.generate_possible_ants(pos_tests, test=test, strong=False)
            self.generate_possible_ants(pos_tests, test=test, strong=True)
            # self.generate_possible_ants2()
            self.debug_ant_selection()
            self.debug_ant_selection(check_list=self.val_triggers)
            self.build_feature_vectors(test_specific=test_specific)
            self.normalize()

        if save:
            self.save_imported_data()

        self.initialize_weights(initial=W, seed=seed)

    def import_data(self, test=None, get_mrg=True):
        """Import data from our XML directory and find all of the antecedents. Takes a bit of time."""
        dirs = listdir(self.file_names.XML_MRG)
        dirs.sort()

        sentnum_modifier = -1
        dnum = 0
        for d in dirs:
            subdir = d + self.file_names.SLASH_CHAR
            if subdir.startswith('.'): continue
            file_list = listdir(self.file_names.XML_MRG + subdir)

            if (self.start_train <= dnum <= self.end_train) or \
                    (self.start_test <= dnum <= self.end_test) or \
                    (self.start_val <= dnum <= self.end_val):

                section_annotation = vpe.AnnotationSection(subdir, self.file_names.VPE_ANNOTATIONS)

                vpe_files = list(set([annotation.file for annotation in section_annotation]))
                vpe_files.sort()

                for f in vpe_files:
                    if not test or (test and f in test):
                        # Here we are now getting the non-MRG POS file that we had neglected to get before.
                        try:
                            # This condition makes it so that we use the same files for auto-parse dataset results.
                            if not f + '.mrg.xml' in file_list:
                                raise IOError

                            if get_mrg:
                                mrg_matrix = vpe.XMLMatrix(f + '.mrg.xml', self.file_names.XML_MRG + subdir, get_deps=True)
                            else:
                                mrg_matrix = vpe.XMLMatrix(f + '.xml', AUTO_PARSE_XML_DIR, get_deps=True)

                        except IOError:
                            print '(auto-parse file to complete MRG dataset)',
                            mrg_matrix = vpe.XMLMatrix(f + '.xml', AUTO_PARSE_XML_DIR, pos_file=True)
                            # NO DEPENDENCIES IN POS FILES!

                        """ Note that I am using the gold standard triggers here. """
                        file_annotations = section_annotation.get_anns_for_file(f)
                        section_triggers = mrg_matrix.get_gs_auxiliaries(file_annotations, sentnum_modifier)
                        # self.triggers.add_auxs(mrg_matrix.get_gs_auxiliaries(file_annotations, sentnum_modifier))
                        try:
                            if self.start_train <= dnum <= self.end_train:
                                self.train_ants += mrg_matrix.get_gs_antecedents(file_annotations, section_triggers,
                                                                                 sentnum_modifier)
                                self.train_triggers += section_triggers

                            if self.start_val <= dnum <= self.end_val:
                                self.val_ants += mrg_matrix.get_gs_antecedents(file_annotations, section_triggers,
                                                                               sentnum_modifier)
                                self.val_triggers += section_triggers

                            if self.start_test <= dnum <= self.end_test:
                                self.test_ants += mrg_matrix.get_gs_antecedents(file_annotations, section_triggers,
                                                                                sentnum_modifier)
                                self.test_triggers += section_triggers

                        except AssertionError:
                            self.missed_vpe += 1
                            print 'Warning: different number of triggers and antecedents!'

                        self.annotations.add_section(section_annotation)
                        self.sentences.add_mrg(mrg_matrix)

                        sentnum_modifier = len(self.sentences) - 1
            dnum += 1

        for trig in self.train_triggers + self.val_triggers + self.test_triggers:
            sent = self.sentences[trig.sentnum].words
            try:
                if sent[trig.wordnum + 1] == 'so' or sent[trig.wordnum + 1] == 'likewise':
                    trig.type = 'so'
                if sent[trig.wordnum + 1] == 'the' and sent[trig.wordnum + 2] in ['same', 'opposite']:
                    trig.type = 'so'
            except IndexError:
                pass

    def generate_possible_ants(self, pos_tests, test=0, delete_random=0.0, only_filter=False, strong=False,
                               test_specific=(None, None)):
        """Generate all candidate antecedents."""
        if not only_filter:
            print 'Generating possible antecedents...'
            bar = ProgBar(len(self.train_triggers) + len(self.val_triggers) + len(self.test_triggers))

            if test:  # ONLY FOR TESTING! THIS CHEATS!!
                for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
                    trigger.possible_ants = []
                    self.sentences.set_possible_ants(trigger, pos_tests)
                    trigger.possible_ants = [trigger.gold_ant] + trigger.possible_ants[0:test]
                return

            # Fair.
            for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
                if not (test_specific[0] and test_specific[1]) or (
                        trigger.sentnum == test_specific[0] and trigger.wordnum == test_specific[1]):
                    trigger.possible_ants = []
                    self.sentences.set_possible_ants(trigger, pos_tests)
                    trigger.possible_ants = list(trigger.possible_ants)

                bar.update()

        print 'Filtering antecedents out...'


        # Get POS tags that we can use to filter out the antecedents:
        all_pos = set()
        for s in self.sentences:
            for p in s.pos:
                all_pos.add(p)
        all_pos_combos = set([(a, b) for a in all_pos for b in all_pos])

        ant_start_pos_combo = set()
        ant_end_pos_combo = set()
        ant_start_pos = set()
        ant_end_pos = set()
        for trig in self.train_triggers:
            ant_start_pos.add(trig.gold_ant.sub_sentdict.pos[0])
            ant_end_pos.add(trig.gold_ant.sub_sentdict.pos[-1])
            if len(trig.gold_ant.sub_sentdict) > 1:
                ant_start_pos_combo.add((trig.gold_ant.sub_sentdict.pos[0], trig.gold_ant.sub_sentdict.pos[1]))
                ant_end_pos_combo.add((trig.gold_ant.sub_sentdict.pos[-1], trig.gold_ant.sub_sentdict.pos[-2]))

        filter_start_combo = all_pos_combos - ant_start_pos_combo
        filter_end_combo = all_pos_combos - ant_end_pos_combo
        filter_start = all_pos - ant_start_pos
        filter_end = all_pos - ant_end_pos

        # Delete antecedents that contain the trigger in them:
        c = 0
        duplicates = 0
        all_ants = set([])
        for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
            deletes = []
            for ant in trigger.possible_ants:
                size = len(all_ants)
                all_ants.add((ant.sentnum, ant.start, ant.end))

                if False:
                    pass

                if ant.sub_sentdict.pos[0] in filter_start:
                    c += 1
                    deletes.append(ant)

                elif ant.sub_sentdict.pos[-1] in filter_end:
                    c += 1
                    deletes.append(ant)

                elif strong and len(ant.sub_sentdict) > 1 and (
                ant.sub_sentdict.pos[0], ant.sub_sentdict.pos[1]) in filter_start_combo:
                    c += 1
                    deletes.append(ant)

                elif strong and len(ant.sub_sentdict) > 1 and (
                ant.sub_sentdict.pos[-1], ant.sub_sentdict.pos[-2]) in filter_end_combo:
                    c += 1
                    deletes.append(ant)

                elif len(all_ants) == size:
                    duplicates += 1
                    deletes.append(ant)

                elif ant.contains_trigger():
                    c += 1
                    deletes.append(ant)

                elif len(ant.sub_sentdict) == 0:
                    c += 1
                    deletes.append(ant)

            for ant in deletes:
                trigger.possible_ants.remove(ant)

        print 'Deleted %d bad antecedents!' % c

    def generate_possible_ants2(self):
        tag_length_dict = {}
        for trig in self.train_triggers:
            tag = trig.gold_ant.sub_sentdict.pos[0]
            if not tag in tag_length_dict:
                tag_length_dict[tag] = len(trig.gold_ant.sub_sentdict)
            else:
                tag_length_dict[tag] = max(tag_length_dict[tag], len(trig.gold_ant.sub_sentdict))

        for trig in self.train_triggers + self.val_triggers + self.test_triggers:
            self.sentences.set_possible_ants2(trig, tag_length_dict)

        self.generate_possible_ants([], only_filter=True, strong=True)

    def bestk_ants(self, trigger, w, k=5):
        """Return the best k antecedents given the weight vector w with respect to the score attained."""
        for a in range(len(trigger.possible_ants)):
            trigger.possible_ants[a].set_score(w)

        return nlargest(k, trigger.possible_ants, key=vpe.Antecedent.get_score)

    def build_feature_vectors(self, verbose=True, test_specific=(None, None)):
        vec_length = None
        word2vec_dict = truth.loadword2vecs()
        all_pos_tags = truth.extract_data_from_file(
            truth.EACH_UNIQUE_POS_FILE)  # We only want to import this file once.

        print 'Building feature vectors...'

        dep_names = ('prep','nsubj','dobj','nmod','adv','conj','vmod','amod','csubj')
        print len(dep_names),dep_names

        bar = ProgBar(len(self.train_triggers) + len(self.val_triggers) + len(self.test_triggers))
        for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
            if not (test_specific[0] and test_specific[1]) \
                    or (test_specific[0] == trigger.sentnum and test_specific[1] == trigger.wordnum):

                alignment_matrix(self.sentences, trigger, word2vec_dict, dep_names=dep_names, pos_tags=all_pos_tags)

                if trigger == self.train_triggers[0]:
                    vec_length = len(trigger.gold_ant.x)
                    print 'Feature vector length: %d' % vec_length

                if vec_length:
                    assert len(trigger.gold_ant.x) == vec_length
                    for ant in trigger.possible_ants:
                        assert len(ant.x) == vec_length

                bar.update()

        return

    def itertrigs(self):
        for trig in self.train_triggers + self.val_triggers + self.test_triggers:
            yield trig

    def initialize_weights(self, initial=None, seed=1917):
        np.random.seed(seed)
        # self.W_old = np.ones(len(self.train_triggers[0].possible_ants[0].x))
        if initial is None:
            self.W_old = np.random.rand(len(self.train_triggers[0].possible_ants[0].x))
            print 'Weight vector length:',len(self.W_old)
        else:
            self.W_old = initial
        self.W_avg = copy(self.W_old)

    def debug_ant_selection(self, check_list=None, verbose=False, write_to_file=None):
        missed = 0
        total = 0
        lengths = []
        head_matches = 0

        if write_to_file:
            f = open(write_to_file, 'w')

        if not check_list:
            check_list = self.train_triggers

        for trigger in check_list:
            lengths.append(len(trigger.possible_ants))
            got_head = False
            try:
                for poss_ant in trigger.possible_ants:
                    if not got_head:
                        if poss_ant.get_head() == trigger.gold_ant.get_head():
                            got_head = True
                            head_matches += 1

                    if poss_ant.sub_sentdict == trigger.gold_ant.sub_sentdict:
                        raise vpe.Finished()
            except vpe.Finished:
                total += 1
                continue

            total += 1
            missed += 1

            if verbose:
                print '\nMISSED ANT FOR TRIGGER', trigger
                print 'ANT: ', trigger.gold_ant.sub_sentdict.pos, trigger.gold_ant.sub_sentdict.words
                print self.sentences.get_sentence(trigger.sentnum)
                print self.sentences.get_sentence_tree(trigger.sentnum)
            if write_to_file:
                f.write('Trigger: %s\n' % trigger.__repr__())
                f.write('Antecedent: %s\n' % trigger.gold_ant.__repr__())
                f.write('Tree:\n%s' % self.sentences.get_sentence_tree(trigger.sentnum).__repr__())

        s = ['Total ants: %d' % sum(lengths),
             '\nMissed this many gold possible ants: %d' % missed,
             'That is %0.2f percent.' % (float(missed) / total),
             'Got this many head matches as a percent of total: %0.2f' % (head_matches / float(len(check_list))),
             'Average length of possible ants: %d' % np.mean(lengths),
             'Min/Max lengths: %d, %d\n' % (min(lengths), max(lengths))]

        for x in s:
            print x
        print '================================================'

        if write_to_file:
            for x in s:
                f.write(x + '\n')
            f.close()

    def normalize(self):
        print 'Normalizing the data...'
        xtrain, xval, xtest = [], [], []

        for trig in self.train_triggers:
            for ant in trig.possible_ants:
                xtrain.append(ant.x)
            xtrain.append(trig.gold_ant.x)

        for trig in self.val_triggers:
            for ant in trig.possible_ants:
                xval.append(ant.x)
            xval.append(trig.gold_ant.x)

        for trig in self.test_triggers:
            for ant in trig.possible_ants:
                xtest.append(ant.x)
            xtest.append(trig.gold_ant.x)

        s = StandardScaler()
        xtrain = s.fit_transform(xtrain)
        xval = s.transform(xval)
        xtest = s.transform(xtest)

        i = 0
        for trig in self.train_triggers:
            for ant in trig.possible_ants:
                ant.x = xtrain[i]
                i += 1
            trig.gold_ant.x = xtrain[i].flatten()
            i += 1
        i = 0
        for trig in self.val_triggers:
            for ant in trig.possible_ants:
                ant.x = xval[i]
                i += 1
            trig.gold_ant.x = xval[i].flatten()
            i += 1
        i = 0
        for trig in self.test_triggers:
            for ant in trig.possible_ants:
                ant.x = xtest[i].flatten()
                i += 1
            trig.gold_ant.x = xtest[i].flatten()
            i += 1
        del i

    def debug_alignment(self, verbose=False):
        X = {}
        for trigger in self.train_triggers:
            for ant in trigger.possible_ants:
                vec_tup = tuple(ant.x)
                if not X.has_key(vec_tup):
                    X[vec_tup] = [ant]
                else:
                    X[vec_tup].append(ant)

        length_list = []
        printed = False
        for key in X:
            length_list.append(len(X[key]))

            if verbose and (not printed and len(X[key]) >= 79):
                for ant in X[key]:
                    print ant
                print "\nTotal ants for this key: %d" % len(X[key])
                print key
                printed = True

        print 'TOTAL : %d, UNIQUE : %d' % (sum(length_list), len(X))
        print 'Average number of ants corresponding to same vec: %0.2f, min= %d, max= %d' % (np.mean(length_list),
                                                                                             min(length_list),
                                                                                             max(length_list))

    def fit(self, epochs=5, verbose=True, k=5, features_to_analyze=5, c_schedule=1.0):
        # Here we are just adding the gold standard to the training set.
        for trig in self.train_triggers:
            has = False
            for ant in trig.possible_ants:
                if ant == trig.gold_ant:
                    has = True
                    break
            if not has:
                trig.possible_ants.append(trig.gold_ant)

        best_score = 0.0
        best_weights = []
        i=0
        for n in range(epochs):
            for trigger in shuffle(self.train_triggers):
                bestk = self.bestk_ants(trigger, self.W_old, k=k)

                self.W_old = mira.update_weights(self.W_old, bestk, trigger.gold_ant, self.loss_function, C=self.C)
                self.W_avg = ((1.0 - self.learn_rate(i)) * self.W_avg) + (self.learn_rate(i) * self.W_old)  # Running average.
                self.analyze(self.W_avg, features_to_analyze)
                i += 1

                # if n % self.SCHEDULE_FREQUENCY == 0:
                #     self.C /= c_schedule

                # if verbose and n>0:
                train_preds = self.predict(self.train_triggers)
                val_preds = self.predict(self.val_triggers)
                test_preds = self.predict(self.test_triggers)

                self.train_err.append(self.accuracy(train_preds))
                self.val_err.append(self.accuracy(val_preds))
                self.test_err.append(self.accuracy(test_preds))

                self.train_results = list(self.train_results)
                self.val_results = list(self.val_results)
                self.test_results = list(self.test_results)

                self.train_results.append(self.criteria_based_results(train_preds))
                self.val_results.append(self.criteria_based_results(val_preds))
                self.test_results.append(self.criteria_based_results(test_preds))

                self.diffs.append(np.mean((self.W_avg - self.W_old) ** 2))

                print '\nEpoch %d Train/val/test error: %0.2f, %0.2f, %0.2f' \
                      % (n, self.train_err[-1], self.val_err[-1], self.test_err[-1])

                print '\tTrain/val/test ExacM: %0.2f, %0.2f, %0.2f' \
                      % (self.train_results[-1][0], self.val_results[-1][0], self.test_results[-1][0])
                print '\tTrain/val/test HeadM: %0.2f, %0.2f, %0.2f' \
                      % (self.train_results[-1][1], self.val_results[-1][1], self.test_results[-1][1])
                print '\tTrain/val/test HeadO: %0.2f, %0.2f, %0.2f' \
                      % (self.train_results[-1][2], self.val_results[-1][2], self.test_results[-1][2])

                if self.val_results[-1][1] > best_score:
                    best_score = self.val_results[-1][1]
                    best_weights = [copy(self.W_avg)]

                elif self.val_results[-1][1] == best_score:
                    best_weights.append(copy(self.W_avg))

                # print 'Trigger sentnum,wordnum: %d,%d'%(trigger.sentnum,trigger.wordnum)
                # print self.sentences.get_sentence(trigger.sentnum)

                # best_ant = self.bestk_ants(trigger, self.W_avg, k=1)[0]
                # print 'Best_ant sentnum = %d, start,end = %d,%d:'%(best_ant.sentnum,best_ant.start,best_ant.end),
                # print 'Best ant: start %d, end %d, trigger wordnum %d: '%(best_ant.start, best_ant.end, best_ant.trigger.wordnum)
                # print best_ant
                # print 'Difference btwen avg vector w_old: %0.6f'%(self.diffs[-1])

        self.train_results = np.array(self.train_results)
        self.val_results = np.array(self.val_results)
        self.test_results = np.array(self.test_results)

        best_weight_vector = np.mean(best_weights, axis=0)
        self.W_avg = best_weight_vector

        val_preds = self.predict(self.val_triggers)
        test_preds = self.predict(self.test_triggers)

        print 'Validation score using best weight vector: %0.3f'%self.criteria_based_results(val_preds)[1]
        print 'Test score using best validation weight vector: %0.3f'%self.criteria_based_results(test_preds)[1]
        
        return self.criteria_based_results(val_preds)[1], self.criteria_based_results(test_preds)[1]

    def predict(self, trigger_list):
        predictions = []
        for trigger in trigger_list:
            predictions.append(self.bestk_ants(trigger, self.W_avg, k=1)[0])
        return predictions

    def loss_function(self, gold_ant, proposed_ant):
        """
        @type gold_ant: vpe.Antecedent
        @type proposed_ant: vpe.Antecedent
        """
        # I think this should be slightly modified:
        # weigh the "head" - the first word of the gold_ant
        # more than the rest of the words.

        if not gold_ant.get_head() in proposed_ant.get_words():
            return 1.0

        gold_vals = gold_ant.get_words()
        proposed_vals = proposed_ant.get_words()

        tp = float(len([val for val in proposed_vals if val in gold_vals]))
        fp = float(len([val for val in proposed_vals if not val in gold_vals]))
        fn = float(len([val for val in gold_vals if not val in proposed_vals]))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision == 0.0 or recall == 0.0:
            return 1.0

        f1 = (2.0 * precision * recall) / (precision + recall)

        # dotprod = proposed_ant.score
        # return 0.0 if dotprod >= 0.5 else 0.5-dotprod
        # norm_diff = np.linalg.norm(proposed_ant.x - gold_ant.x)
        # return 0.0 if norm_diff <= 0.25 else norm_diff-0.25

        return 1.0 - f1

    def accuracy(self, predictions):
        errors = []
        for ant in predictions:
            errors.append(self.loss_function(ant.trigger.gold_ant, ant))
        return float(np.mean(errors, dtype=np.float64))

    def criteria_based_results(self, predictions):
        exact_match = 0
        head_match = 0
        head_overlap = 0

        for ant in predictions:
            gold = ant.trigger.gold_ant

            if ant.sentnum != gold.sentnum:
                continue

            antwords = ant.sub_sentdict.words
            goldwords = gold.sub_sentdict.words

            if len(antwords) > 0:
                em = antwords == goldwords
                hm = ant.get_head() == gold.get_head()
                ho = gold.get_head() in antwords

            if em:
                exact_match += 1
            if hm:
                head_match += 1
            if ho:
                head_overlap += 1

        # print exact_match, len(predictions)
        # print head_match, len(predictions)
        # print head_overlap, len(predictions)

        exact_match /= float(len(predictions))
        head_match /= float(len(predictions))
        head_overlap /= float(len(predictions))

        return [exact_match, head_match, head_overlap]

    def analyze(self, w, num_features):
        self.norms.append(np.linalg.norm(w))

        if not self.feature_vals:
            self.feature_vals = [[] for _ in range(num_features)]
            self.feature_names = ['Bias weight']
            for i in range(num_features - 2):
                self.random_features.append(randrange(1, len(w)))
                self.feature_names.append('Rand_feature %d' % self.random_features[i])
            self.feature_names.append('NP dot product/norm')

        self.feature_vals[0].append(w[0])
        for i in range(num_features - 2):
            self.feature_vals[i + 1].append(w[self.random_features[i]])
        self.feature_vals[-1].append(w[-1])

    def make_graphs(self, name):
        # params = 'tests/post_changes/%s'%name
        params = 'best_results_analysis/%s' % name

        plt.figure(1)
        plt.title('Running Average Weight Vector 2-Norm over time')
        plt.plot(range(len(self.norms)), self.norms, 'b-')
        plt.savefig(params + 'norm_change1.png', bbox_inches='tight')
        plt.clf()

        colors = ['b', 'y', 'm', 'r', 'g']
        plt.figure(2)
        ax = plt.subplot(111)
        plt.title('Exact Feature Values over time')
        for i in range(len(self.feature_vals)):
            ax.plot(range(len(self.feature_vals[i])), self.feature_vals[i], colors[i % len(colors)] + '-',
                    label=self.feature_names[i])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(params + 'feature_value_change1.png', bbox_inches='tight')
        plt.clf()

        plt.figure(3)
        ax = plt.subplot(111)
        plt.title('Train/Val/Test Error over time')
        ax.plot(range(len(self.train_err)), self.train_err, 'b-', label='Train')
        ax.plot(range(len(self.val_err)), self.val_err, 'y-', label='Validation')
        ax.plot(range(len(self.test_err)), self.test_err, 'r-', label='Test')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(params + 'errors.png', bbox_inches='tight')
        plt.clf()

        plt.figure(4)
        plt.title('Squared Difference between average and MIRA weight vector')
        plt.plot(range(len(self.diffs)), self.diffs, 'bo')
        plt.savefig(params + 'weight_vector_change1.png', bbox_inches='tight')
        plt.clf()

        plt.figure(5)
        ax = plt.subplot(111)
        # plt.title('Head match results over time')
        ax.plot(range(len(self.train_results)), self.train_results[:,1], 'b-', label='Train')
        ax.plot(range(len(self.val_results)), self.val_results[:,1], 'y-', label='Validation')
        ax.plot(range(len(self.test_results)), self.test_results[:,1], 'r-', label='Test')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='best')#, bbox_to_anchor=(1, 0.5))
        plt.savefig(params+'results.png', bbox_inches='tight')
        plt.clf()

    def log_results(self, name):
        destination = 'ant_results/' + name

        errors = []
        for lst in self.train_err, self.val_err, self.test_err:
            lowest_error = 100
            epoch = 0
            for i in range(len(lst)):
                if lst[i] < lowest_error:
                    lowest_error = lst[i]
                    epoch = i
            errors.append((lowest_error, epoch))

        results = []
        for lst in self.train_results, self.val_results, self.test_results:
            best_em, best_hm, best_ho = 0, 0, 0
            em_epoch, hm_epoch, ho_epoch = 0, 0, 0
            for i in range(len(lst)):
                if lst[i][0] > best_em:
                    best_em = lst[i][0]
                    em_epoch = i

                if lst[i][1] > best_hm:
                    best_hm = lst[i][1]
                    hm_epoch = i

                if lst[i][2] > best_ho:
                    best_ho = lst[i][2]
                    ho_epoch = i

            results.append((best_em, em_epoch, best_hm, hm_epoch, best_ho, ho_epoch))

        write_string = ''
        names = ['Train', 'Validation', 'Test']
        res_names = ['exact match', 'head match', 'head overlap']
        for i in range(len(names)):
            write_string += '%s results: lowest error = %0.2f at epoch %d\n' % (names[i], errors[i][0], errors[i][1])
            for j in range(len(res_names)):
                write_string += '\tBest %s = %0.2f at epoch %d\n' % (
                res_names[j], results[i][j * 2], results[i][(j * 2) + 1])

        with open(destination + '.txt', 'w') as f:
            f.write(write_string)

        np.save(destination,
                np.array([self.train_err, self.val_err, self.test_err,
                          [self.train_results, self.val_results, self.test_results]]))

    def save_imported_data(self, name='', auto_parse_data=False):
        print 'Saving the data...'
        if not auto_parse_data:
            np.save(self.file_names.IMPORTED_DATA + name,
                    np.array([self.sentences, self.train_triggers, self.val_triggers, self.test_triggers]))
        else:
            np.save(AUTO_PARSE_NPY_DATA,
                    np.array([self.sentences, self.train_triggers, self.val_triggers, self.test_triggers]))

    def load_imported_data(self, auto_parse_data=False, fname=''):
        print 'Loading the data...'
        if 'orig' in sys.argv:
            data = np.load('/home/2014/kkenyo1/vpe_project/npy_data/BEST_RESULTS_SO_FAR_imported_data.npy')
        elif auto_parse_data:
            data = np.load(AUTO_PARSE_NPY_DATA)
        else:
            if fname:
                data = np.load(fname)
            else:
                data = np.load(self.file_names.IMPORTED_DATA)
        self.sentences = data[0]
        self.train_triggers = data[1]
        self.val_triggers = data[2]
        self.test_triggers = data[3]

        for trig in self.train_triggers + self.val_triggers + self.test_triggers:
            sent = self.sentences[trig.sentnum].words
            try:
                if sent[trig.wordnum + 1] == 'so' or sent[trig.wordnum + 1] == 'likewise':
                    trig.type = 'so'
                if sent[trig.wordnum + 1] == 'the' and sent[trig.wordnum + 2] in ['same', 'opposite']:
                    trig.type = 'so'
            except IndexError:
                pass

    def set_trigger_type(self, type_, alter_train=False, alter_test=True):
        """
        We use this if we want to see how the algorithm performs when it only trains and tests
        on one type of trigger, i.e. "do".
        """
        assert type_ in ['modal', 'be', 'have', 'do', 'to', 'so']
        if alter_train:
            self.train_triggers = [trig for trig in self.train_triggers if trig.type == type_]
        if alter_test:
            old_len = len(self.val_triggers)
            self.val_triggers = [trig for trig in self.val_triggers if trig.type == type_]
            print 'Frequency of %s in validation: %d out of a total of %d.'%(type_, len(self.val_triggers), old_len)

            old_len = len(self.test_triggers)
            self.test_triggers = [trig for trig in self.test_triggers if trig.type == type_]
            print 'Frequency of %s in test: %d out of a total of %d.'%(type_, len(self.test_triggers), old_len)

        print len(self.train_triggers), len(self.val_triggers), len(self.test_triggers)

    def baseline_prediction(self, verbose=False):
        """Baseline algorithm that only considers nearest VP."""
        train_ant_pred = [self.sentences.nearest_vp(trig) for trig in self.train_triggers]
        val_ant_pred = [self.sentences.nearest_vp(trig) for trig in self.val_triggers]
        test_ant_pred = [self.sentences.nearest_vp(trig) for trig in self.test_triggers]
        if verbose:
            for trig in self.train_triggers:
                print '---------------'
                print self.sentences[trig.sentnum]
                print 'ANT: ',
                print self.sentences.nearest_vp(trig)
                print 'GOLD ANT: ',
                print trig.gold_ant

        print 'Baseline results:'
        print '\tTrain: ', self.criteria_based_results(train_ant_pred)
        print '\tVal: ', self.criteria_based_results(val_ant_pred)
        print '\tTest: ', self.criteria_based_results(test_ant_pred)

        # return head match % for validation and test results
        return self.criteria_based_results(val_ant_pred)[1], self.criteria_based_results(test_ant_pred)[2]

    def gold_analysis(self):
        print 'Triggers:',
        freq = {}
        for trig in self.train_triggers + self.test_triggers + self.val_triggers:
            if not freq.has_key(trig.type):
                freq[trig.type] = 1
            else:
                freq[trig.type] += 1
        print freq
        print 'Total:', sum(freq.values())

        d = {'starts_with_aux': [], 'ends_with_aux': []}
        all_pos = set()
        for s in self.sentences:
            for p in s.pos:
                all_pos.add(p)

        ant_start_pos = set()
        for trig in self.train_triggers:
            if wc.is_aux_lemma(trig.gold_ant.sub_sentdict.words[0]):
                d['starts_with_aux'].append(trig.gold_ant)
            ant_start_pos.add(trig.gold_ant.sub_sentdict.pos[0])

        ant_end_pos = set()
        for trig in self.train_triggers:
            if wc.is_aux_lemma(trig.gold_ant.sub_sentdict.words[-1]):
                d['ends_with_aux'].append(trig.gold_ant)
            ant_start_pos.add(trig.gold_ant.sub_sentdict.pos[-1])

        print 'Ants never start with these tags: ', all_pos - ant_start_pos
        print 'Percent of ants that start with auxs: ', len(d['starts_with_aux']) / float(len(self.train_triggers))

        print 'Ants never END with these tags: ', all_pos - ant_end_pos
        print 'Percent of ants that END with auxs: ', len(d['ends_with_aux']) / float(len(self.train_triggers))

if __name__ == '__main__':
    pos_tests = ['VP', wc.is_adjective, wc.is_verb]

    try:
        debug = sys.argv[1] == 'debug'
    except IndexError:
        debug = False

    # if debug:
    #     a = AntecedentClassifier(0,14, 15,19, 20,24, C=0.075, learn_rate=lambda x: 0.0001)
    #     a.initialize(pos_tests, save=False, load=True, update=False, seed=2384834)
    #     a.set_trigger_type("do")
    #     a.baseline_prediction()
    #     exit(0)

    start_time = time.clock()

    if not debug:
        pass
    #     a = AntecedentClassifier(0,0, None,None, None,None, C=0.075, learn_rate=lambda x: 0.0001)
    #     print 'We missed %d vpe instances.'%a.missed_vpe
    #     initial_weights = None
    #
    #     a.initialize(pos_tests, W=initial_weights, test=0, delete_random=0,
    #              save=False, load=False, update=False, seed=2384834)
    #
    else:
        a = AntecedentClassifier(0,14, 15,19, 20,24)
        print 'Debugging...'
        if 'align' in sys.argv:
            a.initialize(pos_tests, seed=123, save=True, load=True, update=True, test_specific=(14036,14))
        else:
            a.initialize(['VP', wc.is_predicative, wc.is_adjective, wc.is_verb], seed=123, save=True, load=False, update=False)
        # a.initialize(pos_tests, save=True, load=True, update=False, seed=2334)
        # a.debug_ant_selection(verbose=False)
        # for trig in a.train_triggers:
        #     ant = trig.gold_ant
        #     print '---------------------'
        #     print a.sentences.get_sentence_tree(ant.sentnum)
        #     print ant.sub_sentdict

        # a.debug_alignment(verbose=False)
        exit(0)

    sign = lambda x: 1 if x>=0 else -1
    rand_range = 5
    a = AntecedentClassifier(0,14, 15,19, 20,24)
    seed = 347890 #int(sys.argv[1].split('seed=')[1]) #347890


    a.initialize(pos_tests, seed=seed, save=False, load=True, update=False)
    # a.generate_possible_ants(pos_tests, only_filter=True, strong=True)
    # a.normalize()
    # a.debug_ant_selection(a.train_triggers)
    # a.debug_ant_selection(a.val_triggers)
    # a.set_trigger_type('do', alter_train=True)
    a.baseline_prediction()
    # exit(0)

    # np.random.seed(seed)
    # my_weights = np.array(
    #     [sign(np.random.randint(-1,1)) * rand_range * np.random.rand() 
    #     for _ in range(len(a.train_triggers[0].possible_ants[0].x))])

    # a.W_old = copy(my_weights)
    # a.W_avg = copy(my_weights)


    # 81% val HM, 82% test HM when gold_ant added to Val and Test (cheatily)
    # This was just used to see how it does with gold standard in, and it does do better
    # which is why we are motivated to improve recall of candidate antecedent generation.
    if 'cheat' in sys.argv:
        print 'Cheating!!'
        for trig in a.val_triggers + a.test_triggers:
            has = False
            for ant in trig.possible_ants:
                if ant == trig.gold_ant:
                    has = True
                    break
            if not has:
                trig.possible_ants.append(trig.gold_ant)

    lr = 0.01
    K = 5
    a.C = 7.0
    a.learn_rate = lambda x: lr #if x == 0 or x % 25 else lr**(x/25)
    # do_weights = 'saved_weights/DO_ULTIMATE_RESULTS_c%s_lr%s_k%s'%(str(a.C),str(lr),str(K))
    # a.W_avg = np.load(do_weights+'.npy')

    # name = 'RANDOMNESS_results_by_trig_c%s_lr%s_k%s_randrange%s_'%(str(a.C),str(lr),str(K),str(rand_range))
    name = 'ALL_ULTIMATE_RESULTS_c%s_lr%s_k%s'%(str(a.C),str(lr),str(K))


    a.fit(epochs=2, k=K, verbose=True)
    # a.make_graphs(name)
    # a.log_results(name)
    # np.save('saved_weights/'+name, np.array(a.W_avg))
    print '\nLEARN RATE, C, K, seed:', lr, a.C, K, seed

    print 'Time taken: %0.2f'%(time.clock() - start_time)


"""

-------------
best results:
c = 5.0, k = 5, lr = 0.01 - regular loss function, over 'do', seed=347890 ==> 71 hm val, 73 hm test
c = 5.0, k = 5, lr = 0.01 - regular loss function, over 'do', seed=124312421 ==> 44 hm val, 53 hm test
c = 5.0, k = 5, lr = 0.01 - regular loss function, over 'do', seed=9000 ==> 46 hm val, 58 hm test

c = 5.0, k = 5, lr = 0.01 - regular loss function, over 'do', weights 1 ==> 52 hm val, 64 hm test
c = 5.0, k = 5, lr = 0.01 - regular loss function, over 'do', seed=7777455577 ==> 58 hm val, 76 hm test

c = 5.0, k = 10, lr = 0.01 - regular loss function, over 'do', seed=347890 ==> 63 hm val, 60 hm test
c = 5.0, k = 15, lr = 0.1 - regular loss function, over 'do', seed=347890 ==> 62 hm val, 67 hm test
c = 5.0, k = 20, lr = 0.1 - regular loss function, over 'do', seed=347890 ==> 62 hm val, 67 hm test
c = 5.0, k = 5, lr = 0.001 - regular loss function, over 'do', seed=347890 ==> 67 hm val, 69 hm test
c = 5.0, k = 5, lr = 0.05 - regular loss function, over 'do', seed=347890 ==> 60 hm val, 73 hm test
c = 5.0, k = 5, lr = 0.1 - regular loss function, over 'do', seed=347890 ==> 58 hm val, 73 hm test
c = 5.0, k = 5, lr = 0.5 - regular loss function, over 'do', seed=347890 ==> 58 hm val, 69 hm test
-------------

1) Delete potential antecedents that contain the trigger in them  - DONE
    --> Figure out why there seems to be low recall in getting gold ants for extraction - DONE AND FIXED
    --> Rewrite potential antecedent extraction - DONE

2) Check alignments between gold alignment vs bad antecedent alignment.
    --> Need to make it so that chunks DONT overlap! No word repeats! - DONE no more overlapping
    --> Figure out why there are so many repeated alignment vectors!
        --> look at and compare ones with same vectors
        --> try to abolish arbitrary featurism
    --> Fix the context extraction so that we don't get super big contexts!

3) Think about adding semantic info to the features - word2vec.
    --> Measure to compare the difference between the word content!
    --> Need to differentiate between different clauses of same syntactic structure
    --> *** Add nice word2vec features:
            Take average word2vec angle between all mappings
            Make new features for:
                if nsub/dobj etc. are mapped, make these 4 different features comparing word2vec sim btwn chunks.


ACL DEADLINE = MARCH 18TH

WHAT IS TO BE DONE:
0) fix the graphs so that we know exactly what features we are looking at - there was no error DONE
1) Baseline results (essentially just graph the most recent VP behind the trigger)
2) Implement HEAD MATCH/exact/overlap to add to the results analysis to have a better way to compare - DONE
3) look at the Hardt paper and see what he does.
4) why is our system working well? See how performance changes w.r.t. with/without word2vec features, alignment features etc.
5) run vpe detection with normalized features and an SVM


0) Analyze results with respect to the type of trigger that we are considering. i.e. DO and not DO SO

BASELINE!!!!!!!!!!!!!!!

# Add Hardt features!
# Test changing Loss function to say 0percent if there is NO head in the proposed ant (hard constraint)
# also can think about how we can weight diff words in loss function.
# Make MIRA vizualization and check obj. function scoring of diff things


- try training it on FULL dataset for testing on just do -*
- have a correctness measure of looking at the top K antecedents and if the exact one
is in this list of K then we can say that, well when we miss we are pretty close
- look at some examples and see what the margin of score between the best and second best
- MOAR FEATURESSSSZZ
"""



# We were getting good with c=0.1 and lr = 0.0001, k=5!!!

