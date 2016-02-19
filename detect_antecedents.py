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
from heapq import nlargest
from os import listdir
from pyprind import ProgBar
from random import randrange,sample
from matplotlib import pyplot as plt
from alignment import alignment_matrix
from copy import copy

from subprocess import call
call('clear')

def shuffle(list_):
    np.random.shuffle(list_)
    return list_

class AntecedentClassifier:
    SCHEDULE_FREQUENCY = 4

    def __init__(self, train_start, train_end, val_start, val_end, test_start, test_end, learn_rate=lambda x: 1.0/(x+1), C=1.0):
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

    def initialize(self, pos_tests, seed=1917, W=None, test=0, delete_random=0.0,
                   save=False, load=False, update=False, verbose=False):
        if not load:
            self.import_data()
            self.generate_possible_ants(pos_tests, test=test, delete_random=delete_random)
            self.debug_ant_selection()
            self.build_feature_vectors()
            self.normalize()

        else:
            self.load_imported_data()
            if update:
                self.normalize()

            # if update:
            #     self.generate_possible_ants(pos_tests)
            #     self.build_feature_vectors()
            #     self.normalize()
            # c=0
            # for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
            #     for ant in trigger.possible_ants:
            #         if ant.contains_trigger():
            #             c+=1
            #             trigger.possible_ants.remove(ant)
            # print 'Deleted %d ants!'%c

        if save:
            self.save_imported_data()

        self.initialize_weights(initial=W, seed=seed)

    def import_data(self, test=None):
        """Import data from our XML directory and find all of the antecedents. Takes a bit of time."""
        dirs = listdir(self.file_names.XML_MRG)
        dirs.sort()

        sentnum_modifier = -1
        dnum = 0
        for d in dirs:
            subdir = d+self.file_names.SLASH_CHAR
            if subdir.startswith('.'): continue

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
                            mrg_matrix = vpe.XMLMatrix(f+'.mrg.xml', self.file_names.XML_MRG+subdir, get_deps=True)
                        except IOError:
                            continue
                            # mrg_matrix = XMLMatrix(f+'.pos.xml', self.file_names.XML_POS, pos_file=True)

                        """ Note that I am using the gold standard triggers here. """
                        file_annotations = section_annotation.get_anns_for_file(f)
                        section_triggers = mrg_matrix.get_gs_auxiliaries(file_annotations, sentnum_modifier)
                        # self.triggers.add_auxs(mrg_matrix.get_gs_auxiliaries(file_annotations, sentnum_modifier))
                        try:
                            if self.start_train <= dnum <= self.end_train:
                                self.train_ants += mrg_matrix.get_gs_antecedents(file_annotations, section_triggers, sentnum_modifier)
                                self.train_triggers += section_triggers

                            if self.start_val <= dnum <= self.end_val:
                                self.val_ants += mrg_matrix.get_gs_antecedents(file_annotations, section_triggers, sentnum_modifier)
                                self.val_triggers += section_triggers

                            if self.start_test <= dnum <= self.end_test:
                                self.test_ants += mrg_matrix.get_gs_antecedents(file_annotations, section_triggers, sentnum_modifier)
                                self.test_triggers += section_triggers

                        except AssertionError:
                            self.missed_vpe += 1
                            print 'Warning: different number of triggers and antecedents!'

                        self.annotations.add_section(section_annotation)
                        self.sentences.add_mrg(mrg_matrix)

                        sentnum_modifier = len(self.sentences)-1
            dnum += 1

    def generate_possible_ants(self, pos_tests, test=0, delete_random=0.0):
        """Generate all candidate antecedents."""
        print 'Generating possible antecedents...'
        if test: # ONLY FOR TESTING! THIS CHEATS!!
            for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
                trigger.possible_ants = []
                self.sentences.set_possible_ants(trigger, pos_tests)
                trigger.possible_ants = [trigger.gold_ant] + trigger.possible_ants[0:test]
            return

        # Fair testing.
        for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
            trigger.possible_ants = []

            self.sentences.set_possible_ants(trigger, pos_tests)
            trigger.possible_ants = list(trigger.possible_ants)

        # Delete antecedents that contain the trigger in them:
        c = 0
        num_ants = 0
        duplicates = 0
        all_ants = set([])
        for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
            deletes = []
            for ant in trigger.possible_ants:
                num_ants += 1

                size = len(all_ants)
                all_ants.add((ant.sentnum,ant.start,ant.end))

                if len(all_ants) == size:
                    duplicates += 1
                    deletes.append(ant)

                elif ant.contains_trigger():
                    c += 1
                    deletes.append(ant)

            for ant in deletes:
                trigger.possible_ants.remove(ant)

        print '%d total ants after deletion - deleted the %d duplicates!'%(num_ants-duplicates, duplicates)
        print 'Deleted %d bad antecedents!'%c

    def bestk_ants(self, trigger, w, k=5):
        """Return the best k antecedents given the weight vector w with respect to the score attained."""
        for a in range(len(trigger.possible_ants)):
            trigger.possible_ants[a].set_score(w)

        return nlargest(k, trigger.possible_ants, key=vpe.Antecedent.get_score)

    def build_feature_vectors(self, verbose=True):

        word2vec_dict = truth.loadword2vecs()
        all_pos_tags = truth.extract_data_from_file(truth.EACH_UNIQUE_POS_FILE) # We only want to import this file once.

        print 'Building feature vectors...'

        bar = ProgBar(len(self.train_triggers)+len(self.val_triggers)+len(self.test_triggers))

        for trigger in self.train_triggers + self.val_triggers + self.test_triggers:
            alignment_matrix(self.sentences, trigger, word2vec_dict,
                             dep_names= ('prep','nsubj','dobj','nmod','adv','conj','vmod','amod','csubj'),
                             pos_tags= all_pos_tags)
            bar.update()

        return

    def initialize_weights(self, initial=None, seed=1917):
        np.random.seed(seed)
        # self.W_old = np.ones(len(self.train_triggers[0].possible_ants[0].x))
        if initial==None:
            self.W_old = np.random.rand(len(self.train_triggers[0].possible_ants[0].x))
        else:
            self.W_old = initial
        self.W_avg = copy(self.W_old)

    def debug_ant_selection(self, verbose=False, write_to_file=None):
        missed = 0
        total = 0
        lengths = []
        missed_pos = {}

        if write_to_file:
            f = open(write_to_file, 'w')

        for trigger in self.train_triggers:
            lengths.append(len(trigger.possible_ants))
            try:
                for poss_ant in trigger.possible_ants:
                    if poss_ant.sub_sentdict == trigger.gold_ant.sub_sentdict:
                        raise vpe.Finished()
            except vpe.Finished:
                total += 1
                continue

            total += 1
            missed += 1

            pos = trigger.gold_ant.sub_sentdict.pos[0]
            if pos in missed_pos:
                missed_pos[pos] +=1
            else:
                missed_pos[pos] = 1

            if verbose:
                print '\nMISSED ANT FOR TRIGGER',trigger
                print 'ANT: ',trigger.gold_ant.sub_sentdict.pos,trigger.gold_ant.sub_sentdict.words
                print self.sentences.get_sentence(trigger.sentnum)
                print self.sentences.get_sentence_tree(trigger.sentnum)
            if write_to_file:
                f.write('Trigger: %s\n'%trigger.__repr__())
                f.write('Antecedent: %s\n'%trigger.gold_ant.__repr__())
                f.write('Tree:\n%s'%self.sentences.get_sentence_tree(trigger.sentnum).__repr__())

        s = ['\nMissed this many gold possible ants: %d'%missed,
            'That is %0.2f percent.'%(float(missed)/total),
            'Average length of possible ants: %d'%np.mean(lengths),
             'Min/Max lengths: %d, %d\n'%(min(lengths),max(lengths))]

        for x in s:
            print x
        print '================================================'

        if write_to_file:
            for x in s:
                f.write(x+'\n')
            f.close()

    def normalize(self):
        print 'Normalizing the data...'
        xtrain,xval,xtest = [],[],[]

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

        X = np.array(xtrain + xval + xtest)
        X = X[:, (X == 0).sum(axis=0) <= len(X)-1] # Remove 0 columns

        xtrain = np.array(X[:len(xtrain)])
        xval = np.array(X[len(xtrain):len(xtrain)+len(xval)])
        xtest = np.array(X[len(xtrain)+len(xval):])

        mean = xtrain.mean(axis=0)
        xtrain -= mean
        xval -= mean
        xtest -= mean

        stdtrain = xtrain.std(axis=0)[1:]
        for v in [np.NAN, np.inf, 0.0]:
            stdtrain[stdtrain == v] = 1.0

        xtrain[:,1:] /= stdtrain # Standard deviation of bias is zero, dont divide it.
        xval[:,1:] /= stdtrain
        xtest[:,1:] /= stdtrain

        i=0
        for trig in self.train_triggers:
            for ant in trig.possible_ants:
                ant.x = xtrain[i]
                i += 1
            trig.gold_ant.x = xtrain[i].flatten()
            i += 1
        i=0
        for trig in self.val_triggers:
            for ant in trig.possible_ants:
                ant.x = xval[i]
                i += 1
            trig.gold_ant.x = xval[i].flatten()
            i += 1
        i=0
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
                print "\nTotal ants for this key: %d"%len(X[key])
                print key
                printed = True

        print 'TOTAL : %d, UNIQUE : %d'%(sum(length_list), len(X))
        print 'Average number of ants corresponding to same vec: %0.2f, min= %d, max= %d'%(np.mean(length_list),
                                                                                           min(length_list),
                                                                                           max(length_list))

    def fit(self, epochs=5, verbose=True, k=5, features_to_analyze=5, c_schedule=1.0):
        # ws = [copy(self.W_old)]
        i=0
        for n in range(epochs):
            for trigger in shuffle(self.train_triggers):
                bestk = self.bestk_ants(trigger, self.W_old, k=k)

                self.W_old = mira.update_weights(self.W_old, bestk, trigger.gold_ant, self.loss_function, C=self.C)
                self.W_avg = ((1.0-self.learn_rate(i)) * self.W_avg) + (self.learn_rate(i) * self.W_old) # Running average.
                self.analyze(self.W_avg, features_to_analyze)
                i+=1

            if n % self.SCHEDULE_FREQUENCY == 0:
                self.C /= c_schedule

            if verbose and n>0:
                train_preds = self.predict(self.train_triggers)
                val_preds = self.predict(self.val_triggers)
                test_preds = self.predict(self.test_triggers)

                self.train_err.append(self.accuracy(train_preds))
                self.val_err.append(self.accuracy(val_preds))
                self.test_err.append(self.accuracy(test_preds))

                self.train_results.append(self.criteria_based_results(train_preds))
                self.val_results.append(self.criteria_based_results(val_preds))
                self.test_results.append(self.criteria_based_results(test_preds))

                print '\nEpoch %d - train/val/test error: %0.2f, %0.2f, %0.2f'\
                      %(n, self.train_err[-1], self.val_err[-1], self.test_err[-1])

                # print 'Trigger sentnum,wordnum: %d,%d'%(trigger.sentnum,trigger.wordnum)
                print self.sentences.get_sentence(trigger.sentnum)

                best_ant = self.bestk_ants(trigger, self.W_avg, k=1)[0]
                # print 'Best_ant sentnum = %d, start,end = %d,%d:'%(best_ant.sentnum,best_ant.start,best_ant.end),
                print 'Best ant: start %d, end %d, trigger wordnum %d: '%(best_ant.start, best_ant.end, best_ant.trigger.wordnum)
                print best_ant
                self.diffs.append(np.mean((self.W_avg-self.W_old)**2))
                print 'Difference btwen avg vector w_old: %0.6f'%(self.diffs[-1])
        self.train_results = np.array(self.train_results)
        self.val_results = np.array(self.val_results)
        self.test_results = np.array(self.test_results)

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

        gold_vals = gold_ant.get_words()
        proposed_vals = proposed_ant.get_words()

        tp = float(len([val for val in proposed_vals if val in gold_vals]))
        fp = float(len([val for val in proposed_vals if not val in gold_vals]))
        fn = float(len([val for val in gold_vals if not val in proposed_vals]))

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        try:
            return 1.0 - (2.0*precision*recall)/(precision+recall)
        except ZeroDivisionError:
            return 1.0

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

            em = antwords == goldwords
            hm = antwords[0] == goldwords[0]
            ho = antwords[0] in goldwords

            if em: exact_match += 1
            if hm: head_match += 1
            if ho: head_overlap += 1

        exact_match /= float(len(predictions))
        head_match /= float(len(predictions))
        head_overlap /= float(len(predictions))

        return [exact_match, head_match, head_overlap]

    def analyze(self, w, num_features):
        self.norms.append(np.linalg.norm(w))

        if not self.feature_vals:
            self.feature_vals = [[] for _ in range(num_features)]
            self.feature_names = ['Bias weight']
            for i in range(num_features-2):
                self.random_features.append(randrange(1,len(w)))
                self.feature_names.append('Rand_feature %d'%self.random_features[i])
            self.feature_names.append('NP dot product/norm')

        self.feature_vals[0].append(w[0])
        for i in range(num_features-2):
            self.feature_vals[i+1].append(w[self.random_features[i]])
        self.feature_vals[-1].append(w[-1])

    def make_graphs(self, name):
        params = 'tests/post_changes/%s'%name

        plt.figure(1)
        plt.title('Running Average Weight Vector 2-Norm over time')
        plt.plot(range(len(self.norms)), self.norms, 'b-')
        plt.savefig(params+'norm_change1.png', bbox_inches='tight')
        plt.clf()

        colors = ['b','y','m','r','g']
        plt.figure(2)
        ax = plt.subplot(111)
        plt.title('Exact Feature Values over time')
        for i in range(len(self.feature_vals)):
            ax.plot(range(len(self.feature_vals[i])), self.feature_vals[i], colors[i%len(colors)]+'-', label=self.feature_names[i])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(params+'feature_value_change1.png', bbox_inches='tight')
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
        plt.savefig(params+'errors.png', bbox_inches='tight')
        plt.clf()

        plt.figure(4)
        plt.title('Squared Difference between average and MIRA weight vector')
        plt.plot(range(len(self.diffs)), self.diffs, 'bo')
        plt.savefig(params+'weight_vector_change1.png', bbox_inches='tight')
        plt.clf()

        plt.figure(5)
        ax = plt.subplot(111)
        plt.title('Head match results over time')
        ax.plot(range(len(self.train_results)), self.train_results[:,1], 'b-', label='Train')
        ax.plot(range(len(self.val_results)), self.val_results[:,1], 'y-', label='Validation')
        ax.plot(range(len(self.test_results)), self.test_results[:,1], 'r-', label='Test')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(params+'results.png', bbox_inches='tight')
        plt.clf()

    def log_results(self, name):
        destination = 'ant_results/'+name

        errors = []
        for lst in self.train_err, self.val_err, self.test_err:
            lowest_error = 100
            epoch = 0
            for i in range(len(lst)):
                if lst[i] < lowest_error:
                    lowest_error = lst[i]
                    epoch = i
            errors.append((lowest_error,epoch))

        results = []
        for lst in self.train_results, self.val_results, self.test_results:
            best_em, best_hm, best_ho = 0,0,0
            em_epoch, hm_epoch, ho_epoch = 0,0,0
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

            results.append((best_em,em_epoch, best_hm,hm_epoch, best_ho,ho_epoch))

        write_string = ''
        names = ['Train','Validation','Test']
        res_names = ['exact match','head match','head overlap']
        for i in range(len(names)):
            write_string += '%s results: lowest error = %0.2f at epoch %d\n'%(names[i],errors[i][0],errors[i][1])
            for j in range(len(res_names)):
                write_string += '\tBest %s = %0.2f at epoch %d\n'%(res_names[j],results[i][j*2],results[i][(j*2)+1])

        with open(destination+'.txt','w') as f:
            f.write(write_string)

        np.save(destination,
                np.array([self.train_err,self.val_err,self.test_err,[self.train_results,self.val_results,self.test_results]]))

    def save_imported_data(self, name=''):
        print 'Saving the data...'
        np.save(self.file_names.IMPORTED_DATA+name, np.array([self.sentences, self.train_triggers, self.val_triggers, self.test_triggers]))

    def load_imported_data(self, name=''):
        print 'Loading the data...'
        folder = self.file_names.IMPORTED_DATA+name
        data = np.load(self.file_names.IMPORTED_DATA)
        self.sentences = data[0]
        self.train_triggers = data[1]
        self.val_triggers = data[2]
        self.test_triggers = data[3]

if __name__ == '__main__':
    pos_tests = ['VP', wc.is_adjective, wc.is_verb]

    try:
        debug = sys.argv[1] == 'debug'
    except IndexError:
        debug = False
        pass

    start_time = time.clock()

    # if not debug:
    #     a = AntecedentClassifier(0,0, None,None, None,None, C=0.075, learn_rate=lambda x: 0.0001)
    #     print 'We missed %d vpe instances.'%a.missed_vpe
    #     initial_weights = None
    #
    #     a.initialize(pos_tests, W=initial_weights, test=0, delete_random=0,
    #              save=False, load=False, update=False, seed=2384834)
    #
    # else:
    #     a = AntecedentClassifier(0,0, None,None, None,None)
    #     print 'Debugging...'
    #     a.initialize(pos_tests, save=False, load=False, update=False, seed=2384834)
    #     # a.debug_ant_selection(verbose=False)
    #     a.debug_alignment(verbose=False)
    #     exit(0)

    a = AntecedentClassifier(0,14, 15,19, 20,24)
    a.initialize(['VP', wc.is_adjective, wc.is_verb], seed=9001, save=False, load=True, update=False)

    for sched in [2,5,10]:
        K = 5
        name = 'schedule%d_c0.5_lr0.05_k5'%sched

        a.C = 0.5
        a.learn_rate = lambda x: 0.05

        a.fit(epochs=100, k=K, verbose=True, c_schedule=float(sched))
        a.make_graphs(name)
        a.log_results(name)
        a.reset()
        a.initialize(seed=9001)

        np.save('saved_weights/'+name, np.array(a.W_avg))

    print 'Time taken: %0.2f'%(time.clock() - start_time)



"""

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

"""





# We were getting good with c=0.1 and lr = 0.0001, k=5!!!

