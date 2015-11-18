import vpe_objects as vpe
import numpy as np
import word_characteristics as wc
import antecedent_vector_creation as avc
import truth
from heapq import nlargest
from os import listdir
from pyprind import ProgBar
from optimize_mira_dual import update_weights


class AntecedentClassifier:
    def __init__(self, train_start, train_end, test_start, test_end, learn_rate=1.0):
        self.sentences = vpe.AllSentences()
        self.annotations = vpe.Annotations()
        self.file_names = vpe.Files()

        self.train_ants = []
        self.test_ants = []
        self.train_triggers = []
        self.test_triggers = []

        self.start_train = train_start
        self.end_train = train_end
        self.start_test = test_start
        self.end_test = test_end
        self.W = []
        self.learn_rate = learn_rate

        self.missed_vpe = 0
        self.num_features = 0

    def initialize(self, pos_tests, sd=99):
        self.import_data()
        print 'Generating possible antecedents...'
        self.generate_possible_ants(pos_tests,sd)
        print 'Building feature vectors...'
        self.build_feature_vectors()
        self.initialize_weights()

    def import_data(self, test=None):
        dirs = listdir(self.file_names.XML_MRG)
        dirs.sort()

        sentnum_modifier = -1
        dnum = 0
        for d in dirs:
            subdir = d+self.file_names.SLASH_CHAR
            if subdir.startswith('.'): continue
            if (self.start_train <= dnum <= self.end_train) or (self.start_test <= dnum <= self.end_test):
                section_annotation = vpe.AnnotationSection(subdir, self.file_names.VPE_ANNOTATIONS)

                vpe_files = list(set([annotation.file for annotation in section_annotation]))
                vpe_files.sort()

                for f in vpe_files:
                    if not test or (test and f in test):
                        # Here we are now getting the non-MRG POS file that we had neglected to get before.
                        try:
                            mrg_matrix = vpe.XMLMatrix(f+'.mrg.xml', self.file_names.XML_MRG+subdir)
                        except IOError:
                            continue
                            # mrg_matrix = XMLMatrix(f+'.pos.xml', self.file_names.XML_POS, pos_file=True)

                        """ Note that I am using the gold standard triggers here. """
                        section_triggers = mrg_matrix.get_gs_auxiliaries(section_annotation.get_anns_for_file(f), sentnum_modifier)
                        # self.triggers.add_auxs(mrg_matrix.get_gs_auxiliaries(section_annotation.get_anns_for_file(f), sentnum_modifier))
                        try:
                            if self.start_train <= dnum <= self.end_train:
                                self.train_ants += mrg_matrix.get_gs_antecedents(section_annotation.get_anns_for_file(f), section_triggers, sentnum_modifier)
                                self.train_triggers += section_triggers

                            if self.start_test <= dnum <= self.end_test:
                                self.test_ants += mrg_matrix.get_gs_antecedents(section_annotation.get_anns_for_file(f), section_triggers, sentnum_modifier)
                                self.test_triggers += section_triggers

                        except AssertionError:
                            self.missed_vpe += 1
                            print 'Warning: different number of triggers and antecedents!'

                        self.annotations.add_section(section_annotation)
                        self.sentences.add_mrg(mrg_matrix)

                        sentnum_modifier = len(self.sentences)-1
            dnum += 1

    def score(self, ant):
        return np.dot(self.weights, ant.x)

    def generate_possible_ants(self, pos_tests, sd=99):
        for trigger in self.train_triggers + self.test_triggers:
            trigger.possible_ants = []
            self.sentences.set_possible_ants(trigger, pos_tests, search_distance=sd)

    def bestk_ants(self, trigger, w, k=5, random=0):
        for a in range(len(trigger.possible_ants)):
            trigger.possible_ants[a].set_score(w)
        return nlargest(k, trigger.possible_ants, key=vpe.Antecedent.get_score)

    def build_feature_vectors(self):
        all_pos_tags = truth.extractdatafromfile(truth.EACH_UNIQUE_POS_FILE)
        for trigger in self.train_triggers + self.test_triggers:
            for ant in trigger.possible_ants:
                ant.x = avc.build_feature_vector(ant, trigger, self.sentences, all_pos_tags)
                if self.num_features == 0:
                    self.num_features = len(ant.x)
                if len(ant.x) != self.num_features:
                    raise Exception("DIFFERENT NUMBER OF FEATURES SHIT")
            trigger.gold_ant.x = avc.build_feature_vector(ant, trigger, self.sentences, all_pos_tags)
        return

    # noinspection PyArgumentList
    def initialize_weights(self, seed=1917):
        np.random.seed(seed)
        self.W.append(np.random.rand(len(self.train_triggers[0].possible_ants[0].x)))

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
            'Average length of possible ants: %d\n'%np.mean(lengths)]

        for x in s:
            print x
        print '================================================'

        if write_to_file:
            for x in s:
                f.write(x+'\n')
            f.close()

    def fit(self, epochs=5, verbose=True, k=5, random=0, debug=False):
        i=0
        for n in range(epochs):
            for trigger in self.train_triggers:
                bestk = self.bestk_ants(trigger, self.W[i], k=k, random=random)
                self.W.append( update_weights(self.W[i], bestk, trigger.gold_ant, self.loss_function, C=self.learn_rate) )
                i+=1
            if debug and n>0:
                print 'Epoch %d - Average error: %0.2f'%(n, self.accuracy(self.predict()))
                print 'Avg difference: %0.3f'%(np.mean(self.W[i]-self.W[0]))


    def loss_function(self, gold_ant, proposed_ant):
        """
        @type gold_ant: vpe.Antecedent
        @type proposed_ant: vpe.Antecedent
        """
        gold_vals = gold_ant.word_pos_tuples()
        proposed_vals = proposed_ant.word_pos_tuples()
        tp = float(len([tup for tup in proposed_vals if tup in gold_vals]))
        fp = float(len([tup for tup in proposed_vals if not tup in gold_vals]))
        fn = float(len([tup for tup in gold_vals if not tup in proposed_vals]))
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        try:
            return 1.0 - (2.0*precision*recall)/(precision+recall)
        except ZeroDivisionError:
            return 1.0

    def predict(self):
        # print self.W
        # print sum(self.W)/len(self.W)
        weights = np.array(sum(self.W)/len(self.W))
        # print weights
        predictions = []
        for trigger in self.test_triggers:
            predictions.append(self.bestk_ants(trigger, weights, k=1)[0])
        return predictions

    def accuracy(self, predictions):
        losses = []
        for ant in predictions:
            losses.append(self.loss_function(ant.trigger.gold_ant, ant))
        return float(np.mean(losses, dtype=np.float64))

if __name__ == '__main__':
    a = AntecedentClassifier(0,0,0,0, learn_rate=1.0)
    print 'We missed %d vpe instances.'%a.missed_vpe
    # for trig in a.train_triggers:
    #     print '--------------------------------------------'
    #     print 'TRIGGER',trig
    #     print a.sentences.get_sentence(trig.sentnum)
    #     print 'ANTECEDENT:',trig.gold_ant
    #     print a.sentences.get_sentence(trig.gold_ant.sentnum)
    # print '---------'

    # Using functional programming here because it's nice
    pos_tests = ['VP','ADJ-PRD','NP-PRD', wc.is_adjective, wc.is_verb]
    a.initialize(pos_tests, sd=5)

    a.fit(epochs=500, debug=True, k=10)
    print np.array(sum(a.W)/len(a.W))
    print a.W[-1]