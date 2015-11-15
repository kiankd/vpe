import vpe_objects as vpe
import numpy as np
import word_characteristics as wc
from heapq import nlargest
from os import listdir

class AntecedentClassifier:
    def __init__(self, train_start, train_end, test_start, test_end):
        self.sentences = vpe.AllSentences()
        self.annotations = vpe.Annotations()
        self.file_names = vpe.Files()

        self.train_ants = []
        self.test_ants = []
        self.train_triggers = []
        self.test_triggers = []

        self.predictions = None

        self.start_train = train_start
        self.end_train = train_end
        self.start_test = test_start
        self.end_test = test_end
        self.weights = None

        self.missed_vpe = 0

        self.import_data()

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

    def bestk_ants(self, trigger, k=5):
        for ant in trigger.possible_ants:
            ant.score = self.score(ant.x)
        return nlargest(k, trigger.possible_ants, key=score)

    def build_feature_vectors(self):
        for trigger in self.train_triggers + self.test_triggers:
            for ant in trigger.possible_ants:
                ant.x = avc.build_feature_vector(ant,trigger)
        return

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

    def fit(self):
        for trigger in self.train_triggers:
            print trigger

if __name__ == '__main__':
    a = AntecedentClassifier(0,14,-1,-1)
    print 'We missed %d vpe instances.'%a.missed_vpe
    # for trig in a.train_triggers:
    #     print '--------------------------------------------'
    #     print 'TRIGGER',trig
    #     print a.sentences.get_sentence(trig.sentnum)
    #     print 'ANTECEDENT:',trig.gold_ant
    #     print a.sentences.get_sentence(trig.gold_ant.sentnum)
    # print '---------'

    # Using functional programming here because it's nice
    a.generate_possible_ants(['VP','ADJ-PRD','NP-PRD', wc.is_adjective, wc.is_verb], sd=5)
    a.debug_ant_selection(verbose=False, write_to_file='antecedent_generation.txt')

    # a.generate_possible_ants(['VP','NP','ADJP'])
    # a.debug_ant_selection()
    # a.generate_possible_ants(['VP','NP'])
    # a.debug_ant_selection()
    # a.generate_possible_ants(['VP'])
    # a.debug_ant_selection()

    # a.generate_possible_ants([wc.is_verb, wc.is_adverb, wc.is_adjective])
    # a.debug_ant_selection()
    # a.generate_possible_ants([wc.is_verb, wc.is_adjective])
    # a.debug_ant_selection()
    # a.generate_possible_ants([wc.is_verb])
    # a.debug_ant_selection()
