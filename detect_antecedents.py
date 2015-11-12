import vpe_objects as vpe
import numpy as np
from os import listdir

def score(gold_ant_x, ant_x):
    return np.dot(gold_ant_x, ant_x)

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
        self.generate_possible_ants()
        self.build_feature_vectors()

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

    def generate_possible_ants(self):
        for trigger in self.train_triggers + self.test_triggers:
            self.sentences.set_possible_ants(trigger)

    def bestk_ants(self, trigger, k=5):

        return

    def build_feature_vectors(self):
        return

    def debug_ant_selection(self):
        missed = 0
        for trigger in self.train_triggers:
            try:
                for poss_ant in trigger.possible_ants:
                    if poss_ant.sub_sentdict == trigger.gold_ant.sub_sentdict:
                        raise vpe.Finished()
            except vpe.Finished:
                continue
            missed += 1
            print '\nMISSED ANT FOR TRIGGER',trigger
            print 'ANT: ',trigger.gold_ant.sub_sentdict.pos,trigger.gold_ant.sub_sentdict.words
            print self.sentences.get_sentence(trigger.sentnum)
        print 'Missed this many gold possible ants: %d'%missed

    def fit(self):
        for trigger in self.train_triggers:
            print trigger

if __name__ == '__main__':
    a = AntecedentClassifier(0,0,-1,-1)
    print 'We missed %d vpe instances.'%a.missed_vpe
    # for trig in a.train_triggers:
    #     print '--------------------------------------------'
    #     print 'TRIGGER',trig
    #     print a.sentences.get_sentence(trig.sentnum)
    #     print 'ANTECEDENT:',trig.gold_ant
    #     print a.sentences.get_sentence(trig.gold_ant.sentnum)
    # print '---------'
    a.debug_ant_selection()
