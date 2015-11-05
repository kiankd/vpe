import vpe_objects as VPE
import numpy as np

from os import listdir

class AntecedentClassifier:
    def __init__(self, train_start, train_end, test_start, test_end):
        self.sentences = VPE.AllSentences()
        self.annotations = VPE.Annotations()
        self.file_names = VPE.Files()
        self.triggers = VPE.Auxiliaries()

        self.gold_ants = VPE.Antecedents()
        self.train_ants = VPE.Antecedents()
        self.test_ants = VPE.Antecedents()

        self.predictions = None

        self.start_train = train_start
        self.end_train = train_end
        self.start_test = test_start
        self.end_test = test_end

        self.missed_vpe = 0

    def import_data(self, test=None):
        dirs = listdir(self.file_names.XML_MRG)
        dirs.sort()

        sentnum_modifier = -1
        dnum = 0
        for d in dirs:
            subdir = d+self.file_names.SLASH_CHAR
            if subdir.startswith('.'): continue
            if (self.start_train <= dnum <= self.end_train) or (self.start_test <= dnum <= self.end_test):
                section_annotation = VPE.AnnotationSection(subdir, self.file_names.VPE_ANNOTATIONS)

                vpe_files = list(set([annotation.file for annotation in section_annotation]))
                vpe_files.sort()

                for f in vpe_files:
                    if not test or (test and f in test):
                        # Here we are now getting the non-MRG POS file that we had neglected to get before.
                        try:
                            mrg_matrix = VPE.XMLMatrix(f+'.mrg.xml', self.file_names.XML_MRG+subdir)
                        except IOError:
                            continue
                            # mrg_matrix = XMLMatrix(f+'.pos.xml', self.file_names.XML_POS, pos_file=True)

                        """ Note that I am using the gold standard triggers here. """
                        section_triggers = mrg_matrix.get_gs_auxiliaries(section_annotation.get_anns_for_file(f), sentnum_modifier)
                        # self.triggers.add_auxs(mrg_matrix.get_gs_auxiliaries(section_annotation.get_anns_for_file(f), sentnum_modifier))
                        try:
                            self.gold_ants.add_ants(mrg_matrix.get_gs_antecedents(section_annotation.get_anns_for_file(f), section_triggers, sentnum_modifier))
                            self.triggers.add_auxs(section_triggers)
                        except AssertionError:
                            self.missed_vpe += 1
                            print 'Warning: different number of triggers and antecedents!'

                        self.annotations.add_section(section_annotation)
                        self.sentences.add_mrg(mrg_matrix)

                        sentnum_modifier = len(self.sentences)-1
            dnum += 1

if __name__ == '__main__':
    a = AntecedentClassifier(0,24,4,4)
    a.import_data()
    print 'We missed %d vpe instances.'%a.missed_vpe
    # for ant in a.gold_ants:
    #     print '--------------------------------------------'
    #     print a.sentences.get_sentence(ant.sentnum)
    #     print 'ANTECEDENT:',ant
    #     print a.sentences.get_sentence(ant.trigger.sentnum)
    #     print ant.trigger
