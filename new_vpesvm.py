import vector_creation as vc
import nltktree as nt
import numpy as np
import word_characteristics as wc
import time
import old.detectVPE as dv
import vpe_objects as vpe

from file_names import Files
from scipy.sparse import csr_matrix,vstack
from sklearn.svm import SVC, LinearSVC, NuSVC
# from sklearn.svm import
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from os import listdir
from sys import argv

# MODALS = ['can','could','may','must','might','will','would','shall','should']
# BE     = ['be']
# HAVE   = ['have']
# DO     = ['do']
# TO     = ['to']
# SO     = ['so','same','likewise','opposite']
#
# AUX_LEMMAS = MODALS+BE+HAVE+DO+TO+SO
# ALL_CATEGORIES = [MODALS, BE, HAVE, DO, TO, SO]
# ALL_AUXILIARIES = Files().extract_data_from_file(Files.UNIQUE_AUXILIARIES_FILE)

""" ---- Primary Classes and methods. ---- """
class VPEDetectionClassifier:
    SVM = 'SVM'
    NUSVM = 'NuSVC'
    LINEAR_SVM = 'Linear SVC'
    LOGREG = 'Logistic regression'
    NAIVE_BAYES = 'Naive Bayes'
    LOGREGCV = 'Logistic regression CV'
    DECISION_TREE = 'Decision Tree'
    DECISION_TREE_WITH_OPTIONS = 'Decision Tree with options'
    RANDOMFOREST = 'Random Forest'
    ADABOOST = 'Adaboost'

    def __init__(self, start_train, end_train, start_test, end_test):
        self.sentences = vpe.AllSentences()
        self.annotations = vpe.Annotations()
        self.file_names = Files()
        self.all_auxiliaries = vpe.Auxiliaries()
        self.gold_standard_auxs = vpe.Auxiliaries()

        self.hyperplane = None
        self.features = []

        """ Train and test_vectors are lists of csr_matrices in order to save memory. """
        self.m = None
        self.m2 = None
        self.train_vectors = []
        self.train_classes = []
        self.test_vectors = []
        self.test_classes = []
        self.predictions = []
        self.result_vector = []

        self.pre_oversample_length = 0

        self.start_train = start_train
        self.end_train = end_train
        self.start_test = start_test
        self.end_test = end_test

    def set_classifier(self, classifier):
        if classifier==self.SVM: self.hyperplane = SVC()
        elif classifier==self.NUSVM: self.hyperplane = NuSVC(nu=0.9)
        elif classifier==self.LINEAR_SVM: self.hyperplane = LinearSVC()
        elif classifier==self.LOGREG: self.hyperplane = LogisticRegression()
        elif classifier==self.NAIVE_BAYES: self.hyperplane = MultinomialNB()
        elif classifier==self.LOGREGCV: self.hyperplane = LogisticRegressionCV()
        elif classifier==self.DECISION_TREE: self.hyperplane = DecisionTreeClassifier()
        elif classifier==self.DECISION_TREE_WITH_OPTIONS: self.hyperplane = DecisionTreeClassifier(max_depth=10, min_samples_leaf=3)
        elif classifier==self.RANDOMFOREST: self.hyperplane = RandomForestClassifier(n_estimators=100, min_samples_leaf=4)
        elif classifier==self.ADABOOST: self.hyperplane = AdaBoostClassifier(random_state=1917,n_estimators=100)
        else:
            self.hyperplane = classifier

    def set_features(self, features):
        self.features = features

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
                            mrg_matrix = vpe.XMLMatrix(f+'.pos.xml', self.file_names.XML_POS, pos_file=True)

                        for sentdict in mrg_matrix:
                            self.all_auxiliaries.add_auxs(sentdict.get_auxiliaries(), sentnum_modifier=sentnum_modifier)

                        self.gold_standard_auxs.add_auxs(mrg_matrix.get_gs_auxiliaries(section_annotation.get_anns_for_file(f), sentnum_modifier))
                        self.annotations.add_section(section_annotation)
                        self.sentences.add_mrg(mrg_matrix)

                        sentnum_modifier = len(self.sentences)-1
            dnum += 1

        # We now just have to say which auxs are the gold standard ones within the 'all_auxiliaries' object by changing their "is_trigger" attribute.
        crt_gold_aux_idx = 0
        crt_gold_aux = self.gold_standard_auxs.get_aux(crt_gold_aux_idx)
        for aux in self.all_auxiliaries:
            if crt_gold_aux.equals(aux):
                aux.is_trigger = True
                crt_gold_aux_idx += 1
                try:
                    crt_gold_aux = self.gold_standard_auxs.get_aux(crt_gold_aux_idx)
                except IndexError:
                    break

    def fix_test_set_triggers(self):
        """
        Some triggers annotated by B&S were missed in our data importation step,
        here we manually set them as actual triggers.
        """
        for i,aux in enumerate(self.all_auxiliaries.auxs):
            if (aux.sentnum,aux.wordnum) in [(12072,39),(10989,30),(11804,12),(11499,11)]:
                print self.sentences.get_sentence(aux.sentnum)
                print aux
                print
                aux.is_trigger = True
                self.test_classes[i-self.pre_oversample_length] = 1

    def save_classifier(self):
        a = np.array([self.hyperplane])
        np.save('vpe_trained_classifier', a)

    def load_classifier(self):
        a = np.load('vpe_trained_classifier.npy')
        self.hyperplane = a[0]

    def save_data_npy(self, val=False):
        a = np.array([self.gold_standard_auxs, self.annotations, self.sentences, self.all_auxiliaries,
                      self.train_vectors, self.train_classes, self.test_vectors, self.test_classes])
        if val:
            np.save('vpe_detect_data_val', a)
        else:
            np.save('vpe_detect_data_test', a)

    def load_data_npy(self, val=False, all_data=True):
        string = '_NON_MRG' if all_data else ''
        if val:
            a = np.load('vpe_detect_data_val'+string+'.npy')
        else:
            a = np.load('vpe_detect_data_test'+string+'.npy')

        self.gold_standard_auxs = a[0]
        self.annotations = a[1]
        self.sentences = a[2]
        self.all_auxiliaries = a[3]
        self.train_vectors = a[4]
        self.train_classes = a[5]
        self.test_vectors = a[6]
        self.test_classes = a[7]
        self.pre_oversample_length = len(self.train_vectors)

    def normalize(self):
        print 'Normalizing the data...'
        s = StandardScaler(with_mean=False) # No need to do mean on sparse
        s.fit_transform(self.vecs_to_mat(train=True))
        s.transform(self.vecs_to_mat(train=False))

    def make_feature_vectors(self, make_test_vectors=True, make_train_vectors=True, use_old_vectors=False):
        if make_train_vectors:
            self.train_vectors, self.train_classes = [],[]
        if make_test_vectors:
            self.test_vectors, self.test_classes = [],[]

        frequent_words = self.file_names.extract_data_from_file(self.file_names.EACH_UNIQUE_WORD_NEAR_AUX)
        all_pos = self.file_names.extract_data_from_file(self.file_names.EACH_UNIQUE_POS_FILE)
        pos_bigrams = wc.pos_bigrams(all_pos)

        for aux in self.all_auxiliaries:
            sentdict = self.sentences.get_sentence(aux.sentnum)

            if make_train_vectors and self.start_train <= sentdict.get_section() <= self.end_train:
                self.train_vectors.append(csr_matrix(vc.make_vector(sentdict, aux, self.features, vpe.ALL_CATEGORIES,
                                                                    vpe.AUX_LEMMAS, vpe.ALL_AUXILIARIES, frequent_words,
                                                                    all_pos, pos_bigrams, make_old=use_old_vectors)))

                self.train_classes.append(vc.bool_to_int(aux.is_trigger))
                if len(self.train_vectors) % 1000 == 0 or len(self.train_vectors) == 1:
                    print 'Making the %dth training vector...'%(len(self.train_vectors))

            if make_test_vectors and self.start_test <= sentdict.get_section() <= self.end_test:
                self.test_vectors.append(csr_matrix(vc.make_vector(sentdict, aux, self.features, vpe.ALL_CATEGORIES,
                                                                   vpe.AUX_LEMMAS, vpe.ALL_AUXILIARIES, frequent_words,
                                                                   all_pos, pos_bigrams, make_old=use_old_vectors)))

                self.test_classes.append(vc.bool_to_int(aux.is_trigger))
                if len(self.test_vectors) % 1000 == 0 or len(self.test_vectors) == 1:
                    print 'Making the %dth testing vector...'%(len(self.test_vectors))

        self.pre_oversample_length = len(self.train_vectors)

    def oversample(self, multiplier=None):
        if not multiplier:
            multiplier = self.train_classes.count(vc.bool_to_int(False))/self.train_classes.count(vc.bool_to_int(True))

        print 'Oversampling by x%d'%multiplier

        new_features = []
        new_classes = []
        for i in range(0,len(self.train_vectors)):
            if self.train_classes[i] == vc.bool_to_int(True):
                for _ in range(0, multiplier):
                    new_features.append(self.train_vectors[i])
                    new_classes.append(vc.bool_to_int(True))
            else:
                new_features.append(self.train_vectors[i])
                new_classes.append(vc.bool_to_int(False))

        self.train_vectors = new_features
        self.train_classes = new_classes

    def vecs_to_mat(self, train=True):
        if train:
            vecs = self.train_vectors
        else:
            vecs = self.test_vectors

        m = vecs[0]
        for i in range(1,len(vecs)):
            m = vstack((m,vecs[i]), format='csr')
        return m

    def train(self):
        print 'Training the model...'
        if self.m == None:
            self.m = self.train_vectors[0]
            for i in range(1,len(self.train_vectors)):
                self.m = vstack((self.m,self.train_vectors[i]), format='csr')
        self.hyperplane.fit(self.m, np.array(self.train_classes))

    def make_so(self):
        for aux in self.all_auxiliaries:
            sent = self.sentences[aux.sentnum].words
            try:
                if sent[aux.wordnum+1] == 'so' or sent[aux.wordnum+1] == 'likewise':
                    aux.type = 'so'
                if (sent[aux.wordnum+1] == 'the' and sent[aux.wordnum+2] in ['same','opposite']):
                    aux.type = 'so'
            except IndexError:
                pass

    def set_aux_type(self, type_):
        # assert type_ in
        new_train,new_test = [],[]
        new_train_classes,new_test_classes = [],[]
        new_auxs = vpe.Auxiliaries()

        for i in range(len(self.train_vectors)):
            if self.all_auxiliaries.get_aux(i).type == type_:
                new_train.append(self.train_vectors[i])
                new_train_classes.append(self.train_classes[i])
                new_auxs.add_aux(self.all_auxiliaries.get_aux(i))

        for i in range(len(self.train_vectors), len(self.all_auxiliaries)):
            if self.all_auxiliaries.get_aux(i).type == type_:
                new_test.append(self.test_vectors[i-len(self.train_vectors)])
                new_test_classes.append(self.test_classes[i-len(self.train_vectors)])
                new_auxs.add_aux(self.all_auxiliaries.get_aux(i))

        self.train_vectors = new_train
        self.train_classes = new_train_classes
        self.test_vectors = new_test
        self.test_classes = new_test_classes
        self.all_auxiliaries = new_auxs

    def analyze_auxs(self):
        d = {}
        for aux in self.all_auxiliaries:
            if aux.is_trigger:
                if not d.has_key(aux.type):
                    d[aux.type] = [aux]
                else:
                    d[aux.type].append(aux)
        print d.keys()
        total = 0
        for k in d:
            total += len(d[k])
            print k, len(d[k])
        print total

    def test(self, mat=None):
        print 'Testing the model...'
        if mat == None:
            if self.m2 == None:
                self.m2 = self.test_vectors[0]
                for j in range(1,len(self.test_vectors)):
                    self.m2 = vstack((self.m2,self.test_vectors[j]), format='csr')
            self.predictions = self.hyperplane.predict(self.m2)
        else:
            self.predictions = self.hyperplane.predict(mat)

    def test_my_rules(self, original_rules=False, idxs=None):
        self.predictions = []
        print 'Length of test set: %d, length of All_auxs-training vectors: %d'%(len(self.test_classes),len(self.all_auxiliaries)-len(self.train_vectors))
        for i in range(self.pre_oversample_length,len(self.all_auxiliaries)):
            if idxs==None or i in idxs:
                aux = self.all_auxiliaries.get_aux(i)
                sendict = self.sentences.get_sentence(aux.sentnum)
                tree = sendict.get_nltk_tree()
                word_subtree_positions = nt.get_smallest_subtree_positions(tree)

                if not original_rules:
                    if aux.type == 'modal': self.predictions.append(vc.bool_to_int(wc.modal_rule(sendict, aux, tree, word_subtree_positions)))
                    elif aux.type == 'be': self.predictions.append(vc.bool_to_int(wc.be_rule(sendict, aux)))
                    elif aux.type == 'have': self.predictions.append(vc.bool_to_int(wc.have_rule(sendict, aux)))
                    elif aux.type == 'do': self.predictions.append(vc.bool_to_int(wc.do_rule(sendict, aux, tree, word_subtree_positions)))
                    elif aux.type == 'so': self.predictions.append(vc.bool_to_int(wc.so_rule(sendict, aux)))
                    elif aux.type == 'to': self.predictions.append(vc.bool_to_int(wc.to_rule(sendict, aux)))
                else:
                    auxidx = aux.wordnum
                    if aux.type == 'modal': self.predictions.append(vc.bool_to_int(dv.modalcheck(sendict, auxidx, tree, word_subtree_positions)))
                    elif aux.type == 'be': self.predictions.append(vc.bool_to_int(dv.becheck(sendict, auxidx, tree, word_subtree_positions)))
                    elif aux.type == 'have': self.predictions.append(vc.bool_to_int(dv.havecheck(sendict, auxidx, tree, word_subtree_positions)))
                    elif aux.type == 'do': self.predictions.append(vc.bool_to_int(dv.docheck(sendict, auxidx, tree, word_subtree_positions)))
                    elif aux.type == 'so': self.predictions.append(vc.bool_to_int(dv.socheck(sendict, auxidx, tree, word_subtree_positions)))
                    elif aux.type == 'to': self.predictions.append(vc.bool_to_int(dv.tocheck(sendict, auxidx, tree, word_subtree_positions)))

    def results(self, name, set_name='Test', test_classes=None, test_auxs=None, v=False):
        if test_classes == None:
            test_classes = self.test_classes

        if test_auxs == None:
            print 'WOIJOWIRJWOIRJWOIRJWORIQJWRPOWQJRPOWQJRPOJQWRPOQWJR'
            # test_auxs = self.all_auxiliaries

        if len(self.predictions) != len(test_classes):
            raise Exception('The number of test vectors != the number of test classes!')

        result_vector = []
        tp,fp,fn = 0.0,0.0,0.0
        for i in range(len(test_classes)):
            if v:
                sent = self.sentences.get_sentence(test_auxs[i].sentnum)

            if test_classes[i] == self.predictions[i] == vc.bool_to_int(True):
                result_vector.append(('tp',i))
                if v:
                    print 'TP',sent.file,sent
                    print test_auxs[i],'\n'
                tp += 1

            elif test_classes[i] == vc.bool_to_int(True) and self.predictions[i] == vc.bool_to_int(False):
                result_vector.append(('fn',i))
                if v:
                    print 'FN',sent.file,sent
                    print test_auxs[i],'\n'
                fn += 1

            elif test_classes[i] == vc.bool_to_int(False) and self.predictions[i] == vc.bool_to_int(True):
                result_vector.append(('fp',i))
                if v:
                    print 'FP',sent.file,sent
                    print test_auxs[i],'\n'
                fp += 1

        try: precision = tp/(tp+fp)
        except ZeroDivisionError: precision = 0.0
        try: recall = tp/(tp+fn)
        except ZeroDivisionError: recall = 0.0

        if precision == 0.0 or recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2*precision*recall/(precision+recall)

        print '\nResults from applying \"%s\" on the %s set.'%(name,set_name)
        print 'TP: %d, FP: %d, FN: %d'%(tp,fp,fn)
        print 'Precision: %0.3f'%precision
        print 'Recall: %0.3f'%recall
        print 'F1: %0.3f\n'%f1

        result_vector += [('precision',precision),('recall',recall), ('f1',f1)]
        self.result_vector = result_vector

    def log_results(self, file_name):
        train_length = self.pre_oversample_length
        with open(self.file_names.RESULT_LOGS_LOCATION + file_name + '.txt', 'w') as f:
            for pair in self.result_vector:
                if pair[0] in ['tp','fp','fn']:
                    aux = self.all_auxiliaries.get_aux(pair[1] + train_length)
                    sentdict = self.sentences.get_sentence(aux.sentnum)

                    # print aux
                    # print pair[0].upper(),
                    # sentdict.print_sentence()
                    # print

                    f.write('%s\n%s: %s\n\n'%(str(aux),pair[0].upper(),sentdict.words_to_string()))
                else:
                    f.write('\n%s: %0.3f\n'%(pair[0],pair[1]))

    def initialize2(self, aux_type=None, rules_test=False, oversample=5):
        if aux_type:
            self.set_aux_type(aux_type)

        if not rules_test:
            self.oversample(multiplier=oversample)



if __name__ == '__main__':
    start_time = time.clock()

    features = vc.get_all_features(old_rules=True)
    print 'Features:',
    print features

    if len(argv) == 5:
        classifier = VPEDetectionClassifier(int(argv[1]),int(argv[2]),int(argv[3]),int(argv[4]))

    # c.all_auxiliaries.print_gold_auxiliaries()

    # for a in c.gold_standard_auxs:
    #     try:
    #         sent = c.sentences.get_sentence(a.sentnum)
    #         sent.print_sentence()
    #         if 'his' in sent.words and 'wife' in sent.words:
    #             print sent.get_nltk_tree()
    #         # print c.sentences.get_sentence(a.sentnum).words[a.wordnum],
    #     except IndexError:
    #         print 'Error on sentnum: %d'%a.sentnum
    #     print a

    # c.file_names.make_all_the_files(c.sentences)
    classifier = VPEDetectionClassifier(0,14,15,19)
    OVERSAMPLE = 5

    rules = False
    load = True
    load_classifier = True
    if not load:
        for t1,t2 in [(20,24)]:
            classifier = VPEDetectionClassifier(0,14,t1,t2)
            classifier.load_data_npy(val=(classifier.start_test==15 and classifier.end_test==19), all_data=False) # We only use MRG files.
            classifier.fix_test_set_triggers()
            exit(0)

            classifier.set_features(features)
            classifier.make_so()
            classifier.make_feature_vectors(make_train_vectors=True, make_test_vectors=True, use_old_vectors=False)
            classifier.normalize()
            classifier.save_data_npy(val=(classifier.start_test==15 and classifier.end_test==19))
        print 'Time taken: %0.2f'%(time.clock()-start_time)
        exit(0)
    else:
        if not load_classifier:
            classifier.load_data_npy(val=False, all_data=False)
            classifier.initialize2(rules_test=False, oversample=OVERSAMPLE)
            classifier.set_classifier(classifier.LOGREGCV)
            classifier.train()
            classifier.save_classifier()
            exit(0)
        else:
            classifier.load_classifier()

        for b in [False]:
            classifier.load_data_npy(val=b, all_data=False)

            if not b:
                classifier.fix_test_set_triggers()

            if rules:
                classifier.test_my_rules(original_rules=False)
                classifier.results('Deterministic Rule testing', set_name='Validation' if b else 'Test', v=False)
            else:
                classifier.test()
                classifier.results('%s oversample %d'%(classifier.LOGREGCV,OVERSAMPLE), set_name='Validation' if b else 'Test', v=False)

            for t in ['do','to','so','be','modal','have']:
                auxs = []
                idxs = []
                all_aux_idxs = []
                for i in range(classifier.pre_oversample_length, len(classifier.all_auxiliaries)):
                    if classifier.all_auxiliaries.get_aux(i).type == t:
                        auxs.append(classifier.all_auxiliaries.get_aux(i))
                        idxs.append(i-classifier.pre_oversample_length)
                        all_aux_idxs.append(i)

                print t.upper()
                if rules:
                    classifier.test_my_rules(idxs=all_aux_idxs)
                    classifier.results(t.capitalize()+': rule-based', v=True,
                                       set_name='Validation' if b else 'Test',
                                       test_classes=list(np.array(classifier.test_classes)[idxs]),
                                       test_auxs=list(np.array(classifier.all_auxiliaries.auxs)[all_aux_idxs]))
                else:
                    classifier.test(mat=classifier.m2[idxs])
                    classifier.results(t.capitalize()+': %s oversample 5'%classifier.LOGREGCV, v=True,
                                       set_name='Validation' if b else 'Test',
                                       test_classes=list(np.array(classifier.test_classes)[idxs]),
                                       test_auxs=list(np.array(classifier.all_auxiliaries.auxs)[all_aux_idxs]))


            # c.set_classifier(c.LINEAR_SVM)
            # c.train()
            # c.test()
            # c.results(type_.capitalize()+': %s oversample 5'%c.LINEAR_SVM, set_name='Validation' if b else 'Test')

            # for C in [0.035]: # BEST HYPER-PARAM FOR LINEAR
            #     # classifier.set_classifier(SVC(C=C))
            #     classifier.set_classifier(LinearSVC(C=C))
            #     print 'C =',C
            #     classifier.train()
            #     classifier.test()
            #     # classifier.results(type_.capitalize()+': %s oversample 5'%classifier.SVM, set_name='Validation' if b else 'Test')
            #     classifier.results(type_.capitalize()+': %s oversample 5'%classifier.LINEAR_SVM, set_name='Validation' if b else 'Test')

            # nu = 0.00
            # while nu<=1:
            #     c.set_classifier(NuSVC(nu=nu))
            #     try:
            #         print 'NU = ',nu
            #         c.train()
            #         c.test()
            #         c.results(type_.capitalize()+': %s oversample 5'%c.NUSVM, set_name='Validation' if b else 'Test')
            #     except ValueError:
            #         print 'Infeasible Nu!!!'
            #     nu += 0.05


            print '--------------------------------------'

    # MRG data set: test - 80 P, 89 R
    # MRG data set: vali - xx P, xx R
    # MRG Do val 85 precision, 83 recall, Do test 95 precision 95 recall
    # MRG So val 100 p 92 r, So test 64 P, 88 R
    # MRG Modal val 95 p 95 r , modal test 92 p, 92 r

    # FULL data set: test - 76 P, 87 R
    # FULL data set: vali - 78 P, 81 R
    # FULL Do val 85 p, 83 r, Do test 94 P, 96 R
    # FULL So val xx p, xx r, So test xx P, xx R
    # FULL Modal val xx p xx r , modal test xx p, xx r
    # FULL My Rules: Val 67 p, 64 r; test 66 p, 70 r

    def test():
        classifier.set_classifier(classifier.LINEAR_SVM)
        classifier.train()
        classifier.test()
        classifier.results('%s normalized oversample 5'%classifier.LINEAR_SVM)

        classifier.set_classifier(classifier.SVM)
        classifier.train()
        classifier.test()
        classifier.results('%s normalized oversample 5'%classifier.SVM)

        classifier.set_classifier(classifier.DECISION_TREE)
        classifier.train()
        classifier.test()
        classifier.results('%s normalized oversample 5'%classifier.DECISION_TREE)

        classifier.set_classifier(classifier.DECISION_TREE_WITH_OPTIONS)
        classifier.train()
        classifier.test()
        classifier.results('%s normalized oversample 5'%classifier.DECISION_TREE_WITH_OPTIONS)

        classifier.set_classifier(classifier.NAIVE_BAYES)
        classifier.train()
        classifier.test()
        classifier.results('%s normalized oversample 5'%classifier.NAIVE_BAYES)

