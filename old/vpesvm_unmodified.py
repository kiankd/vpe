from detectVPE import *
import vectorcreation
import nltktree as NT
from truth import *
import numpy as np
from scipy.sparse import csr_matrix as csr
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from os import listdir
import time

# A machine learning approach to detecting VPE. It has to be done per each auxiliary, not each sentence.
# @author Kian Kenyon-Dean


def getlastauxnumberinfile(mrgmatrix):
    count = 0
    for sentdict in mrgmatrix:
        for i in range(0,len(sentdict['lemmas'])):
            if isauxiliary(sentdict, i):
                count += 1
    return count


def idxindict(idx, dic, sentences_with_vpe, sentnum):
    sentences_to_triggers_dict = dict(zip(sentences_with_vpe, dic))
    try:
        return idx == sentences_to_triggers_dict[sentnum][0]
    except KeyError:
        return False

def auxindict(aux, dic, sentences_with_vpe, sentnum):
    sentences_to_triggers_dict = dict(zip(sentences_with_vpe, dic))
    try:
        return aux == sentences_to_triggers_dict[sentnum][1]
    except KeyError:
        return False

def auxindictlax(aux, dic):
    for idx,trigger in dic:
        print 'Comparing %s with %s'%(aux,trigger)
        if aux == trigger:
            return True
    return False

def auxandidxindict(aux, idx, dic):
    if (idx,aux) in dic:
        return True
    return False

def listtoints(lst):
    return [int(x) for x in lst]

class NSentences:
    def __init__(self):
        self.sentences = []

    def get_sentence(self, n): return self.sentences[n]

    def addsentences(self, mrgmatrix):
        for sentdict in mrgmatrix:
            if sentdict['words'] is not None:
                self.sentences.append(sentdict)

    def printsentence(self, n):
        for w in self.sentences[n]['words']:
            print w,
        print

class GoldVector:
    def __init__(self):
        self.sentence_booleans = []
        self.aux_bools = []
        self.auxiliary_names = []
        self.nth_aux = 0
        self.missed_sentences = 0

    def addsentences(self, gsvpelist, size):
        for i in range(0, size):
            if i in gsvpelist:
                self.sentence_booleans.append(truth(True))
            else:
                self.sentence_booleans.append(truth(False))

    # TODO: FIX THIS - THERE IS A BUG. DONE NOPE
    def addauxs(self, mrgmatrix, gsdict, gs_sent_list, make_file=False):
        crt_sentnum,crt_auxidx = 0,-1
        found_aux = False
        sent_has_vpe = False

        #print gsdict
        #print gs_sent_list

        while crt_sentnum < len(mrgmatrix):

            try:
                old_sentnum = crt_sentnum

                # Reassign the values for the next auxiliary, recursively.
                crt_sentnum,crt_auxidx = nextaux(mrgmatrix, crt_sentnum, crt_auxidx+1)

                if make_file: self.auxiliary_names.append(mrgmatrix[crt_sentnum]['words'][crt_auxidx].lower())

                # This is to check if we missed a GS aux by accident.
                if old_sentnum+1 == crt_sentnum:
                    if sent_has_vpe and not found_aux:

                        auxs = getauxs(mrgmatrix[old_sentnum])
                        crt_auxnum_out_of_total = len(self.aux_bools) - len(auxs)

                        for idx,aux in auxs:

                            if auxindict(aux, idx, gsdict):
                                self.aux_bools[crt_auxnum_out_of_total] = truth(True)
                                found_aux = True
                                print 'RULE 1 Added sentence.\n'

                            crt_auxnum_out_of_total += 1

                        if not found_aux:
                            print 'We missed the sentence below.'#sentence: %d'%old_sentnum
                            printsent(mrgmatrix, old_sentnum)
                            print dict(zip(gs_sent_list,gsdict))
                            print '*',
                            self.missed_sentences += 1

                    found_aux = False
                    sent_has_vpe = False

            except TypeError:
                return

            self.nth_aux += 1

            if crt_sentnum in gs_sent_list:
                sent_has_vpe = True

                if auxandidxindict(mrgmatrix[crt_sentnum]['words'][crt_auxidx], crt_auxidx, gsdict): #idxindict(crt_auxidx, gsdict, gs_sent_list, crt_sentnum):
                    found_aux = True
                    self.aux_bools.append(truth(True))
                else:
                    self.aux_bools.append(truth(False))
            else:
                self.aux_bools.append(truth(False))

    def number_of_positive_auxs(self):
        count = 0
        for aux_bool in self.aux_bools:
            if aux_bool == truth(True): count+=1
        return count

# noinspection PyChainedComparisons
class Classifier:
    TRAINING_SET_LIMIT = 14

    def __init__(self):
        self.each_sentence = NSentences()
        self.aux_feature_vectors = []
        self.gold_standard_sentences = GoldVector()
        self.gold_standard_auxs = GoldVector()
        self.section_split = {-1:0}
        self.hyperplane = None
        self.auxnum_to_sent_map = []

    def setclassifier(self, classifier='', kernel='rbf'):
        if classifier=='SVM': self.hyperplane = SVC(kernel=kernel)
        elif classifier=='Regression': self.hyperplane = LogisticRegression()
        elif classifier=='NaiveBayes': self.hyperplane = MultinomialNB()
        elif classifier=='RegressionCV': self.hyperplane = LogisticRegressionCV()
        elif classifier=='Dtree': self.hyperplane = DecisionTreeClassifier()#max_depth=10, min_samples_leaf=3

    def importdata(self, start_section, end_section, from_file=False):
        dnum = 0
        auxnum = 0

        if from_file:
            self.gold_standard_auxs.aux_bools = listtoints(extractdatafromfile(GOLD_STANDARD_FILE))
            self.gold_standard_auxs.auxiliary_names = extractdatafromfile(UNIQUE_AUXILIARIES_FILE)

        dirs = listdir(XML_MRG)
        dirs.sort()

        for d in dirs:
            subdir = d+SLASH_CHAR
            if subdir.startswith('.'): continue

            if dnum >= start_section and dnum <= end_section:
                annotationmatrix = getvpeannotationsdata(subdir, VPE_ANNOTATIONS)

                files = listdir(XML_MRG+subdir)
                files.sort()
                for test_file in files:
                    if filehasvpe(test_file, annotationmatrix):# and test_file=='wsj_0039.mrg.xml':
                        mrgmatrix = getdataMRG(test_file,XML_MRG+subdir)
                        gs_dict,gs_triggers = goldstandardVPEsentences(test_file, XML_RAW_TOKENIZED, subdir, annotationmatrix, mrgmatrix)

                        # gs_dict is a dictionary mapping sentence numbers to auxiliaries.
                        # gs_triggers is a list of tuples matching the auxiliary word with the aux idx FROM THE RAW FILES.

                        gs_sent_list = [k for k in gs_dict]

                        self.each_sentence.addsentences(mrgmatrix)
                        self.gold_standard_sentences.addsentences(gs_sent_list, len(mrgmatrix))

                        if not from_file:
                            self.gold_standard_auxs.addauxs(mrgmatrix, gs_triggers, gs_sent_list, make_file=True)
                            auxnum = len(self.gold_standard_auxs.aux_bools)-1
                        else:
                            auxnum += getlastauxnumberinfile(mrgmatrix)

                self.section_split[dnum] = auxnum
                dnum+=1

    def setsentmap(self):
        for i in range(0, len(self.each_sentence.sentences)):
            for j in range(0,len(self.each_sentence.sentences[i]['lemmas'])):
                if isauxiliary(self.each_sentence.sentences[i], j):
                    self.auxnum_to_sent_map.append(i)

    def getallauxiliaries(self):
        if not self.gold_standard_auxs.auxiliary_names:
            self.gold_standard_auxs.auxiliary_names = extractdatafromfile(UNIQUE_AUXILIARIES_FILE)
        return self.gold_standard_auxs.auxiliary_names

    def getfeaturevectors(self, start_section, end_section):
        # print 'Getting the GS data from aux %d to aux %d'%(self.section_split[start_section], self.section_split[end_section])
        return self.aux_feature_vectors[self.section_split[start_section] : self.section_split[end_section]]

    def getgsdata(self, start_section, end_section):
        # print 'Getting the GS data from aux %d to aux %d'%(self.section_split[start_section], self.section_split[end_section])
        return self.gold_standard_auxs.aux_bools[self.section_split[start_section] : self.section_split[end_section]]

    def getfeaturevector(self, i):
        return self.aux_feature_vectors[i]

    def getgsentry(self, i):
        return self.gold_standard_auxs.aux_bools[i]

    # Here we are iterating through each sentence to make our feature vectors for each auxiliary.
    def makefeaturevectors(self, features):
        count = 0
        got_first = False

        # This will contain a set of all the columns that are zeros for every vector in the matrix so that we
        # can delete unnecessary columns. I uncommented the code for doing this because the csr matrix compression
        # makes this unnecessary and this takes a while.
        #columns_to_delete_for_cleaning = set()
        length = 0

        word2vec_dict = {}
        if 'word2vec' in features:
            print 'Loading the Word2vec vectors...'
            word2vec_dict = loadword2vecs()

        for sentdict in self.each_sentence.sentences:
            for i in range(0,len(sentdict['lemmas'])):
                if isauxiliary(sentdict, i):
                    if got_first and count%1000==0: print 'Building feature vectors from %d'%count
                    self.aux_feature_vectors.append(self.buildfeaturevector(sentdict, i, word2vec_dict, features))

                    # This is a set of indexes that are equal to zero for the current vector.
                    #empty_columns_in_vector = set([k for k in range(0, len(self.aux_feature_vectors[-1])) if not untruth(self.aux_feature_vectors[-1][k])])

                    if not got_first:
                        #columns_to_delete_for_cleaning = empty_columns_in_vector
                        length = len(self.aux_feature_vectors[0])
                        print 'Feature vector length: %d'%length
                        print 'Building feature vectors from %d'%count
                        got_first=True
                    else:
                        #columns_to_delete_for_cleaning.intersection_update(empty_columns_in_vector)
                        if len(self.aux_feature_vectors[-1]) != length:
                            raise Exception('ERROR - FEATURE VECTORS HAVE DIFFERENT LENGTHS:',length,len(self.aux_feature_vectors[-1]))

                    count += 1

        #print 'Cleaning feature vectors...'
        #self.aux_feature_vectors = vectorcreation.cleanvectors(self.aux_feature_vectors, columns_to_delete_for_cleaning)
        #print 'After cleaning: feature vector length',len(self.aux_feature_vectors[0])

    # It will be structured as follows:
    # First set will be what category of auxiliary it belongs to,
    # i.e. [ismodal, isbe, ishave, isdo, isto, isso] - only one of these will be true.
    # The next set will be what lemma it is, i.e. an extension of the above categories.
    # The next set will be what exact word it is, i.e. is,doing,did,having,being etc.
    def buildfeaturevector(self, sentdict, auxidx, word2vec_dict, features):
        vector = []

        aux_lemma = sentdict['lemmas'][auxidx]
        aux_word  = sentdict['words'][auxidx].lower()

        # These just pinpoint the nature of the auxiliary it is passed.
        vector += vectorcreation.lemmacategoryvector(aux_lemma)
        vector += vectorcreation.lemmavector(aux_lemma)
        vector += vectorcreation.auxwordvector(aux_word, extractdatafromfile(UNIQUE_AUXILIARIES_FILE))

        # This explains the surrounding structure of the auxiliary, three words ahead and in front.
        vector += vectorcreation.auxstruct(sentdict, auxidx, features)

        # These are my features.
        vector += vectorcreation.myfeaturesvector(sentdict, auxidx, features)

        if 'verb_locative' in features:
            vector += vectorcreation.verblocativevector(sentdict, auxidx)

        if 'word2vec' in features:
            vector += vectorcreation.word2vecvectors(sentdict, auxidx, word2vec_dict, average='avg_vecs' in features)

        return vector

    def train(self, start_section, end_section):
        print 'Fitting the hyperplane...'

        #Applying 'csr' to the vectors makes a sparse matrix.
        X_matrix = csr(self.getfeaturevectors(start_section, end_section))
        y_vector = np.array(self.getgsdata(start_section, end_section))

        self.hyperplane.fit(X_matrix, y_vector)

# I have included the options for oversampling, but you actually aren't supposed to oversample
# with the test.
    def devtest(self, start_section, end_section, verbose=False):
        print 'Predicting...'

        #Applying 'csr' to the vectors makes a sparse matrix.
        predictions = self.hyperplane.predict(csr(self.getfeaturevectors(start_section, end_section)))
        self.compare(self.getgsdata(start_section, end_section), predictions, start_section, verbose=verbose)

    def compare(self, gs, my_alg, end_training_set, multiplier=1, verbose=False):
        results = {'tp': 0, 'fp': 0, 'fn': 0, 'tn':0}

        if len(gs) != len(my_alg):
            print 'Error -> the vectors are not the same size!'
            print 'GS length %d, comparison length %d'%(len(gs), len(my_alg))
            quit()

        try:
            training_data_length = len(self.getgsdata(-1,end_training_set))
        except KeyError:
            training_data_length = 0

        for i in range(0, len(gs)):
            # print '%dv%d'%(gs[i],my_alg[i]),

            mapped_index = i+training_data_length

            if gs[i] == truth(True) and my_alg[i] == truth(True):
                results['tp'] += 1
                if verbose and False:
                    print '\nTrue positive: %s'%self.gold_standard_auxs.auxiliary_names[mapped_index],
                    self.each_sentence.printsentence(self.auxnum_to_sent_map[mapped_index])

            elif gs[i] == truth(True) and my_alg[i] == truth(False):
                results['fn'] += 1
                if verbose:
                    print '\nFalse negative: %s'%self.gold_standard_auxs.auxiliary_names[mapped_index],
                    self.each_sentence.printsentence(self.auxnum_to_sent_map[mapped_index])

            elif gs[i] == truth(False) and my_alg[i] == truth(True):
                results['fp'] += 1
                if verbose:
                    print '\nFalse positive: %s'%self.gold_standard_auxs.auxiliary_names[mapped_index],
                    self.each_sentence.printsentence(self.auxnum_to_sent_map[mapped_index])

            else:
                results['tn'] += 1

        for k in results:
            if k in ['tp', 'fn']:
                results[k] /= multiplier

        print results
        scores = f1(results)
        for k in scores:
            print k.capitalize()+' : %0.2f' %scores[k]

    # This will oversample the data by a constant multipler, thus copying M positive occurrences per each positive
    # occurrence.
    def oversample(self, start_section, end_section, multiplier):
        new_feature_vector,new_gs_bools = [],[]

        print 'Adding x%d oversample vectors...'%multiplier
        for i in range(0, len(self.getgsdata(start_section, end_section))):
            if self.gold_standard_auxs.aux_bools[i] == truth(True):
                for k in range(0, multiplier):
                    new_feature_vector.append(self.getfeaturevector(i))
                    new_gs_bools.append(self.getgsentry(i))
            else:
                new_feature_vector.append(self.getfeaturevector(i))
                new_gs_bools.append(self.getgsentry(i))

        return new_feature_vector,new_gs_bools

def make_all_the_files(classifier, appearance_threshold=3, option2=False):
    makefile(UNIQUE_AUXILIARIES_FILE, set(classifier.gold_standard_auxs.auxiliary_names))
    makefile(GOLD_STANDARD_FILE, classifier.gold_standard_auxs.aux_bools)

    words,lemmas,postags,words_near_aux = [],[],[],[]
    for sentdict in classifier.each_sentence.sentences:
        for word in sentdict['words']: words.append(word)
        for i in range(0,len(sentdict['lemmas'])):
            lemmas.append(sentdict['lemmas'][i])
            if isauxiliary(sentdict, i):
                words_near_aux += getsurroundingstruct(sentdict, i, 'pre', 3)['words']
                words_near_aux += getsurroundingstruct(sentdict, i, 'post', 3)['words']
        for pos in sentdict['pos']: postags.append(pos)

    words = set(words)
    lemmas = set(lemmas)
    postags = set(postags)

    compressed_words_near_aux = []
    appears_n_times = {}

    for w in words_near_aux:
        if not w in appears_n_times:
            appears_n_times[w] = 1
        else:
            appears_n_times[w] += 1
            if not option2 and appears_n_times[w] > appearance_threshold:
                compressed_words_near_aux.append(w)

    if option2:
        freqlist = [appears_n_times[w] for w in appears_n_times]
        freqlist.sort()
        freqlist = freqlist[-1*appearance_threshold:len(freqlist)]

        freqlist_frequencies = {}
        for val in freqlist:
            if not val in freqlist_frequencies: freqlist_frequencies[val] = 1
            else: freqlist_frequencies[val] += 1

        for w in appears_n_times:
            if appears_n_times[w] in freqlist_frequencies and freqlist_frequencies[appears_n_times[w]] > 0:
                freqlist_frequencies[appears_n_times[w]] -= 1
                compressed_words_near_aux.append(w)

    words_near_aux = set(compressed_words_near_aux)

    makefile(EACH_UNIQUE_WORD_FILE, words)
    makefile(EACH_UNIQUE_LEMMA_FILE, lemmas)
    makefile(EACH_UNIQUE_POS_FILE, postags)
    makefile(EACH_UNIQUE_WORD_NEAR_AUX, words_near_aux)

def runmodel(classifier, train_start, train_end, test_start, test_end, oversample=1, verbose=False):
    pre_over_sample_features,pre_over_sample_bools = classifier.aux_feature_vectors,classifier.gold_standard_auxs.aux_bools
    classifier.aux_feature_vectors,classifier.gold_standard_auxs.aux_bools = classifier.oversample(train_start, train_end, oversample)
    classifier.train(train_start, train_end)
    classifier.aux_feature_vectors,classifier.gold_standard_auxs.aux_bools = pre_over_sample_features,pre_over_sample_bools
    classifier.devtest(test_start, test_end, verbose=verbose)

def testmyrules(classifier, section_start, section_end):
    gs_vector = classifier.getgsdata(section_start, section_end)

    aux_start,aux_end = classifier.section_split[section_start], classifier.section_split[section_end]

    my_rules_return_vector = []
    count = 0
    for sentdict in classifier.each_sentence.sentences:
        for i in range(0,len(sentdict['lemmas'])):
            word = sentdict['lemmas'][i]
            if isauxiliary(sentdict, i):
                count += 1
                if aux_start < count <= aux_end:
                    tree = NT.maketree(sentdict['tree'][0])
                    subtree_positions = NT.getsmallestsubtreepositions(tree)
                    if word in MODALS: my_rules_return_vector.append(truth(modalcheck(sentdict, i, tree, subtree_positions))) #Todo: I modified these b/c they were incorrectly written.
                    elif word in BE: my_rules_return_vector.append(truth(becheck(sentdict, i, tree, subtree_positions)))
                    elif word in HAVE: my_rules_return_vector.append(truth(havecheck(sentdict, i, tree, subtree_positions)))
                    elif word in DO: my_rules_return_vector.append(truth(docheck(sentdict, i, tree, subtree_positions)))
                    elif word in TO: my_rules_return_vector.append(truth(tocheck(sentdict, i, tree, subtree_positions)))
                    elif word in SO: my_rules_return_vector.append(truth(socheck(sentdict, i, tree, subtree_positions)))

    classifier.compare(gs_vector, my_rules_return_vector, section_start-1, verbose=False)

def main():
    print 'Start.'
    start_time = time.clock()
    data_is_in_files = False # ALWAYS MAKE THIS FALSE

    classifier = Classifier()
    classifier.importdata(0, 1, from_file=data_is_in_files)
    classifier.setsentmap()

    if not data_is_in_files:
        print '\nNumber of auxiliaries in data: %d'%len(classifier.gold_standard_auxs.aux_bools)
        print 'Number of missed sentences %d'%classifier.gold_standard_auxs.missed_sentences
        print 'number of true positives in data: %d\n'%classifier.gold_standard_auxs.number_of_positive_auxs()
        # make_all_the_files(classifier, appearance_threshold=1000, option2=True)
        print 'Length of each word near aux file:',len(extractdatafromfile(EACH_UNIQUE_WORD_NEAR_AUX))

    # aux_bools = classifier.gold_standard_auxs.aux_bools
    # for i in range(0, len(aux_bools)):
    #     if untruth(aux_bools[i]):
    #         classifier.each_sentence.printsentence(classifier.auxnum_to_sent_map[i])

    testmyrules(classifier,0,1)

    structural_set = ['pos','words','bigrams'] # 6 of these. (including 'combine_bigrams' and 'avg_vecs', and 'verb_locative')
    my_set = ['my_rules','my_features', 'combine_aux_type','square_rules'] # 4 of these. including 'combine_aux_type' & 'square_rules'
    
    features =  structural_set + my_set
    classifier.makefeaturevectors(features)

    oversamp = 5

    print 'Using these features: '
    print features
    def runmodels(train_start, train_end, test_start, test_end):
        
        print '\nSVM:'
        classifier.setclassifier(classifier='SVM', kernel='rbf')
        runmodel(classifier, train_start, train_end, test_start, test_end, oversample=oversamp)

        print '\nLogistic regression:'
        classifier.setclassifier(classifier='Regression')
        runmodel(classifier, train_start, train_end, test_start, test_end, oversample=oversamp)

        if not 'word2vec' in features:
            print '\nNaive Bayes:'
            classifier.setclassifier(classifier='NaiveBayes')
            runmodel(classifier, train_start, train_end, test_start, test_end, oversample=oversamp)

        print '\nLogistic Regression CV:'
        classifier.setclassifier(classifier='RegressionCV')
        runmodel(classifier, train_start, train_end, test_start, test_end, oversample=oversamp)
        
        print '\nDecision tree classifier:'
        classifier.setclassifier(classifier='Dtree')
        runmodel(classifier, train_start, train_end, test_start, test_end, oversample=oversamp)


    runmodels(-1,0,0,1)

    print 'Time taken: ',(time.clock()-start_time)

main()