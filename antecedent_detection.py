from truth import *
from old.detectVPE import *
from os import listdir
from old.vpesvm import NSentences
from scipy.sparse import csr_matrix as csr
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import nltktree as NT
import numpy as np
import alignment_vector_creation as AVC
import time
import copy

"""
@author Kian Kenyon-Dean
Purpose: to identify the antecedent of VP Ellipsis.
We assume that we are given an auxiliary that we know is a trigger for VPE.
"""

class Found(Exception): pass

class Trigger:
    def __init__(self, idx, sentnum, section, pattern=None):
        self.idx = idx
        self.sentnum = sentnum
        self.section = section
        self.pattern = pattern
        self.subtree = None
        self.context = None

    def get_idx(self): return self.idx
    def get_sentnum(self): return self.sentnum
    def get_subtree(self): return self.subtree
    def get_context(self): return self.context
    def get_section(self): return self.section
    def get_pattern(self): return self.pattern
    def get(self): return self.idx,self.sentnum

    # Just to copy the antecedent class.
    def get_words(self): return self.subtree.leaves()

    def set_subtree(self, subtree): self.subtree = subtree
    def set_context(self, context): self.context = context
    def set_pattern(self, pattern): self.pattern = pattern

    def equals(self, trigger2): return self.idx == trigger2.idx and self.sentnum == trigger2.sentnum

class Triggers:
    def __init__(self):
        self.vpe_sentences = []
        self.triggers = []

    def get_trigger(self, idx):
        try:
            return self.triggers[idx]
        except EOFError:
            print '\n!! Warning: no trigger at this index!!\n'
            return None

    def get_triggers(self): return self.triggers
    def get_triggers_sentnums_and_indexes(self):
        return [t.get_sentnum() for t in self.triggers],[t.get_idx() for t in self.triggers]
    def get_triggers_between_sections(self, start, end):
        ret = []
        for trig in self.triggers:
            if start <= trig.get_section() <= end:
                ret.append(trig)
        return ret

    def set_triggers(self, sentnums_with_vpe, trigger_idxs, section):
        # Both parameters passed to this function have the same length.
        for i in range(0, len(sentnums_with_vpe)):
            self.triggers.append(Trigger(trigger_idxs[i], sentnums_with_vpe[i], section))

    def add_trigger(self, idx, sentnum, section, pattern=None):
        self.triggers.append(Trigger(idx, sentnum, section, pattern=pattern))
    def remove_trigger(self, idx):
        new_trigs = []
        for t in range(0,len(self.triggers)):
            if t != idx:
                new_trigs.append(self.triggers[t])
        self.triggers = new_trigs

class Antecedent:
    def __init__(self, sentnum, subtree, trigger, words, section=-1):
        self.sentnum = sentnum
        self.subtree = subtree
        self.trigger = trigger
        self.words = words
        self.section = section
        self.context = None
        self.score = 0
        self.feature_vec_num = None

    def get_sentnum(self): return self.sentnum
    def get_subtree(self): return self.subtree
    def get_trigger(self): return self.trigger
    def get_words(self): return self.words
    def get_words_as_string(self):
        ret = ''
        for w in self.words: ret += w+' '
        return ret
    def get_section(self): return self.section
    def get_context(self): return self.context
    def get_score(self): return self.score
    def get_vec_idx(self): return self.feature_vec_num

    def set_context(self, context): self.context = context
    def set_score(self, score): self.score = score
    def set_section(self, section): self.section = section
    def set_vec_idx(self, idx): self.feature_vec_num = idx

    def equals(self, ant2):
        return self.sentnum == ant2.sentnum and self.trigger.equals(ant2.trigger) and self.words == ant2.words
    def exact_match(self, gold_ant): return self.equals(gold_ant)
    def head_match(self, gold_ant):
        try: return self.sentnum == gold_ant.sentnum and self.trigger.equals(gold_ant.trigger) and self.words[0] == gold_ant.words[0]
        except IndexError:
            print 'My ant: %s, gs ant: %s'%(self.get_words_as_string(), gold_ant.get_words_as_string())
    def head_overlap(self, gold_ant):
        try: return self.sentnum == gold_ant.sentnum and self.trigger.equals(gold_ant.trigger) and gold_ant.words[0] in self.words
        except IndexError:
            print 'GS index error wtf.'
            return False

class Antecedents:
    def __init__(self):
        self.antecedents = []

    def get_ant(self, idx): return self.antecedents[idx]
    def get_ants(self): return self.antecedents
    def get_ants_with_this_trigger(self, trigger):
        ret = []
        for ant in self.antecedents:
            if ant.get_trigger().equals(trigger):
                ret.append(ant)
        return ret
    def get_ants_between_sections(self, start, end):
        ret = []
        for ant in self.antecedents:
            if start <= ant.get_section() <= end:
                ret.append(ant)
        return ret

    def add_ant(self, sentnum, vp, trigger, words, section=-1):
        new_ant = Antecedent(sentnum, vp, trigger, words, section)
        for ant in self.antecedents:
            if new_ant.equals(ant):
                return
        self.antecedents.append(new_ant)

    def add(self, antecedent):
        self.antecedents.append(antecedent)
    def remove(self, ant):
        try: self.antecedents.remove(ant)
        except ValueError: return

class PPClassifier:
    classifier_names = ['LogReg','LogRegCV','DTree','Forest']

    def __init__(self):
        self.hyperplane = None
        self.name = ''
        self.feature_vectors = []
        self.classes = []

        self.ants_to_predict = Antecedents()
        self.vectors_to_predict = []
        self.gs_classes = []
        self.predictions = None

    def add_feature_vector(self, vec, classification):
        self.feature_vectors.append(vec)
        self.classes.append(classification)

    def add_prediction_vector(self, ant, vec, gs_class):
        self.ants_to_predict.add(ant)
        self.vectors_to_predict.append(vec)
        self.gs_classes.append(gs_class)

    def set_classifier(self, name, options=None):
        self.name = name
        if name == 'LogReg': self.hyperplane = LogisticRegression()
        elif name == 'LogRegCV': self.hyperplane = LogisticRegressionCV()
        elif name == 'SVM': self.hyperplane = SVC()
        elif name == 'DTree':
            if options: self.hyperplane = DecisionTreeClassifier(max_depth=10, min_samples_leaf=3)
            else: self.hyperplane = DecisionTreeClassifier()
        elif name == 'Forest':
            if options: self.hyperplane = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_leaf=3)
            else: self.hyperplane = RandomForestClassifier()

    def get_predictions(self): return self.predictions

    def train(self):
        print 'Post-process training...'
        self.hyperplane.fit(csr(self.feature_vectors), np.array(self.classes))

    def predict(self, probabilty=None):
        print 'Post-process predicting...'
        if probabilty: self.predictions = self.hyperplane.predict_proba(csr(self.vectors_to_predict))
        else: self.predictions = self.hyperplane.predict(csr(self.vectors_to_predict))

    def return_predictions(self):
        choices = []
        for i in range(0,len(self.predictions)):
            if self.predictions[i] == 1:
                choices.append(self.ants_to_predict.get_ant(i))
        return choices

    def evaluate(self):
        tp,fp,fn = 0.0,0.0,0.0

        for i in range(0,len(self.predictions)):
            if self.predictions[i] == 1 and self.gs_classes[i] == 1: tp += 1
            elif self.predictions[i] == 1 and self.gs_classes[i] == 0: fp += 1
            elif self.predictions[i] == 0 and self.gs_classes[i] == 1: fn += 1

        if tp == 0.0: tp+=1

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)

        print 'Classifier %s for post-processing choice:'%self.name
        print 'TP: %d, FP: %d, FN: %d.'%(tp,fp,fn)
        print 'Precision: %0.2f, Recall: %0.2f, F1: %0.2f.\n'%(precision,recall,f1)

    def evaluate_with_choice(self, choice_sequences):
        ant_choices = []

        i,choice_idx = 0,0

        indexes = []
        prev = 0
        for i in range(0,len(choice_sequences)):
            crt_best_ant = self.ants_to_predict.get_ant(choice_sequences[i]-1)
            crt_best_prob = 0.0
            crt_best_idx = choice_sequences[i]-1

            for j in range(prev, choice_sequences[i]):
                if self.predictions[j][1] > crt_best_prob:
                    crt_best_ant = self.ants_to_predict.get_ant(j)
                    crt_best_prob = self.predictions[j][1]
                    crt_best_idx = j

            ant_choices.append(crt_best_ant)
            indexes.append(crt_best_idx)

            prev = choice_sequences[i]

        new_predictions = []
        new_gs = []
        for j in range(0,len(self.predictions)):
            if j in indexes:
                new_predictions.append(1)
                new_gs.append(self.gs_classes[j])
        self.predictions = new_predictions
        self.gs_classes = new_gs

        self.evaluate()
        return ant_choices

class AntecedentClassifier:

    classifier_names = ['LogReg','DTree','Forest','SVM']
    best_score = 100

    exact_match = 3
    head_match = 2
    head_overlap = 1

    def __init__(self):
        self.sentences = NSentences()
        self.triggers = Triggers()
        self.antecedents = Antecedents()
        self.goldstandard_antecedents = Antecedents()

        self.feature_vectors = []
        self.gold_standard = []
        self.balanced_features = []
        self.balanced_gs = []

        self.get_bos_trigger_patterns = False

        self.classifier_name = ''
        self.hyperplane = None
        self.section_split = {-1:0}
        self.pp_classifier = PPClassifier()

        self.cheated_antecedents = 0
        self.missed = 0
        self.ants_deleted = 0

    #### Utility methods ####
    def get_triggers(self):
        return self.triggers.get_triggers()
    def get_sentence(self, i):
        try: return self.sentences.get_sentence(i)
        except IndexError: return None
    def add_sentences(self, mrgmatrix): self.sentences.addsentences(mrgmatrix)
    def print_sent(self, i):
        try: self.sentences.printsentence(i)
        except IndexError: print 'NO SENTENCE AT THIS INDEX\n'
    #### --------------- ####

    def importdata(self, start_section, end_section):
        dnum = 0
        prev_total_sentences = 0

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
                    if filehasvpe(test_file, annotationmatrix):
                        mrgmatrix = getdataMRG(test_file,XML_MRG+subdir)
                        self.add_sentences(mrgmatrix)

                        gs_dict,gs_triggers = goldstandardVPEsentences(test_file, XML_RAW_TOKENIZED, subdir, annotationmatrix, mrgmatrix, get_words_as_sections=False, get_trig_patterns=self.get_bos_trigger_patterns)
                        gs_sent_list = [k for k in gs_dict]

                        triggers_added = Triggers()

                        for k in range(0,len(gs_sent_list)):
                            gs_sent_list.sort()
                            sentnum = gs_sent_list[k]

                            if not self.get_bos_trigger_patterns:
                                idx,trigger = gs_triggers[k]
                                pattern = None
                            else:
                                idx,trigger,pattern = gs_triggers[k]

                            def add_trigger_from_gs_list(sentnum, idx, trigger_word, pattern=None):
                                # print 'Checking trigger: ',trigger_word,
                                sentence = mrgmatrix[sentnum]['words']
                                # print sentence
                                # Easy case.
                                try:
                                    if sentence[idx] == trigger_word:
                                        triggers_added.add_trigger(idx, sentnum+prev_total_sentences, dnum, pattern=pattern)
                                        # print '- Got it!'
                                        return True
                                except IndexError:
                                    pass

                                # Easier case.
                                count,last_aux_idx = 0,0
                                for i in range(0,len(sentence)):
                                    if sentence[i] == trigger_word:
                                        last_aux_idx = i
                                        count += 1

                                if count == 1:
                                    # print '- Got it!'
                                    # print 'Got this sentence with this trigger:',trigger
                                    # print sentence
                                    triggers_added.add_trigger(last_aux_idx, sentnum+prev_total_sentences, dnum, pattern=pattern)
                                    return True

                                else:
                                    # if sentnum in gs_sent_list: print gs_sent_list,gs_triggers
                                    # print 'Checking next sentence for this trigger: %s'%trigger_word
                                    # try:
                                    #     if add_trigger_from_gs_list(sentnum+1, idx, trigger_word):
                                    #         print '- Got it with last rule!'
                                    #         print sentence
                                    # except IndexError:
                                    print 'Missed a trigger! :('
                                    self.missed += 1
                                    return False

                            add_trigger_from_gs_list(sentnum, idx, trigger, pattern=pattern)

                        # Below is for getting the gold standard antecedents.
                        ant_sentnums,ant_words = import_goldstandard_ants(test_file, XML_RAW_TOKENIZED, subdir, annotationmatrix, mrgmatrix, prev_total_sentences)

                        # These equalize the num triggs and ants, sometimes it fucks up.
                        if len(triggers_added.get_triggers()) < len(ant_sentnums):
                            print 'Less triggers than ants, deleting ants from %d to %d'%(len(ant_sentnums), len(triggers_added.get_triggers()))
                            ant_sentnums = ant_sentnums[0:len(triggers_added.get_triggers())]
                        else:
                            while len(ant_sentnums) < len(triggers_added.get_triggers()):
                                triggers_added.remove_trigger(len(triggers_added.get_triggers())-1)
                                self.missed += 1

                        for i in range(0,len(ant_sentnums)):
                            t = NT.maketree(mrgmatrix[ant_sentnums[i]-prev_total_sentences]['tree'][0])

                            self.goldstandard_antecedents.add_ant(ant_sentnums[i], NT.lowest_common_subtree(t, ant_words[i]), triggers_added.get_trigger(i), ant_words[i], section=dnum)
                            idx,trig_sentnum = triggers_added.get_trigger(i).get()
                            self.triggers.add_trigger(idx,trig_sentnum, dnum, pattern=triggers_added.get_trigger(i).get_pattern())

                        prev_total_sentences += len(mrgmatrix)
                self.section_split[dnum] = prev_total_sentences
            dnum += 1

    def set_classifer(self, name, options=False):
        self.hyperplane = None
        if name == 'SVM': self.hyperplane,self.classifier_name = SVC(),'SVM'
        elif name == 'LogReg': self.hyperplane,self.classifier_name = LogisticRegression(),'Logistic Regression'
        elif name == 'LogRegCV': self.hyperplane,self.classifier_name = LogisticRegressionCV(),'Logistic Regression CV'
        elif name == 'DTree':
            if options: self.hyperplane,self.classifier_name = DecisionTreeClassifier(max_depth=10, min_samples_leaf=3),'Decision Tree'
            else: self.hyperplane,self.classifier_name = DecisionTreeClassifier(),'Decision Tree'
        elif name == 'NaiveBayes': self.hyperplane,self.classifier_name = MultinomialNB(),'Naive Bayes'
        elif name == 'Forest':
            if options: self.hyperplane,self.classifier_name = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_leaf=3),'Random Forest'
            else: self.hyperplane,self.classifier_name = RandomForestClassifier(),'Random Forest'

    # Sets the possible antecedents. Each antecedent has a corresponding trigger associated with it.
    def set_ants_and_triggers(self, verbose=False):
        for trigger in self.get_triggers():
            trigger_idx,trigger_sentnum = trigger.get()

            for sentnum in range( max(0,trigger_sentnum - SENTENCE_SEARCH_DISTANCE), trigger_sentnum+1 ): # Include the aux's sentence.
                tree = NT.maketree(self.get_sentence(sentnum)['tree'][0])

                # Setting the trigger.
                if sentnum == trigger_sentnum:
                    vp = NT.get_nearest_vp(tree, trigger_idx)
                    trigger.set_subtree(vp)
                    trigger.set_context(vp.parent())
                """
                 Setting the VPs as possible antecedents. This looks at each VP & its leaves. VP=[w1, w2, ... wn]
                 It adds antecedents that all share the same VP, but linearly changes the words that are added
                 such that the first antecedent's words are just [w1], next is [w1, w2], etc. It stops once we hit the trigger.
                """
                for position in NT.phrase_positions_in_tree(tree, 'VP'):
                    vp_words = []
                    for w in tree[position].leaves():
                        if w == self.get_sentence(trigger_sentnum)['words'][trigger_idx]: break
                        vp_words.append(w)
                        self.antecedents.add_ant(sentnum, tree[position], trigger, copy.copy(vp_words))

                # This is too complicated...
                """
                for subtree in NT.get_linear_phrase_combinations(tree, 'VP'):
                    self.antecedents.add_ant(sentnum, subtree, trigger, subtree.leaves())
                """

                for position in NT.phrase_positions_in_tree(tree, 'predicative'):
                    self.antecedents.add_ant(sentnum, tree[position], trigger, tree[position].leaves())

                # Setting any single adjectives as possible antecedents.
                for subtree in NT.getsmallestsubtrees(tree):
                    if isadj(subtree.label()):
                        self.antecedents.add_ant(sentnum, subtree, trigger, subtree.leaves())


            dont_cheat = False
            for my_ant in self.antecedents.get_ants_with_this_trigger(trigger):
                if self.goldstandard_antecedents.get_ants_with_this_trigger(trigger)[0].equals(my_ant):
                    dont_cheat = True
                    break

            if not dont_cheat:
                if verbose:
                    print '\nCheating:'
                    print self.goldstandard_antecedents.get_ants_with_this_trigger(trigger)[0].get_words()
                    print self.goldstandard_antecedents.get_ants_with_this_trigger(trigger)[0].get_subtree()
                # self.antecedents.add(self.goldstandard_antecedents.get_ants_with_this_trigger(trigger)[0])
                self.cheated_antecedents += 1

    # This method will set the context around antecedents and triggers. This is fundamental for the alignment algorithm.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def set_contexts(self, verbose=False):
        def set_ant_context(ant, before=True, verbose=verbose):
            if verbose:
                print '\nTesting antecedent/trigger:',
                print ant.get_words()
                self.print_sent(ant.get_sentnum())

            context_idxs = []

            tree = nltktree.maketree(self.get_sentence(ant.get_sentnum())['tree'][0])
            sentdict = self.get_sentence(ant.get_sentnum())
            head_idx = get_antecedent_head_index(sentdict, ant)

            # First we get what's behind the antecedent.
            # Here we will get the NP that is closest to the antecedent.

            noun_phrase_idxs = (-1,-1)
            search_range = reversed(range(0, head_idx))
            if not before: search_range = range(head_idx, len(sentdict['words']))
            for i in search_range:
                word = sentdict['words'][i]
                tag  = sentdict['pos'][i]

                if ispunctuation(tag): break

                context_idxs.append(i)

                # This is for getting all the words in a noun phrase.
                if not isnoun(tag): continue
                else:
                    nphrase = NT.get_nearest_phrase(tree, i, ['NP','NP-PRD'])
                    if nphrase != None:
                        leaves = nphrase.leaves()
                        for k in range(0,len(leaves)):
                            if leaves[k] == word:
                                noun_phrase_idxs = (i-k,i+(len(leaves)-k))
                                break
                            context_idxs.append(context_idxs[-1]-1)
                    else:
                        noun_phrase_idxs = (i,i+1)
                    break

            context_idxs.sort()

            # We are returning a smaller sentdict.
            context_sentdict = {'words':[], 'lemmas':[], 'pos':[], 'np':None}

            try:
                for i in range(context_idxs[0], context_idxs[-1]+1):
                    context_sentdict['words'].append(sentdict['words'][i])
                    context_sentdict['lemmas'].append(sentdict['lemmas'][i])
                    context_sentdict['pos'].append(sentdict['pos'][i])
            except IndexError:
                pass
            if noun_phrase_idxs != (-1,-1) and noun_phrase_idxs != (1, len(sentdict['words'])):
                context_sentdict['np'] = noun_phrase_idxs
            else: context_sentdict['np'] = (-1,-1)


            if noun_phrase_idxs != (-1,-1) and noun_phrase_idxs != (1, len(sentdict['words'])):
                if verbose:
                    print noun_phrase_idxs
                    print (0, len(sentdict['words'])-1)
                    print 'NP:',
                    for i in range(noun_phrase_idxs[0], noun_phrase_idxs[1]):
                        print sentdict['words'][i],
                    print
                ant.set_context(context_sentdict)

            elif before:
                set_ant_context(ant, before=False)

                if not isinstance(ant.get_context(), dict):
                    pseudo_context_distance = len(ant.get_subtree().leaves())
                    if pseudo_context_distance <= 2: pseudo_context_distance = MINIMUM_CONTEXT_DISTANCE
                    try:
                        for i in range(0, pseudo_context_distance):
                            context_sentdict['words'].append(sentdict['words'][head_idx+i])
                            context_sentdict['lemmas'].append(sentdict['lemmas'][head_idx+i])
                            context_sentdict['pos'].append(sentdict['pos'][head_idx+i])
                    except IndexError:
                        pass
                    context_sentdict['np'] = (-1,-1)
                    ant.set_context(context_sentdict)

        for ant in self.goldstandard_antecedents.get_ants():
            set_ant_context(ant)
        for ant in self.antecedents.get_ants():
            set_ant_context(ant)
        for trig in self.get_triggers():
            set_ant_context(trig)

    # This returns whether or not we have an exact, head, or overlap match.
    def in_gold_standard(self, ant, trig):
        for gold_ant in self.goldstandard_antecedents.get_ants_with_this_trigger(trig):
            if ant.exact_match(gold_ant): return self.exact_match
            if ant.head_match(gold_ant): return self.head_match
            if ant.head_overlap(gold_ant): return self.head_overlap #TODO: CHANGE BACK MAYBE - and self.bos_spenader_ant_fscore(ant,gold_ant) > 0.5
        return 0

    def corresponding_gs_ant(self, ant):
        return self.goldstandard_antecedents.get_ants_with_this_trigger(ant.get_trigger())[0]

    # Make the feature vectors.
    def make_feature_vectors(self):
        POS_TAGS = extractdatafromfile(EACH_UNIQUE_POS_FILE)
        AUX_WORDS = extractdatafromfile(UNIQUE_AUXILIARIES_FILE)

        patterns = []
        if self.get_bos_trigger_patterns:
            for trig in self.get_triggers():
                patterns.append(trig.get_pattern())

        if self.get_bos_trigger_patterns: unique_patterns = list(set(patterns))
        else: unique_patterns = None

        print 'Loading the Word2vec vectors...'
        word2vec_dict = loadword2vecs()

        first = True
        length = 0
        for trig in self.get_triggers():
            for ant in self.antecedents.get_ants_with_this_trigger(trig):
                ant.set_vec_idx(len(self.feature_vectors))

                vec = AVC.make_alignment_vector(self.get_sentence(trig.get_sentnum()), self.get_sentence(ant.get_sentnum()), ant, trig, POS_TAGS, AUX_WORDS, word2vec_dict, unique_patterns)
                vec = [float(v) for v in vec]

                # vec = [self.in_gold_standard(ant,trig)]
                if not first and len(vec) != length:
                    raise Exception('Feature vectors are not same length! %d, %d'%(length, len(vec)))

                self.feature_vectors.append(vec)
                self.gold_standard.append(self.in_gold_standard(ant, trig))

                if first:
                    print 'Length of feature vectors: %d'%len(self.feature_vectors[0])
                    first = False
                    length = len(self.feature_vectors[0])

                if len(self.feature_vectors)%2500 == 0:
                    print 'Making feature vector %d'%len(self.feature_vectors)

    def balance_data(self, start_section, end_section):
        print 'Balancing the data set...'

        new_feature_vectors,new_gs_bools = [],[]

        class_frequencys = [0,0,0,0]
        gs = self.get_gs_data(start_section, end_section)
        feature_vectors = self.get_feature_vectors(start_section, end_section)
        for i in range(0,len(gs)): class_frequencys[gs[i]] += 1

        biggest_class = 0
        for i in range(0,len(class_frequencys)):
            if class_frequencys[i] > class_frequencys[biggest_class]:
                biggest_class = i

        for i in range(0,len(class_frequencys)):
            multiplier = 1
            while multiplier*class_frequencys[i] < class_frequencys[biggest_class]: multiplier+=1
            for j in range(0, len(gs)):
                if gs[j] == i:
                    count = 0
                    while count < multiplier:
                        new_feature_vectors.append(feature_vectors[j])
                        new_gs_bools.append(gs[j])
                        count += 1

        self.balanced_features,self.balanced_gs = new_feature_vectors,new_gs_bools

    def get_feature_vectors(self, start, end):
        ret = []
        count = 0
        for trig in self.get_triggers():
            if start <= trig.get_section() <= end:
                for ant in self.antecedents.get_ants_with_this_trigger(trig):
                    ret.append(self.feature_vectors[count])
                    count += 1
            elif trig.get_section() > end: break
            else:
                for ant in self.antecedents.get_ants_with_this_trigger(trig):
                    count += 1
        return ret

    def get_gs_data(self, start, end):
        ret = []
        count = 0
        for trig in self.get_triggers():
            if start <= trig.get_section() <= end:
                for ant in self.antecedents.get_ants_with_this_trigger(trig):
                    ret.append(self.gold_standard[count])
                    count += 1
            elif trig.get_section() > end: break
            else:
                for ant in self.antecedents.get_ants_with_this_trigger(trig):
                    count += 1
        return ret

    def train(self, feature_vecs, gs_data):
        print 'Fitting the hyperplane...'
        X_matrix = csr(feature_vecs)
        y_vector = np.array(gs_data)
        self.hyperplane.fit(X_matrix, y_vector)

    def devtest(self, start_train, end_train, start_section, end_section, pp_classifier_name, options=None, verbose=False, probability=None, log_start=None):
        print 'Predicting...'

        if probability != None: predictions = self.hyperplane.predict_proba(csr(self.get_feature_vectors(start_section, end_section)))
        else: predictions = self.hyperplane.predict(csr(self.get_feature_vectors(start_section, end_section)))

        if probability != None:
            total_gs_ants = len(self.goldstandard_antecedents.get_ants_between_sections(start_section, end_section))

            if pp_classifier_name != 'dont':
                self.pp_classifier = PPClassifier()
                chosen_ants = self.pp_classify(start_train, end_train, start_section, end_section, predictions, pp_classifier_name, probability, options=options, pp_discrimination='all')
                self.prob_evaluation(chosen_ants, total_gs_ants, verbose=verbose, log_start=log_start)

                self.pp_classifier = PPClassifier()
                chosen_ants = self.pp_classify(start_train, end_train, start_section, end_section, predictions, pp_classifier_name, probability, options=options, pp_discrimination='everything')
                self.prob_evaluation(chosen_ants, total_gs_ants, verbose=verbose, log_start=log_start)
            else:
                chosen_ants = self.post_process(predictions, start_section, end_section, probability, test='exact')
                self.prob_evaluation(chosen_ants, total_gs_ants, verbose=verbose, log_start=log_start)

                chosen_ants = self.post_process(predictions, start_section, end_section, probability, test='average')
                self.prob_evaluation(chosen_ants, total_gs_ants, verbose=verbose, log_start=log_start)

                chosen_ants = self.post_process(predictions, start_section, end_section, probability, test='all')
                self.prob_evaluation(chosen_ants, total_gs_ants, verbose=verbose, log_start=log_start)
        else:
            self.evaluation(self.get_gs_data(start_section, end_section), predictions)

    def pp_classify(self, start_train, end_train, start_test, end_test, my_alg, classifier_name, probability, options=None, pp_discrimination=None):
        for trig in self.get_triggers():
            if start_train <= trig.get_section() <= end_train:
                added_ho, added_hm, added_em, added_miss, added_miss = 0,False,False,False,0

                for ant in self.antecedents.get_ants_with_this_trigger(trig):
                    crt_vec = self.feature_vectors[ant.get_vec_idx()]
                    crt_score = self.gold_standard[ant.get_vec_idx()]

                    if added_ho<1 and crt_score == self.head_overlap:
                        self.pp_classifier.add_feature_vector(crt_vec, 0)
                        added_ho += 1

                    elif not added_hm and crt_score == self.head_match:
                        self.pp_classifier.add_feature_vector(crt_vec, 0)
                        added_hm = True

                    elif not added_em and crt_score == self.exact_match:
                        self.pp_classifier.add_feature_vector(crt_vec, 1)
                        added_em = True

                    elif added_miss<3 and crt_score == 0:
                        self.pp_classifier.add_feature_vector(crt_vec, 0)
                        added_miss += 1

        choices = []
        processed_ants = self.post_process(my_alg, start_test, end_test, probability, test=pp_discrimination)

        if classifier_name == 'oracle':
            print 'ORACLE!'
            for lst in processed_ants:
                best_ant = lst[0]
                for ant in lst:
                    ant.set_score(self.in_gold_standard(ant, ant.get_trigger()))
                    if ant.get_score() > best_ant.get_score():
                        best_ant = ant
                choices.append(best_ant)
            return choices

        self.pp_classifier.set_classifier(classifier_name, options=options)
        self.pp_classifier.train()

        print 'Length of PPed ants:',len(processed_ants)
        count = 0
        for lst in processed_ants:
            for ant in set(lst):
                gs_class = self.gold_standard[ant.get_vec_idx()]
                crt_vec = self.feature_vectors[ant.get_vec_idx()]
                if gs_class > 0:
                    self.pp_classifier.add_prediction_vector(ant, crt_vec, 1)
                else:
                    self.pp_classifier.add_prediction_vector(ant, crt_vec, 0)
                count += 1
            choices.append(count)

        self.pp_classifier.predict(probabilty=probability)
        if probability != None:
            self.pp_classifier.evaluate_with_choice(choices)
        else:
            self.pp_classifier.evaluate()

        print 'Length of the PP-classifiers predictions:',len(self.pp_classifier.return_predictions())
        return self.pp_classifier.return_predictions()

    # Takes the predictions that my algorithm created.
    def post_process(self, my_alg, start_test, end_test, probability, test=None):
        print '\nPost-processing with test as:',test
        chosen_ants = []
        i,t = 0,0

        section_triggers = [ant.get_trigger() for ant in self.goldstandard_antecedents.get_ants_between_sections(start_test, end_test)]
        while i < len(my_alg) and t < len(section_triggers):
            crt_trigger = section_triggers[t]

            max_overlap, max_head_match, max_exact_match, max_mean = 0.0,0.0,0.0,0.0

            if test == 'everything':
                chosen_ants.append([ant for ant in self.antecedents.get_ants_with_this_trigger(crt_trigger)])
                t += 1
                continue

            first = True
            for ant in self.antecedents.get_ants_with_this_trigger(crt_trigger):
                if my_alg[i][self.head_overlap] > max_overlap and my_alg[i][self.head_overlap] > probability or first:
                    best_overlap_ant = ant
                    max_overlap = my_alg[i][self.head_overlap]

                if my_alg[i][self.head_match] > max_head_match and my_alg[i][self.head_match]*HEAD_MATCH_MODIFIER > probability or first:
                    best_head_match_ant = ant
                    max_head_match = my_alg[i][self.head_match]

                if my_alg[i][self.exact_match] > max_exact_match and my_alg[i][self.exact_match]*EXACT_MATCH_MODIFIER > probability or first:
                    best_exact_match_ant = ant
                    max_exact_match = my_alg[i][self.exact_match]

                if test=='weighted':
                    crt_mean = np.mean([my_alg[i][self.head_overlap], HEAD_MATCH_MODIFIER*my_alg[i][self.head_match], EXACT_MATCH_MODIFIER*my_alg[i][self.exact_match]])
                else:
                    crt_mean = np.mean([my_alg[i][self.head_overlap:self.exact_match+1]])

                if crt_mean > max_mean or first: max_mean, best_average_ant = crt_mean, ant

                if i+1 >= len(my_alg): break
                i += 1
                first = False

            # Now we have to compare the antecedents that we got.
            best_ant_list = [best_overlap_ant, best_head_match_ant, best_exact_match_ant, best_average_ant]
            equality_list = [1,1,1,1]

            if test=='overlap': chosen_ants.append(best_overlap_ant)
            elif test=='head': chosen_ants.append(best_head_match_ant)
            elif test=='exact': chosen_ants.append(best_exact_match_ant)
            elif test=='average': chosen_ants.append(best_average_ant)
            elif test=='all': chosen_ants.append(best_ant_list)
            elif test=='weighted': chosen_ants.append(best_average_ant)
            else:
                try:
                    for ii in range(0,len(best_ant_list)):
                        for j in range(0,len(best_ant_list)):
                            if ii != j:
                                if best_ant_list[ii].equals(best_ant_list[j]):
                                    equality_list[ii] += 1
                                    # If there is agreement on 3 or more of the best antecedents...
                                    if equality_list[ii] >= 3:
                                        chosen_ants.append(best_ant_list[ii])
                                        raise Found
                except Found: pass

                max_head_match = max_head_match * HEAD_MATCH_MODIFIER
                max_exact_match = max_exact_match * EXACT_MATCH_MODIFIER

                if equality_list == [2,2,2,2] or equality_list == [1,1,1,1]:
                    best_value = max(max_overlap, max_head_match, max_exact_match, max_mean)
                    if best_value == max_overlap: chosen_ants.append(best_overlap_ant)
                    elif best_value == max_head_match: chosen_ants.append(best_head_match_ant)
                    elif best_value == max_exact_match: chosen_ants.append(best_exact_match_ant)
                    elif best_value == max_mean: chosen_ants.append(best_average_ant)
                else:
                    for ii in range(0,len(equality_list)):
                        if equality_list[ii] == 2:
                            chosen_ants.append(best_ant_list[ii])
                            break

            t += 1

        return chosen_ants

    def decision_tree_post_process(self, my_alg, start_test, end_test, probability_threshold):
        print 'DTree Post-processing...'
        chosen_ants = []

        i,t = 0,0
        section_triggers = [ant.get_trigger() for ant in self.goldstandard_antecedents.get_ants_between_sections(start_test, end_test)]

        while i < len(my_alg) and t < len(section_triggers):
            crt_trigger = section_triggers[t]

            overlaps, head_matches, exact_matches = [],[],[]

            first_ant = None
            for ant in self.antecedents.get_ants_with_this_trigger(crt_trigger):
                if not first_ant: first_ant = ant

                if my_alg[i][self.head_overlap] > probability_threshold: overlaps.append(ant)
                if my_alg[i][self.head_match] > probability_threshold: head_matches.append(ant)
                if my_alg[i][self.exact_match] > probability_threshold: exact_matches.append(ant)

                if i+1 >= len(my_alg): break
                i += 1

            if len(overlaps+head_matches+exact_matches) == 0:
                overlaps.append(first_ant)

            chosen_ants.append([ant for ant in overlaps+head_matches+exact_matches])
            t += 1
        return chosen_ants

    # In this evaluation of probabilities, the predictions will be 4-dimensional:
    # [p(not_antecedent), p(head_overlap), p(head_match), p(exact_match)]
    def prob_evaluation(self, chosen_ants, total_gs_ants, verbose=False, log_start=None):
        results = [0.0,0.0,0.0,0.0] # miss, overlap, head match, exact match
        false_positives = 0.0
        true_postives = 0.0

        maxx = 1000
        for f in listdir(ANT_LOG_LOCATIONS):
            if f.startswith(log_start):
                file_number = int(f[len(log_start):len(log_start)+4])
                if file_number > maxx: maxx = file_number

        filename = log_start+str(maxx+1)+'.log'

        newf = open(ANT_LOG_LOCATIONS+filename,'w')
        newf.write('%s\n'%self.classifier_name)

        list_check = False
        for val in chosen_ants:
            if isinstance(val, list):
                list_check = True
                break

        if list_check:
            print 'Testing over lists!'
            for lst in chosen_ants:
                best_ant,list_best_score = None,0
                for ant in lst:
                    ant.set_score(self.in_gold_standard(ant,ant.get_trigger()))
                    if ant.get_score() > list_best_score:
                        best_ant = ant
                        list_best_score = ant.get_score()
                    if ant.get_score() == 0: false_positives += 1
                    else: true_postives += 1

                if best_ant != None: results[best_ant.get_score()] += 1
                else: results[0] += 1

                if best_ant:
                    if verbose:
                        print '\nTrigger sentence:'
                        self.print_sent(best_ant.get_trigger().get_sentnum())
                        print 'Selected best_antecedent, score = %d: %s'%(best_ant.get_score(),best_ant.get_words_as_string())

                    newf.write('\nTrigger sentence:\n')
                    for w in self.get_sentence(best_ant.get_trigger().get_sentnum())['words']: newf.write(w+' ')
                    newf.write('\nSelected best_antecedent, score = %d: %s\n\n'%(best_ant.get_score(),best_ant.get_words_as_string()))
                else:
                    if verbose:
                        print 'No antecedent was selected.'
        else:
            for ant in chosen_ants:
                ant.set_score(self.in_gold_standard(ant,ant.get_trigger()))
                results[ant.get_score()] += 1

                if verbose:
                    print '\nTrigger sentence:'
                    self.print_sent(ant.get_trigger().get_sentnum())
                    print 'Selected antecedent, score = %d: %s'%(ant.get_score(),ant.get_words_as_string())

                newf.write('\nTrigger sentence:\n')
                for w in self.get_sentence(ant.get_trigger().get_sentnum())['words']: newf.write(w+' ')
                newf.write('\nSelected antecedent, score = %d: %s\n\n'%(ant.get_score(),ant.get_words_as_string()))

        gold_exact_match = total_gs_ants
        exact_match = results[self.exact_match]
        head_match = exact_match+results[self.head_match]
        head_overlap = head_match+results[self.head_overlap]
        misses = results[0]

        e = 'Got %d exact matches. Out of %d, thats %0.2f percent.\n'%(exact_match, gold_exact_match,(exact_match/gold_exact_match))
        h = 'Got %d head matches. Out of %d, thats %0.2f percent.\n'%(head_match, gold_exact_match,(head_match/gold_exact_match))
        o = 'Got %d head overlap matches. Out of %d, thats %0.2f percent.\n'%(head_overlap, gold_exact_match,(head_overlap/gold_exact_match))
        m = '%d misses. Percentage: %0.2f.\n'%(misses, misses/gold_exact_match)
        try: fp = '%d false positives. Precision = %0.2f.\n'%(false_positives,true_postives/(true_postives+false_positives))
        except ZeroDivisionError: fp = '%d false positives. Precision = 0.\n'%false_positives

        bos_spen_fscore = self.bos_spenader_evaluation(chosen_ants)
        f1 = 'Got a %0.2f F1 score using B&S evaluation criterion.'%bos_spen_fscore

        newf.write(e+h+o+m+fp+f1)
        newf.close()
        print e,h,o,m,fp,f1

    def bos_spenader_ant_fscore(self, chosen_ant, gs_ant):
        correctly_identified_tokens = 0.0

        chosen_words = copy.copy(chosen_ant.get_words())
        correct_words = copy.copy(gs_ant.get_words())

        for w in copy.copy(chosen_words):
            if w in correct_words:
                correctly_identified_tokens += 1.0
                chosen_words.remove(w)
                correct_words.remove(w)
        if not correctly_identified_tokens == 0.0:
            precision = correctly_identified_tokens/len(chosen_ant.get_words())
            recall = correctly_identified_tokens/len(gs_ant.get_words())
            return (2.0*precision*recall)/(precision+recall)
        else:
            return 0.0

    def bos_spenader_evaluation(self, chosen_ants):
        scores = []
        for val in chosen_ants:
            if isinstance(val,list):
                for ant in val:
                    scores.append(self.bos_spenader_ant_fscore(ant, self.corresponding_gs_ant(ant)))
            else:
                scores.append(self.bos_spenader_ant_fscore(val, self.corresponding_gs_ant(val)))
        total_f1 = np.mean(scores)
        return total_f1

    def evaluation(self, gs, my_alg, verbose=False):
        exact_match,head_match,head_overlap = 0.0,0.0,0.0
        false_positives = 0.0
        gold_exact_match = 0.0
        for i in range(0,len(my_alg)):
            if gs[i] == self.exact_match: gold_exact_match+=1

            if my_alg[i] == self.exact_match and gs[i] == self.exact_match: exact_match+=1
            if my_alg[i] == self.head_match and gs[i] == self.exact_match: head_match+=1
            if my_alg[i] == self.head_overlap and gs[i] == self.exact_match: head_overlap+=1

            if my_alg[i] >= 1 and gs[i] == 0: false_positives += 1

        head_match += exact_match
        head_overlap += head_match

        print 'Got %d exact matches. Out of %d, thats %0.2f percent.'%(exact_match, gold_exact_match,(exact_match/gold_exact_match))
        print 'Got %d head matches. Out of %d, thats %0.2f percent.'%(head_match, gold_exact_match,(head_match/gold_exact_match))
        print 'Got %d head overlap matches. Out of %d, thats %0.2f percent.'%(head_overlap, gold_exact_match,(head_overlap/gold_exact_match))
        if head_overlap+false_positives == 0: print 'Got %d false positives'%false_positives
        else: print 'Got %d false positives. Precision: %0.2f'%(false_positives, head_overlap/(head_overlap+false_positives))

    def print_gs(self):
        for ant in self.goldstandard_antecedents.get_ants():
            if ant.words[0] == 'works':
                print '-----'
                trig = ant.get_trigger()
                print 'Section:',ant.get_section()
                self.print_sent(trig.get_sentnum())
                self.print_sent(ant.get_sentnum())
                print ant.words

    def analyze_gs(self, verbose=False):
        perfect_match,vps,total,sentenial = 0.0,0.0,0.0,0.0

        types_of_ants = []

        aux_starters = 0.0
        for ant in self.goldstandard_antecedents.antecedents:
            vp = ant.get_subtree()
            types_of_ants.append(vp.label())
            if len(vp.leaves()) == len(ant.get_words()): perfect_match += 1.0
            elif verbose and False:
                print '-----BAD CONSTITUENT MATCH-----'
                self.print_sent(ant.get_sentnum())
                print ant.get_words()
                print vp
            if vp.label() == 'VP': vps += 1.0
            elif verbose:
                print '-----NOT A VP-----'
                self.print_sent(ant.get_sentnum())
                print ant.get_words()
                print vp
            total += 1.0

            if ant.get_sentnum() == ant.get_trigger().get_sentnum():
                sentenial += 1.0

            sentdict = self.get_sentence(ant.get_sentnum())
            ant_idx = get_antecedent_head_index(sentdict, ant)
            if isauxiliary(sentdict, ant_idx):
                aux_starters += 1

        d = {}
        for ant_type in set(types_of_ants):
            d[ant_type] = types_of_ants.count(ant_type)

        print '\nThe different possible antecedent labels: '
        for k in d:
            print k,d[k],' ',
        print
        print 'Percentage of perfect matches from GS to constituent: %0.2f'%(perfect_match/total)
        print 'Percentage of VP antecedents: %0.2f'%(vps/total)
        print 'Total number of antecedents: %d'%total
        print 'Number of sentenial antecedents: %d. That is, %0.2f percent of the data.'%(sentenial,sentenial/total)
        print 'Number of antecedents that start with an auxiliary: %d. That is, %0.2f percent of the data'%(aux_starters,aux_starters/total)

        # Ends up giving me: 61% perfect match constituents (may not be VPs), 74% of the antecedents are VPs (may not be perfect matches)
        # but there are about 20 different possible labels for the antecedents.

def runmodel(classifier, name, train_start, train_end, test_start, test_end, pp_classifier_name ,options=False, oversample=1, verbose=False, probability=None, log_start=None):
    classifier.set_classifer(name, options=options)
    if oversample > 1: classifier.train(classifier.balanced_features, classifier.balanced_gs)
    else: classifier.train(classifier.get_feature_vectors(train_start,train_end), classifier.get_gs_data(train_start, train_end))

    classifier.devtest(train_start, train_end, test_start, test_end, pp_classifier_name, options=options, verbose=verbose, probability=probability, log_start=log_start)

def main():
    print 'GOOOOKU'
    train_start,train_end,test_start,test_end = 0,2,4,24

    start_time = time.clock()

    classifier = AntecedentClassifier()
    classifier.get_bos_trigger_patterns = True

    if train_start == -1: classifier.importdata(0, test_end)
    else: classifier.importdata(train_start, test_end)
    # classifier.set_ants_and_triggers(verbose=False)
    classifier.print_gs()
    """
    print 'Generating possible antecedents...'

    print 'Time taken to this point ',(time.clock()-start_time)

    print 'Setting antecedent and trigger contexts...'
    classifier.set_contexts(verbose=False)
    classifier.make_feature_vectors()

    classifier.balance_data(train_start, train_end)

    def runmodels(train_start, train_end, test_start, test_end, pp_classifier_name, options=False, oversample=1, probability=None, single_test=None, verbose=False, log_start=None):
        for name in classifier.classifier_names:
            if (single_test and name==single_test) or not single_test:
                print '\n%s with oversample x%d:'%(name,oversample)
                runmodel(classifier, name, train_start, train_end, test_start, test_end, pp_classifier_name, options=options, oversample=oversample, probability=probability ,verbose=verbose, log_start=log_start)

    log_file_start = 'RESULT'

    prob = 0.0001

    runmodels(train_start,train_end,test_start,test_end, 'LogReg', options=True, oversample=2, probability=prob, single_test='DTree', verbose=False, log_start=log_file_start)
    runmodels(train_start,train_end,test_start,test_end, 'LogReg', options=True, oversample=2, probability=prob, single_test='Forest', verbose=False, log_start=log_file_start)
    # runmodels(train_start,train_end,test_start,test_end, 'LogReg', options=True, oversample=2, probability=prob, single_test='SVM', verbose=False, log_start=log_file_start)
    runmodels(train_start,train_end,test_start,test_end, 'LogRegCV', options=True, oversample=2, probability=prob, single_test='DTree', verbose=False, log_start=log_file_start)
    runmodels(train_start,train_end,test_start,test_end, 'LogRegCV', options=True, oversample=2, probability=prob, single_test='Forest', verbose=False, log_start=log_file_start)
    # runmodels(train_start,train_end,test_start,test_end, 'LogRegCV', options=True, oversample=2, probability=prob, single_test='SVM', verbose=False, log_start=log_file_start)
    runmodels(train_start,train_end,test_start,test_end, 'oracle', options=True, oversample=2, probability=prob, single_test='DTree', verbose=False, log_start=log_file_start)
    runmodels(train_start,train_end,test_start,test_end, 'oracle', options=True, oversample=2, probability=prob, single_test='Forest', verbose=False, log_start=log_file_start)
    # runmodels(train_start,train_end,test_start,test_end, 'oracle', options=True, oversample=2, probability=prob, single_test='SVM', verbose=False, log_start=log_file_start)
    runmodels(train_start,train_end,test_start,test_end, 'dont', options=True, oversample=2, probability=prob, single_test='DTree', verbose=False, log_start=log_file_start)
    runmodels(train_start,train_end,test_start,test_end, 'dont', options=True, oversample=2, probability=prob, single_test='Forest', verbose=False, log_start=log_file_start)
    # runmodels(train_start,train_end,test_start,test_end, 'dont', options=True, oversample=2, probability=prob, single_test='SVM', verbose=False, log_start=log_file_start)
    """

    print '\nThere were supposed to be %d TP antecedents.'%(len(classifier.goldstandard_antecedents.antecedents))
    print 'We missed %d antecedents because our algorithm didnt account for them.'%classifier.cheated_antecedents
    print 'Time taken: ',(time.clock()-start_time)

main()
