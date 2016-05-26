# coding=utf-8
__author__ = 'kian'

import vpe_objects as vpe
import vector_creation as vc
import word_characteristics as wc
import numpy as np
import nltktree as nt
import warnings
from file_names import Files
from os import listdir
from sys import argv
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, vstack

files = Files()
MRG_DATA_FILE = 'dataset_with_features_ALL_AUXS.npy'
AUTO_PARSE_FILE = 'auto_parse_with_features_FULL_DATASET.npy'
AUTO_PARSE_XML_DIR = '/Users/kian/Documents/HONOR/xml_annotations/raw_auto_parse/'

class Dataset(object):
    def __init__(self):
        self.sentences = []
        self.auxs = []
        self.gold_auxs = []
        self.X = []
        self.Y = []
        self.section_ends = {k:-1 for k in range(0,25)}

    def add(self, section):
        self.sentences += section.sentences
        self.auxs += section.all_auxs
        self.gold_auxs += section.gold_auxs
        self.section_ends[int(section.section_num)] = len(self.sentences)

    def total_length(self):
        return len(self.auxs)

    def set_all_auxs(self, features=vc.get_all_features(), reset=False):
        if reset:
            self.X = []
            self.Y = []

        if not self.X:
            self.X += self.all_auxs_to_features(features)
            self.Y += self.get_aux_classifications()

    def get_auxs_by_type(self, type_):
        if type_ == 'all':
            return self.X, self.Y

        x,y = [],[]
        for i,aux in enumerate(self.auxs):
            if aux.type == type_:
                x.append(self.X[i])
                y.append(self.Y[i])
        return x,y

    def get_aux_list_by_type(self, type_):
        if type_ == 'all':
            return self.auxs
        return [aux for aux in self.auxs if aux.type==type_]

    def fix_auxs(self):
        for i,aux in enumerate(self.auxs):
            if (aux.wordnum, aux.sentnum) in [(30,14811),(11,15321),(12,15626),(39,15894)]:
                aux.is_trigger = True
                self.Y[i] = 1

        for aux in self.gold_auxs + self.auxs:
            sent = self.sentences[aux.sentnum]
            try:
                if sent.words[aux.wordnum + 1] == 'so' or sent.words[aux.wordnum + 1] == 'likewise':
                    aux.type = 'so'
                if sent.words[aux.wordnum + 1] == 'the' and sent.words[aux.wordnum + 2] in ['same', 'opposite']:
                    aux.type = 'so'
            except IndexError:
                pass

    def test_rules(self, train_auxs):
        f = lambda x: 1 if x else 0

        predictions = []
        for i in range(len(train_auxs)):
            aux = train_auxs[i]
            sendict = self.sentences[aux.sentnum]
            tree = sendict.get_nltk_tree()
            word_subtree_positions = nt.get_smallest_subtree_positions(tree)

            if aux.type == 'modal': predictions.append(f(wc.modal_rule(sendict, aux, tree, word_subtree_positions)))
            elif aux.type == 'be': predictions.append(f(wc.be_rule(sendict, aux)))
            elif aux.type == 'have': predictions.append(f(wc.have_rule(sendict, aux)))
            elif aux.type == 'do': predictions.append(f(wc.do_rule(sendict, aux, tree, word_subtree_positions)))
            elif aux.type == 'so': predictions.append(f(wc.so_rule(sendict, aux)))
            elif aux.type == 'to': predictions.append(f(wc.to_rule(sendict, aux)))

        return predictions

    def serialize(self, mrg_data=True):
        print 'Serializing data...'
        fname = MRG_DATA_FILE if mrg_data else AUTO_PARSE_FILE
        np.save(fname, np.array([self]))

    def all_auxs_to_features(self, features):
        x = []

        frequent_words = files.extract_data_from_file(Files.EACH_UNIQUE_WORD_NEAR_AUX)
        all_pos = files.extract_data_from_file(Files.EACH_UNIQUE_POS_FILE)
        pos_bigrams = wc.pos_bigrams(all_pos)

        for aux in self.auxs:
            sentdict = self.sentences[aux.sentnum]
            x.append(csr_matrix(vc.make_vector(sentdict, aux, features, vpe.ALL_CATEGORIES, vpe.AUX_LEMMAS,
                                               vpe.ALL_AUXILIARIES, frequent_words, all_pos, pos_bigrams)))
        return x

    def get_aux_classifications(self):
        return [1 if aux.is_trigger else 0 for aux in self.auxs]

    def run_cross_validation(self, X, Y, model, k_fold=5, oversample=1, verbose=False,
                             check_fp=False, rand=1917, aux_type='all'):
        if verbose: print 'Performing cross-validation...'
        model_name = type(model).__name__

        train_results = []
        test_results = []
        baseline_results = []

        kf = KFold(len(X), n_folds=k_fold, shuffle=True, random_state=rand)

        assert len(X) == len(Y)

        X = np.array(X)
        Y = np.array(Y)

        fold = 1
        for train_idx, test_idx in kf:
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            assert len(X_train) == len(Y_train) and len(X_test) == len(Y_test)

            if oversample > 1:
                X_train, Y_train = Dataset.oversample(X_train, Y_train, multiplier=oversample)

            X_train = vstack_csr_vecs(X_train)
            X_test = vstack_csr_vecs(X_test)
            test_auxs = np.array(self.get_aux_list_by_type(aux_type))[test_idx]

            # Normalize data according to the standard deviation of the training set.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # ignore the convert int to float warnings
                if verbose: print 'Normalizing data...'
                s = StandardScaler(with_mean=False)  # no mean because sparse data
                X_train = s.fit_transform(X_train)
                X_test = s.transform(X_test)

            # Fit the model.
            if verbose: print 'Fitting %s...'%model_name
            model.fit(X_train, np.array(Y_train))

            # Predict.
            if verbose: print 'Predicting with %s...'%model_name
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Results.
            train_results.append(accuracy_results(Y_train, train_pred))
            test_results.append(accuracy_results(Y_test, test_pred))
            baseline_results.append(accuracy_results(Y_test, self.test_rules(test_auxs)))

            if check_fp:
                analyze_results(Y_test, test_pred, self.sentences, self.auxs)

            if verbose:
                print 'Fold %d train results: '%fold,train_results[-1]
                print 'Fold %d test  results: '%fold,test_results[-1]
            fold += 1

        for lst in train_results,test_results,baseline_results:
            if lst == train_results:
                if verbose:
                    print '\nTraining set - average CV results for %s:'%model_name
            elif lst == test_results:
                print '\nTesting sets - average CV results for %s:'%model_name
            else:
                print '\nTesting sets - BASELINE CV results for %s:'%model_name

            print 'Precision: %0.2f'%np.mean([t[0] for t in lst])
            print 'Recall: %0.2f'%np.mean([t[1] for t in lst])
            print 'F1: %0.2f'%np.mean([t[2] for t in lst])

    @staticmethod
    def oversample(x, y, multiplier=5):
        assert len(x) == len(y)

        new_x = []
        new_y = []
        for i in range(len(x)):
            if y[i] == 1:
                for _ in range(multiplier):
                    new_x.append(x[i])
                    new_y.append(y[i])
            else:
                new_x.append(x[i])
                new_y.append(y[i])

        assert len(new_x) == len(new_y)
        return new_x, new_y

    @classmethod
    def load_dataset(cls, mrg_data=True):
        """
        @type return: Dataset
        """
        print 'Loading data...'
        fname = MRG_DATA_FILE if mrg_data else AUTO_PARSE_FILE
        d = np.load(fname)[0]
        d.fix_auxs()
        return d


class Section(object):
    def __init__(self, section_num, sentences, auxs, gold_auxs):
        self.section_num = section_num
        self.sentences = sentences
        self.all_auxs = auxs
        self.gold_auxs = gold_auxs

        # We now just have to say which auxs are the gold standard ones within the 'all_auxiliaries' object by
        # changing their "is_trigger" attribute.
        crt_gold_aux_idx = 0
        crt_gold_aux = self.gold_auxs[crt_gold_aux_idx]
        for aux in self.all_auxs:
            if crt_gold_aux.equals(aux):
                aux.is_trigger = True
                crt_gold_aux_idx += 1
                try:
                    crt_gold_aux = self.gold_auxs[crt_gold_aux_idx]
                except IndexError:
                    break
            try:
                sent = self.sentences[aux.sentnum]
                if sent[aux.wordnum + 1] == 'so' or sent[aux.wordnum + 1] == 'likewise':
                    aux.type = 'so'
                if sent[aux.wordnum + 1] == 'the' and sent[aux.wordnum + 2] in ['same', 'opposite']:
                    aux.type = 'so'
            except IndexError:
                pass

    def aux_length(self):
        return len(self.all_auxs)

    def get_aux_classifications(self):
        return [1 if aux.is_trigger else 0 for aux in self.all_auxs]

    def all_auxs_to_features(self, features):
        x = []

        # Parameters for the feature extraction.
        frequent_words = files.extract_data_from_file(Files.EACH_UNIQUE_WORD_NEAR_AUX)
        all_pos = files.extract_data_from_file(Files.EACH_UNIQUE_POS_FILE)
        pos_bigrams = wc.pos_bigrams(all_pos)

        for aux in self.all_auxs:
            sentdict = self.sentences[aux.sentnum]
            x.append(csr_matrix(vc.make_vector(sentdict, aux, features, vpe.ALL_CATEGORIES, vpe.AUX_LEMMAS,
                                               vpe.ALL_AUXILIARIES, frequent_words, all_pos, pos_bigrams)))
        return x


def load_data_into_sections(get_mrg=True, complete_mrg=True, sec_fun=lambda sec: True):
    dataset = Dataset()

    acc_sentnum = -1

    section_dirs = sorted(listdir(Files.XML_MRG))
    for d in section_dirs:
        if d.startswith('.') or not sec_fun(d):
            continue

        # Get all files we are concerned with. We don't load data from files with no instances of VPE.
        subdir = d + Files.SLASH_CHAR
        annotations = vpe.AnnotationSection(subdir, Files.VPE_ANNOTATIONS)
        vpe_files = sorted(set([annotation.file for annotation in annotations]))

        file_list = listdir(Files.XML_MRG + subdir)

        sentences, auxs, gold_auxs = [], [], []
        for f in vpe_files:
            try:
                extension = '.mrg.xml' if get_mrg else '.xml'
                path = Files.XML_MRG + subdir if get_mrg else AUTO_PARSE_XML_DIR

                # This condition makes it so that we use the same files for auto-parse dataset results.
                # if not f + '.mrg.xml' in file_list:
                #     raise IOError

                xml_data = vpe.XMLMatrix(f + extension, path)

            except IOError:  # The file doesn't exist as an MRG.
                if complete_mrg:
                    xml_data = vpe.XMLMatrix(f + '.pos.xml', Files.XML_POS)
                else:
                    continue

            auxs += xml_data.get_all_auxiliaries(sentnum_modifier=acc_sentnum).auxs
            gold_auxs += xml_data.get_gs_auxiliaries(annotations.get_anns_for_file(f), acc_sentnum)

            xml_sents = xml_data.get_sentences()
            sentences += xml_sents

            acc_sentnum += len(xml_sents)

        dataset.add(Section(int(d), sentences, auxs, gold_auxs))

    return dataset

def vstack_csr_vecs(vecs):
    m = vecs[0]
    for i in range(1, len(vecs)):
        m = vstack((m, vecs[i]), format='csr')
    return m

def accuracy_results(y_true, y_pred):
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)

def analyze_results(y_true, y_pred, sentences, auxs):
    for i,value in enumerate(y_true):
        if value == 0 and y_pred[i] == 1:  # false positive
            aux = auxs[i]
            print sentences[aux.sentnum]
            print aux
            print

def run_feature_ablation(loaded_data, exclude=True):
    # Features: ['words','pos','bigrams','my_features','old_rules','square_rules','combine_aux_type']
    all_features = vc.get_all_features()
    features = [['words'],
                ['pos'],
                ['bigrams'],
                ['my_features'],
                ['old_rules'],
                ['old_rules','square_rules'],
                ['combine_aux_type']]
    newf = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            newf.append(features[i] + features[j])

    newf.remove(['old_rules','old_rules','square_rules'])
    for ablated in features:
        if exclude:
            print 'Feature EXCLUDED: ',ablated
            ablation_features = [f for f in all_features if f not in ablated]
        else:
            print 'Only feature included: ',ablated
            # ablation_features = [f for f in features if f!=ablated]
            ablation_features = ablated

        loaded_data.set_all_auxs(ablation_features, reset=True)
        loaded_data.run_cross_validation(loaded_data.X, loaded_data.Y, LogisticRegressionCV(),
                                         oversample=5, check_fp=False, rand=1489987)
        print '------------------------------------------'

def analyze_auxs(data):
    freqs = {}
    for aux in data.auxs:
        if aux.type not in freqs:
            freqs[aux.type] = 1
        else:
            freqs[aux.type] += 1
    gs_freqs = {}
    for aux in data.gold_auxs:
        if aux.type not in gs_freqs:
            gs_freqs[aux.type] = 1
        else:
            gs_freqs[aux.type] += 1

    print 'All auxs:'
    for key in freqs: print key,freqs[key]
    print '\nGold auxs:'
    for key in gs_freqs: print key,gs_freqs[key]
    print 'Total gold auxs:',sum(gs_freqs.itervalues())

def load_bos_2012_partition():
    data = Dataset.load_dataset(mrg_data=False)

    train_secs = [0,1,2,3,4,5,6,7,8,10,12,14]
    test_secs = [9,11,13,15]

    train_auxs = []
    train_idxs = []
    test_auxs = []
    test_idxs = []

    for i,aux in enumerate(data.auxs):
        if aux.type == 'do': # bos only considered do-vpe
            section = None # first find section the aux belongs to
            for sec in sorted(data.section_ends.iterkeys()):
                if aux.sentnum < data.section_ends[sec]:
                    section = sec
                    break

            if section in train_secs:
                train_auxs.append(aux)
                train_idxs.append(i)

            if section in test_secs:
                test_auxs.append(aux)
                test_idxs.append(i)

    data.X = np.array(data.X)
    data.Y = np.array(data.Y)

    train_X = data.X[train_idxs]
    train_Y = data.Y[train_idxs]
    test_X = data.X[test_idxs]
    test_Y = data.Y[test_idxs]

    train_X, train_Y = Dataset.oversample(train_X, train_Y, 5)

    print 'Training classifier...'
    classifier = LogisticRegressionCV()
    classifier.fit(vstack_csr_vecs(train_X), train_Y)

    predictions = classifier.predict(vstack_csr_vecs(test_X))

    print 'Results acquired from using our algorithm on Bos\' train-test split:'
    print accuracy_results(test_Y, predictions)

def bos_train_test_split():
    data = Dataset.load_dataset(mrg_data=False)
    train = range(0,15)
    test = range(20,25)

    train_auxs, test_auxs = [], []
    train_idxs, test_idxs = [], []

    for i,aux in enumerate(data.auxs):
        section = find_section(aux.sentnum, data.section_ends)

        if section in train:
            train_auxs.append(aux)
            train_idxs.append(i)

        if section in test:
            test_auxs.append(aux)
            test_idxs.append(i)

    data.X = np.array(data.X)
    data.Y = np.array(data.Y)

    train_X = data.X[train_idxs]
    train_Y = data.Y[train_idxs]
    test_X = data.X[test_idxs]
    test_Y = data.Y[test_idxs]

    train_X, train_Y = Dataset.oversample(train_X, train_Y, 5)

    print 'Training classifier...'
    classifier = LogisticRegressionCV()
    classifier.fit(vstack_csr_vecs(train_X), train_Y)

    predictions = classifier.predict(vstack_csr_vecs(test_X))

    print 'Results acquired from using our algorithm on the bos train-test split:'
    print accuracy_results(test_Y, predictions)


def find_section(sentnum, section_dict):
    for sec in sorted(section_dict.iterkeys()):
        if sentnum < section_dict[sec]:
            return sec

if __name__ == '__main__':
    mrg = 'mrg' in argv

    if 'save' in argv:
        data = load_data_into_sections(get_mrg=mrg, complete_mrg=True)
        data.set_all_auxs()
        data.serialize(mrg_data=mrg)

    if 'load' in argv:
        data = Dataset.load_dataset(mrg_data=mrg)
        for type_ in ['all','so','do','to','modal','have','be']:
            type_x, type_y = data.get_auxs_by_type(type_)
            for model in [LogisticRegressionCV()]:#[LogisticRegression(), LogisticRegressionCV(), SVC(), LinearSVC()]:
                print type_
                data.run_cross_validation(type_x, type_y, model, oversample=5, check_fp=False, rand=1489987, aux_type=type_)
                print '------------------------------------------'

    if 'ablate' in argv:
        data = Dataset.load_dataset(mrg_data=mrg) #MRG OR NO?
        run_feature_ablation(data)

    if 'analyze' in argv:
        data = Dataset.load_dataset(mrg_data=mrg)
        analyze_auxs(data)

    if 'bos' in argv:
        load_bos_2012_partition()

    if 'bos_spen' in argv:
        bos_train_test_split()
