__author__ = 'kian'

import vpe_objects as vpe
import vector_creation as vc
import word_characteristics as wc
import numpy as np
import warnings
from file_names import Files
from os import listdir
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, vstack

files = Files()

class Dataset(object):
    def __init__(self):
        self.sentences = []
        self.auxs = []
        self.gold_auxs = []
        self.X = []
        self.Y = []

    def add(self, section):
        self.sentences += section.sentences
        self.auxs += section.all_auxs
        self.gold_auxs += section.gold_auxs

    def total_length(self):
        return len(self.auxs)

    def set_all_auxs(self):
        if not self.X:
            self.X += self.all_auxs_to_features(vc.get_all_features())
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

    def fix_auxs(self):
        for i,aux in enumerate(self.auxs):
            if (aux.wordnum, aux.sentnum) in [(30,14811),(11,15321),(12,15626),(39,15894)]:
                aux.is_trigger = True
                self.Y[i] = 1

    def serialize(self):
        print 'Serializing data...'
        np.save('dataset_with_features.npy', np.array([self]))

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

    def run_cross_validation(self, X, Y, model, k_fold=5, oversample=1, verbose=False, check_fp=False):
        if verbose: print 'Performing cross-validation...'
        model_name = type(model).__name__

        train_results = []
        test_results = []

        kf = KFold(len(X), n_folds=k_fold, shuffle=True, random_state=1917)

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

            if check_fp:
                analyze_results(Y_test, test_pred, self.sentences, self.auxs)

            if verbose:
                print 'Fold %d train results: '%fold,train_results[-1]
                print 'Fold %d test  results: '%fold,test_results[-1]
            fold += 1

        for lst in train_results,test_results:
            if lst == train_results:
                print '\nTraining set - average CV results for %s:'%model_name
            else:
                print '\nTesting sets - average CV results for %s:'%model_name

            print 'Precision: %0.2f'%np.mean([t[0] for t in lst])
            print 'Recall: %0.2f'%np.mean([t[1] for t in lst])
            print 'F1: %0.2f'%np.mean([t[2] for t in lst])

    @classmethod
    def load_dataset(cls):
        print 'Loading data...'
        d = np.load('dataset_with_features.npy')[0]
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


def load_data_into_sections(get_mrg=True):
    dataset = Dataset()

    acc_sentnum = -1

    section_dirs = sorted(listdir(Files.XML_MRG))
    for d in section_dirs:
        if d.startswith('.'):
            continue

        # Get all files we are concerned with. We don't load data from files with no instances of VPE.
        subdir = d + Files.SLASH_CHAR
        annotations = vpe.AnnotationSection(subdir, Files.VPE_ANNOTATIONS)
        vpe_files = sorted(set([annotation.file for annotation in annotations]))

        sentences, auxs, gold_auxs = [], [], []
        for f in vpe_files:
            try:
                extension = '.mrg.xml' if get_mrg else '.xml'
                xml_data = vpe.XMLMatrix(f + extension, Files.XML_MRG + subdir)
            except IOError:  # The file doesn't exist.
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

if __name__ == '__main__':
    # data = load_data_into_sections()
    # data.set_all_auxs()
    # data.serialize()
    data = Dataset.load_dataset()

    for type_ in ['do','to','modal','have','be']:
        type_x, type_y = data.get_auxs_by_type(type_)
        for model in [LogisticRegressionCV()]:#[LogisticRegression(), LogisticRegressionCV(), SVC(), LinearSVC()]:
            data.run_cross_validation(type_x, type_y, model, oversample=5, check_fp=False)
            print '------------------------------------------'
