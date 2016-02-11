from sys import platform
import word_characteristics as wc
import operator

def number_of_words_surrounding_aux():
    return 1000

class Files:
    if platform == 'darwin':
        FULL_PROJECT_DIR = '/Users/kian/Documents/HONOR/'

        XML_ANNOTATIONS_DIR = FULL_PROJECT_DIR+'xml_annotations/'
        XML_MRG = XML_ANNOTATIONS_DIR+'wsj_mrg_good_extraction/'
        XML_POS = XML_ANNOTATIONS_DIR+'wsj_pos/'
        XML_RAW_TOKENIZED = XML_ANNOTATIONS_DIR+'tokenized_raw/'
        VPE_ANNOTATIONS = FULL_PROJECT_DIR+'vpe_annotations/wsj/'

        IMPORTED_DATA = FULL_PROJECT_DIR + 'npy_data/imported_data.npy'

        LOG_LOCATIONS = FULL_PROJECT_DIR+'project/logs/'

        SLASH_CHAR = '/'
        DROP_BOX_DIR = '/Users/kian/Dropbox/'
        RESULT_LOGS_LOCATION = DROP_BOX_DIR + 'project/result_logs/'
   
    elif platform == 'linux2':
        FULL_PROJECT_DIR = '/home/2014/kkenyo1/vpe_project/'

        IMPORTED_DATA = FULL_PROJECT_DIR + 'npy_data/imported_data.npy'      
        DROP_BOX_DIR = FULL_PROJECT_DIR
        SLASH_CHAR = '/'
    else:
        FULL_PROJECT_DIR = 'C:\\Users\\Kian\\Sura\\'

        XML_ANNOTATIONS_DIR = FULL_PROJECT_DIR+'xml_annotations\\'
        XML_MRG = XML_ANNOTATIONS_DIR+'wsj_mrg_good_extraction\\'
        XML_POS = XML_ANNOTATIONS_DIR+'wsj_pos\\'
        XML_RAW_TOKENIZED = XML_ANNOTATIONS_DIR+'tokenized_raw\\'
        VPE_ANNOTATIONS = FULL_PROJECT_DIR+'vpe_annotations\\wsj\\'

        LOG_LOCATIONS = FULL_PROJECT_DIR+'project\\logs\\'

        SLASH_CHAR = '\\'
        DROP_BOX_DIR = 'C:\\Users\\Kian\\Dropbox\\'

    UNIQUE_AUXILIARIES_FILE = 'unique_auxs.txt'
    GOLD_STANDARD_FILE = 'gs_hasvpe_each_aux.txt'
    EACH_UNIQUE_WORD_FILE = 'unique_words.txt'
    EACH_UNIQUE_LEMMA_FILE = 'unique_lemmas.txt'
    EACH_UNIQUE_POS_FILE = 'unique_postags.txt'
    EACH_UNIQUE_WORD_NEAR_AUX = 'unique_words_close_to_aux.txt'
    SECTION_SPLIT = 'section_split.txt'
    WORD2VEC_FILE = 'word2vec_vectors_wsj_all_words.txt'

    WORD2VEC_LENGTH = 300

    def __init__(self):
        self.SVM_FILE_LOCATIONS = self.DROP_BOX_DIR+'project'+self.SLASH_CHAR+'svm_logs'+self.SLASH_CHAR

        if platform == 'linux2':
            self.SVM_FILE_LOCATIONS = self.FULL_PROJECT_DIR+'helper_files/'

    def extract_data_from_file(self, file_name):
        ret = []
        f = open(self.SVM_FILE_LOCATIONS+file_name, 'r')
        for line in f:
            l = line[0:-1] # Don't include \n
            if '\r' in l:
                l = l[0:-1]
            ret.append(l)
        f.close()
        return ret

    def make_file(self, new_file_name, data):
        print 'Writing new file, %s...'%new_file_name

        f = open(self.SVM_FILE_LOCATIONS+new_file_name, 'w')
        for item in data:
            f.write('%s\n'%item)
        f.close()

    def make_all_the_files(self, sentdicts, word_distance_from_aux=3):

        words,lemmas,pos_tags,words_near_aux = [],[],[],[]
        for sentdict in sentdicts:
            for i in range(0,len(sentdict)):
                if wc.is_auxiliary(sentdict, i, [], [], raw=False):
                    for j in range(max(0,i-word_distance_from_aux), min(len(sentdict),i+word_distance_from_aux+1)):
                        if j != i:
                            words_near_aux.append(sentdict.words[j])

                words.append(sentdict.words[i])
                lemmas.append(sentdict.lemmas[i])
                pos_tags.append(sentdict.pos[i])

        words = set(words)
        lemmas = set(lemmas)
        pos_tags = set(pos_tags)

        freq_dict = {}
        for w in words_near_aux:
            if w not in freq_dict:
                freq_dict[w] = 1
            else:
                freq_dict[w] += 1

        sorted_by_freq = sorted(freq_dict.items(), key=operator.itemgetter(1))
        most_frequent_words_near_aux = [pair[0] for pair in sorted_by_freq[-1*number_of_words_surrounding_aux():len(sorted_by_freq)]]

        self.make_file(self.EACH_UNIQUE_WORD_FILE, words)
        self.make_file(self.EACH_UNIQUE_LEMMA_FILE, lemmas)
        self.make_file(self.EACH_UNIQUE_POS_FILE, pos_tags)
        self.make_file(self.EACH_UNIQUE_WORD_NEAR_AUX, most_frequent_words_near_aux)

    def load_word2vecs(self):
        f = open(self.SVM_FILE_LOCATIONS+self.WORD2VEC_FILE, 'r')
        dic = {}
        for line in f:
            str_vec = line.split(',')
            word = str_vec[0]
            str_vec = str_vec[1:len(str_vec)]
            dic[word] = []
            for string in str_vec:
                dic[word].append(round(float(string),10)) # TODO: I AM ROUNDING TO NEAREST 10 digits past decimal!

        return dic


NIELSON_SENTENIAL_COMPLEMENT_PHRASES = ['WHNP','ADVP','ADV','SINV','WHADVP','QP','RB','IN']
ALL_PHRASES = ['ADJP','ADVP','CONJP','FRAG','INTJ','LST','NAC','NP','NX','PP','PRN','PRT','QP','RRC','UCP','VP','WHADJP','WHADVP','WHNP','WHPP','X','PRD']
