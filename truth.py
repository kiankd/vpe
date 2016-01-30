from old.detectVPE import maprawtomrg,filevpelocations
import xml.etree.ElementTree as ET

# Just a small file for truth values if we ever decide to not use 1s and 0s.
# Also stores globals. And other useful methods

FULL_PROJECT_DIR = '/Users/kian/Documents/HONOR/'
# FULL_PROJECT_DIR = '/home/2014/kkenyo1/sura/'

XML_ANNOTATIONS_DIR = FULL_PROJECT_DIR+'xml_annotations/'
XML_MRG = XML_ANNOTATIONS_DIR+'wsj_mrg_good_extraction/'
XML_POS = XML_ANNOTATIONS_DIR+'wsj_pos/'
XML_RAW_TOKENIZED = XML_ANNOTATIONS_DIR+'tokenized_raw/'
VPE_ANNOTATIONS = FULL_PROJECT_DIR+'vpe_annotations/wsj/'

LOG_LOCATIONS = FULL_PROJECT_DIR+'project/logs/'

SLASH_CHAR = '/'
DROP_BOX_DIR = '/Users/kian/Dropbox/'

# For PC:
# #
# FULL_PROJECT_DIR = 'C:\\Users\\Kian\\Sura\\'
#
# XML_ANNOTATIONS_DIR = FULL_PROJECT_DIR+'xml_annotations\\'
# XML_MRG = XML_ANNOTATIONS_DIR+'wsj_mrg_good_extraction\\'
# XML_POS = XML_ANNOTATIONS_DIR+'wsj_pos\\'
# XML_RAW_TOKENIZED = XML_ANNOTATIONS_DIR+'tokenized_raw\\'
# VPE_ANNOTATIONS = FULL_PROJECT_DIR+'vpe_annotations\\wsj\\'
#
# LOG_LOCATIONS = FULL_PROJECT_DIR+'project\\logs\\'
# ANT_LOG_LOCATIONS = FULL_PROJECT_DIR+'project\\ant_logs\\'
# SLASH_CHAR = '\\'
# DROP_BOX_DIR = 'C:\\Users\\Kian\\Dropbox\\'


ANT_LOG_LOCATIONS = DROP_BOX_DIR+'project'+SLASH_CHAR+'ant_logs'+SLASH_CHAR
SVM_FILE_LOCATIONS = DROP_BOX_DIR+'project'+SLASH_CHAR+'svm_logs'+SLASH_CHAR

UNIQUE_AUXILIARIES_FILE = 'unique_auxs.txt'
GOLD_STANDARD_FILE = 'gs_hasvpe_each_aux.txt'
EACH_UNIQUE_WORD_FILE = 'unique_words.txt'
EACH_UNIQUE_LEMMA_FILE = 'unique_lemmas.txt'
EACH_UNIQUE_POS_FILE = 'unique_postags.txt'
EACH_UNIQUE_WORD_NEAR_AUX = 'unique_words_close_to_aux.txt'
SECTION_SPLIT = 'section_split.txt'
WORD2VEC_FILE = 'word2vec_vectors_wsj_all_words.txt'


WORD2VEC_LENGTH = 300
SENTENCE_SEARCH_DISTANCE = 1 #TODO: NOTE THAT I CHANGE THIS TO SEARCH ONLY 1 SENTENCE BEHIND. PROBS CHANGE BACK!!!!!
MINIMUM_CONTEXT_DISTANCE = 3

PP_TRAIN_ANTS_PER_TRIG = 5

HEAD_MATCH_MODIFIER = 1.3
EXACT_MATCH_MODIFIER = 1.50

NIELSON_SENTENIAL_COMPLEMENT_PHRASES = ['WHNP','ADVP','ADV','SINV','WHADVP','QP','RB','IN']
ALL_PHRASES = ['ADJP','ADVP','CONJP','FRAG','INTJ','LST','NAC','NP','NX','PP','PRN','PRT','QP','RRC','UCP','VP','WHADJP','WHADVP','WHNP','WHPP','X','PRD']

def truth(boolean):
    if boolean: return 1
    return 0

def untruth(i):
    if i==1: return True
    return False

def extract_data_from_file(file_name):
    ret = []
    f = open(SVM_FILE_LOCATIONS+file_name, 'r')
    for line in f:
        l = line[0:-1] # Don't include \n
        if '\r' in l:
            l = l[0:-1]
        ret.append(l)
    f.close()
    return ret

def makefile(new_file_name, data):
    print 'Writing new file, %s...'%new_file_name

    f = open(SVM_FILE_LOCATIONS+new_file_name, 'w')
    for item in data:
        f.write('%s\n'%item)
    f.close()

def loadword2vecs():
    f = open(SVM_FILE_LOCATIONS+WORD2VEC_FILE, 'r')
    dic = {}
    for line in f:
        str_vec = line.split(',')
        word = str_vec[0]
        str_vec = str_vec[1:len(str_vec)]
        dic[word] = []
        for string in str_vec:
            dic[word].append(round(float(string),10)) # TODO: I AM ROUNDING TO NEAREST 10 digits past decimal!

    return dic

def get_antecedent_head_index(sentdict, ant):
    idx = -1

    head = ant.get_words()[0]
    for i in range(0,len(sentdict['words'])):
        if sentdict['words'][i] == head:
            if idx == -1:
                idx = i
            # Else, the antecedent head repeats in the sentence.
            elif ant.get_words()[1:len(ant.get_words())] == sentdict['words'][i+1:len(ant.get_words())]:
                idx = i
    if idx == -1:
        return 0
    return idx

def import_goldstandard_ants(xml_f, xml_raw_tok_path, subdir, annotationmatrix, mrgmatrix, total_sentences):
    antecedent_sentence_numbers = []
    antecedents = [] # List of indexes where the antecedent takes place in the corresponding sentence above.

    xml_file = xml_f[0:8]+'.xml'

    ann_file_indexes = filevpelocations(xml_f, annotationmatrix)[0]

    sentence_map = maprawtomrg(xml_file, xml_raw_tok_path, subdir, mrgmatrix)

    parser = ET.XMLParser()
    tokenizedtree = ET.parse(xml_raw_tok_path + subdir + xml_file, parser=parser)
    root = tokenizedtree.getroot()

    if ann_file_indexes:
        for sentence in root.iter('sentence'):
            rawidx = int(sentence.get('id'))
            words = [word.text for word in sentence.iter('word')]
            offset_starts = [int(offset_start.text) for offset_start in sentence.iter('CharacterOffsetBegin')]
            offset_ends   = [int(offset_end.text) for offset_end in sentence.iter('CharacterOffsetEnd')]

            for idx in ann_file_indexes:
                start,end = (-1,-1)

                if int(annotationmatrix[idx]['antoffsetstart']) in offset_starts and int(annotationmatrix[idx]['antoffsetend']) in offset_ends:
                    start,end = int(annotationmatrix[idx]['antoffsetstart']),int(annotationmatrix[idx]['antoffsetend'])

                    try:
                        antecedent_sentence_numbers.append(sentence_map[rawidx]+total_sentences)
                    except KeyError:
                        break

                if [start,end].count(-1) == 1:
                    print '\nUh-oh! An offset_start or offset_end is in the file, but the corresponding one isnt!\nLook at the words:'
                    print words
                    print

                elif [start,end].count(-1) == 0: # We found one!
                    antecedent = []
                    for i in range(0,len(words)):
                        if offset_starts[i] >= start and offset_ends[i] <= end:
                            antecedent.append(words[i])

                    antecedents.append(antecedent)

    return antecedent_sentence_numbers,antecedents























