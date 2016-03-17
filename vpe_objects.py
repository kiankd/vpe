import xml.etree.ElementTree as ET
import nltktree as nt
import word_characteristics as wc
from operator import attrgetter
from file_names import Files
from os import listdir
from truth import SENTENCE_SEARCH_DISTANCE
from numpy import dot

MODALS = ['can','could','may','must','might','will','would','shall','should']
BE     = ['be']
HAVE   = ['have']
DO     = ['do']
TO     = ['to']
SO     = ['so','same','likewise','opposite']

AUX_LEMMAS = MODALS+BE+HAVE+DO+TO+SO
ALL_CATEGORIES = [MODALS, BE, HAVE, DO, TO, SO]
ALL_AUXILIARIES = Files().extract_data_from_file(Files.UNIQUE_AUXILIARIES_FILE)

""" ---- Exception classes. ---- """
class AuxiliaryHasNoTypeException:
    def __init__(self, aux_name):
        print 'The following auxiliary, %s, has no category!'%aux_name

class EmptySentDictException:
    def __init__(self): pass

class GoldStandardComesFromRawException:
    def __init__(self): pass

class NoSubsequenceFoundException:
    def __init__(self): pass

class Finished:
    def __init__(self): pass

class WrongClassEquivalenceException:
    def __init__(self): pass

def ant_after_trigger(sentnum,i,j,trig):
    if sentnum > trig.sentnum:
        return True

    if sentnum < trig.sentnum:
        return False

    if i <= trig.wordnum <= j: # trig is in the ant
        return True

    if i >= trig.wordnum: #ant follows the trig
        return True

    return False


""" ---- Data importation classes. ---- """
class AllSentences:
    """ A class that contains all of the StanfordCoreNLP sentences. """
    def __init__(self):
        self.sentences = []

    def __iter__(self):
        """
        @type yield: SentDict
        """
        for sentdict in self.sentences:
            yield sentdict

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.get_sentence(item)

    def add_mrg(self, mrg_matrix):
        for sentdict in mrg_matrix:
            self.sentences.append(sentdict)

    def get_sentence(self, i):
        """
        @type return: SentDict
        """
        return self.sentences[i]

    def nearest_vp(self, trigger):
        """
            Returns an antecedent that represents the VP that is nearest to the trigger on the left.
            @type return: Antecedent
        """
        vp = None
        sent = None
        try:
            vp,start,end = nt.get_nearest_vp_exceptional(self.get_sentence_tree(trigger.sentnum), trigger.wordnum-1, trigger, trigger.sentnum) # Go behind trigger.
            sent = trigger.sentnum
        except nt.NoVPException:
            for sentnum in reversed(range(trigger.sentnum-2, trigger.sentnum)):
                try:
                    vp,start,end = nt.get_nearest_vp_exceptional(self.get_sentence_tree(sentnum), len(self[sentnum])-2, trigger, sentnum)
                    sent = sentnum
                except nt.NoVPException:
                    continue

        # assert vp,start,end
        # print vp,start,end
        return self.idxs_to_ant(sent, start, end, trigger)

    def set_possible_ants(self, trigger, pos_tests):
        for sentnum in range(max(0, trigger.sentnum - SENTENCE_SEARCH_DISTANCE), trigger.sentnum+1):
            functions = [f for f in pos_tests if hasattr(f, '__call__')]

            for i in range(len(self.sentences[sentnum])):
                 tag = self.sentences[sentnum].pos[i]

                 # TODO: ADDED SECOND CLAUSE TO THIS IF TO LOWER NUMBER OF CANDIDATES GENERATED
                 if True in (f(tag) for f in functions): #and not wc.is_aux_lemma(self.sentences[sentnum].lemmas[i]):
                     phrase = nt.get_nearest_phrase(nt.maketree(self.sentences[sentnum]['tree'][0]), i, pos_tests)
                     phrase_length = nt.get_phrase_length(phrase)

                     # if phrase_length <= 2:
                     #     print phrase

                     for j in range(i, min(i+phrase_length+1, len(self.sentences[sentnum]))):
                         if not ant_after_trigger(sentnum, i, j, trigger):
                             bad = False
                             for pos_check in [wc.is_preposition, wc.is_punctuation, wc.is_determiner]:
                                 if pos_check(self.sentences[sentnum].pos[j-1]):
                                     bad = True

                             if not bad:
                                 ant = self.idxs_to_ant(sentnum, i, j, trigger)
                                 if len(ant.sub_sentdict) > 0:
                                     trigger.add_possible_ant(ant)

            #Lets use the tree to decrease the number of dumb candidates.

            phrases = [p for p in pos_tests if type(p)==str]
            tree = nt.maketree(self.get_sentence(sentnum)['tree'][0])
            tree_pos = nt.getsmallestsubtrees(tree)
            tree_words = tree.leaves()
            leaves_dict = { (tree_pos[i].label(), tree_words[i]) : i+1 for i in range(len(tree_words))}

            new_ants = []
            for pos in phrases:
                for position in nt.phrase_positions_in_tree(tree, pos):
                    start = None
                    for w in nt.getsmallestsubtrees(tree[position]):
                        # print w.label(), w[0]
                        end = (w.label(), w[0])

                        if not start:
                            start = end

                        if sentnum == trigger.sentnum and leaves_dict[end] == trigger.wordnum or wc.is_punctuation(end[0]):
                            break

                        ant = self.idxs_to_ant(sentnum, leaves_dict[start], leaves_dict[end]+1, trigger)

                        i,j = leaves_dict[start],leaves_dict[end]
                        if len(ant.sub_sentdict) > 0 and not ant_after_trigger(sentnum, start, end, trigger)\
                            and not ((sentnum, leaves_dict[start], leaves_dict[end]+1, trigger) in new_ants):

                            bad = False
                            for pos_check in [wc.is_preposition, wc.is_punctuation]:
                                if pos_check(self.sentences[sentnum].pos[j-1]):
                                    bad = True

                            if not bad:
                                ant = self.idxs_to_ant(sentnum, i, j, trigger)
                                if len(ant.sub_sentdict) > 0:
                                    trigger.add_possible_ant(ant)
                                    new_ants.append((sentnum, leaves_dict[start], leaves_dict[end]+1, trigger))

                            # print 'added poss_ant: %d to %d'%(leaves_dict[start], leaves_dict[end]+1),trigger.possible_ants[-1]

                    # raise Finished()

    def idxs_to_ant(self, sentnum, start, end, trigger):
        sentdict = self.sentences[sentnum]
        return Antecedent(sentnum,
                          trigger,
                          SubSentDict(sentdict.words[start:end],
                                      sentdict.pos[start:end],
                                      sentdict.lemmas[start:end]),
                          start, end-1)

    def get_sentence_tree(self, i):
        return nt.maketree(self.sentences[i]['tree'][0])

class XMLMatrix:
    """ A matrix of SentDicts built by getting passed a Stanford CoreNLP XML file. """
    def __init__(self, xml_file, path, pos_file=False, get_deps=False):
        if not xml_file in listdir(path):
            raise IOError

        self.matrix = []
        self.file_name = xml_file
        self.pos_file = pos_file
        print 'Processing: %s' %xml_file
        parser = ET.XMLParser()
        tree = ET.parse(path+xml_file, parser=parser)
        root = tree.getroot()

        # For each <sentence> in the file...
        for sentence in root.iter('sentence'):
            try:
                int(sentence.get('id'))
            except TypeError:
                break
            try:
                s = SentDict(sentence, f_name=xml_file, get_deps=get_deps)
                self.matrix.append(s)
            except EmptySentDictException:
                continue

    def __iter__(self):
        for sentdict in self.matrix:
            yield sentdict

    def find_word_sequence(self, words, minimum_match=None):
        """
            Returns the (i,j) start and (i,k) end indexes for the mrgmatrix from which the 'words' sequence starts and then ends.
            Note that this assumes that the antecedent spans one single sentence (the ith sentdict).
            minimum_match allows for us to be less strict than an exact match of the words.
        """
        if not minimum_match:
            minimum_match = len(words)

        for i in range(len(self.matrix)):
            sentdict = self.matrix[i]
            for j in range(len(sentdict)):
                if sentdict.words[j] == words[0]:
                    count = 0
                    start = (i,j)
                    end = (-1,-1)
                    for k in range(j, min(len(sentdict), j+len(words))):
                        if sentdict.words[k] == words[count]:
                            count += 1
                        end = (i,k)

                    if count >= minimum_match:
                        return start,end
                    else:
                        continue
        raise NoSubsequenceFoundException()

    def get_subdir(self):
        return self.file_name[4:6]

    def get_gs_auxiliaries(self, annotations, sentnum_modifier):
        parser = ET.XMLParser()
        tree = ET.parse(Files.XML_RAW_TOKENIZED+self.get_subdir()+Files.SLASH_CHAR+self.file_name[0:8]+'.xml', parser=parser)
        root = tree.getroot()

        ann_num = 0
        crt_annotation = annotations[ann_num]
        raw_matrix = []
        raw_gold_auxiliaries = []
        raw_all_auxiliaries = Auxiliaries()
        raw_gold_indexes = []

        for sentence in root.iter('sentence'):
            try:
                s = SentDict(sentence, raw=True)
                for i in range(0,len(s)):
                    if s.offset_starts[i] == crt_annotation.vpe_offset_start:
                            #in range(crt_annotation.vpe_offset_start-1, crt_annotation.vpe_offset_start+2): #TODO: I MADE THIS LESS STRICT
                        raw_gold_indexes.append(len(raw_all_auxiliaries.auxs))
                        raw_gold_auxiliaries.append(RawAuxiliary(s.words[i], i, s.sentnum))
                        ann_num += 1
                        if not ann_num >= len(annotations):
                            crt_annotation = annotations[ann_num]

                    if wc.is_auxiliary(s, i, AUX_LEMMAS, ALL_AUXILIARIES, raw=True):
                        raw_all_auxiliaries.add_aux(RawAuxiliary(s.words[i], i, s.sentnum))

                raw_matrix.append(s)
            except EmptySentDictException:
                continue

        if len(raw_gold_auxiliaries) != len(annotations):
            print '\nAnnotations:'
            for ann in annotations:
                print ann
            print 'Number of auxs we got: %d, number of annotations for file %s: %d.'%(len(raw_gold_auxiliaries),crt_annotation.file,len(annotations))
            raise Exception('Error! When extracting the annotations using the raw data, we didn\'t get the correct number of auxiliaries!')

        """ Now that we got the auxiliaries according to their location/offsets within the raw text files (above),
            we now have to link the RawAuxiliaries with their corresponding MRG/POS XML file auxiliaries. """

        mrg_raw_type_auxiliaries = Auxiliaries()

        # We pass "raw" as true here because we want to use the same "is_auxiliary" method for the comparison.
        for sentdict in self:
            mrg_raw_type_auxiliaries.add_auxs(sentdict.get_auxiliaries(raw=True))

        gold_standard = []
        if len(raw_all_auxiliaries.auxs) == len(mrg_raw_type_auxiliaries.auxs):
            for raw_aux_idx in raw_gold_indexes:
                raw_aux = mrg_raw_type_auxiliaries.auxs[raw_aux_idx]
                aux_sentdict = self.matrix[raw_aux.sentnum-1]
                idx = raw_aux.wordnum

                mrg_aux = Auxiliary(aux_sentdict.sentnum+sentnum_modifier, aux_sentdict.words[idx], aux_sentdict.lemmas[idx], aux_sentdict.pos[idx], idx)
                gold_standard.append(mrg_aux)

        return gold_standard

    def get_gs_antecedents(self, annotations, triggers, sentnum_modifier):
        """Finds the gold standard antecedents using the old raw xml files and annotations to link to mrg xmls."""
        parser = ET.XMLParser()
        tree = ET.parse(Files.XML_RAW_TOKENIZED+self.get_subdir()+Files.SLASH_CHAR+self.file_name[0:8]+'.xml', parser=parser)
        root = tree.getroot()

        ann_num = 0
        crt_annotation = annotations[ann_num]
        raw_matrix = []
        raw_gold_ants = []
        try:
            for sentence in root.iter('sentence'):
                try:
                    s = SentDict(sentence, raw=True)
                    i = 0
                    ant_words = []
                    got = False
                    while i < len(s):
                        # print s.offset_starts[i],crt_annotation.ant_offset_start
                        if got or s.offset_starts[i] in range(crt_annotation.ant_offset_start-1,crt_annotation.ant_offset_start+2): #TODO: I MADE THIS LESS STRICT
                            ant_words.append(s.words[i])
                            if not got:
                                ant_start = i
                                got = True
                            if s.offset_ends[i] in range(crt_annotation.ant_offset_end-1, crt_annotation.ant_offset_end+2):
                                ann_num += 1
                                raw_gold_ants.append(RawAntecedent(ant_words, ant_start, i, s.sentnum))
                                try:
                                    crt_annotation = annotations[ann_num]
                                except IndexError:
                                    raise Finished()
                                ant_words = []
                                got = False
                        i+=1
                    raw_matrix.append(s)
                except EmptySentDictException:
                    continue
        except Finished:
            pass
        if len(raw_gold_ants) != len(annotations):
            print '\nAnnotations:'
            for ann in annotations:
                print ann
            print 'Number of ants we got: %d, number of annotations for file %s: %d.'%(len(raw_gold_ants),crt_annotation.file,len(annotations))
            assert False
            # raise Exception('Error! When extracting the annotations using the raw data, we didn\'t get the correct number of antecedents!')

        """ Now that we got the antecedents according to their location/offsets within the raw text files (above),
            we now have to link the RawAntecedents with their corresponding MRG/POS XML file Antecedents. """

        # print 'Len triggers: %d. Len Ants: %d.'%(len(triggers),len(raw_gold_ants))

        mrg_gold_ants = []
        assert len(triggers) == len(raw_gold_ants)
        for i in range(len(raw_gold_ants)):
            raw_ant = raw_gold_ants[i]
            k = len(raw_ant.words)
            while True:
                try:
                    start_idx,end_idx = self.find_word_sequence(raw_ant.words, minimum_match=k)
                    mrg_gold_ants.append(self.idxs_to_ant(start_idx, end_idx, triggers[i], sentnum_modifier))
                    triggers[i].set_antecedent(mrg_gold_ants[-1]) # Set the trigger to match the gold antecedent.
                    break
                except NoSubsequenceFoundException:
                    k -= 1
        return mrg_gold_ants

    def idxs_to_ant(self, start, end, trigger, sentnum_modifier):
        sentdict = self.matrix[start[0]]
        i,j = start[1],end[1]+1
        return Antecedent(start[0]+sentnum_modifier+1, trigger, SubSentDict(sentdict.words[i:j], sentdict.pos[i:j], sentdict.lemmas[i:j]), i, j)

class SubSentDict:
    """
        Antecedents have this.
        @type words: list
        @type pos: list
        @type lemmas: list
    """
    def __init__(self, words, pos, lemmas, start=None, end=None):
        self.words = words
        self.pos = pos
        self.lemmas = lemmas
        self.start,self.end = start,end

    def __eq__(self, other):
        if type(self) is type(other):
            return self.words == other.words and self.pos == other.pos and \
                   self.start == other.start and self.end == other.end
        else:
            raise WrongClassEquivalenceException()

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return str((self.words,self.pos))

class SentDict:
    """ Dictionary for any sentence from a Stanford CoreNLP XML file. Can be modified to have more attributes. """
    def __init__(self, sentence, f_name=None, raw=False, get_deps=False):
        self.file = f_name
        self.sentnum = int(sentence.get('id'))
        self.raw = raw

        if not raw:
            self.tree_text = [tree.text for tree in sentence.iter('parse')]
            self.tree_text[0] = self.fix_quotes_str(self.tree_text[0])

            if len(self.tree_text) != 0:
                self.words = ['ROOT']+[word.text for word in sentence.iter('word')]
                self.lemmas = ['root']+[lemma.text for lemma in sentence.iter('lemma')]
                self.pos = ['root']+[pos.text for pos in sentence.iter('POS')]

                self.fix_quotes(self.words)
                self.fix_quotes(self.lemmas)
                self.fix_quotes(self.pos)

                if get_deps:
                    for dep_type in sentence.iter('dependencies'):
                        if dep_type.get('type') == "collapsed-ccprocessed-dependencies": #this only happens once
                            names = [dep.get('type') for dep in dep_type.iter('dep')]
                            govs = [int(gov.get('idx')) for gov in dep_type.iter('governor')]
                            depends = [int(d.get('idx')) for d in dep_type.iter('dependent')]
                            self.dependencies = [Dependency(names[i], govs[i], depends[i]) for i in range(len(names))]
            else:
                raise EmptySentDictException()
        else:
            self.words = [word.text for word in sentence.iter('word')]
            self.fix_quotes(self.words)

            self.offset_starts = [int(start.text) for start in sentence.iter('CharacterOffsetBegin')]
            self.offset_ends = [int(end.text) for end in sentence.iter('CharacterOffsetEnd')]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        if item == 'pos': return self.pos
        elif item == 'words': return self.words
        elif item == 'lemmas': return self.lemmas
        elif item == 'tree': return self.tree_text

    def __repr__(self):
        return self.words_to_string()

    def fix_quotes(self, lst):
        bads = ['``', "''"]

        for i,string in enumerate(lst):
            if string in bads:
                lst[i] = "\""

    def fix_quotes_str(self, s):
        bads = ['``', "''"]

        news = s
        for bad in bads:
            news = news.replace(bad,"\"")
        return news

    def get_auxiliaries(self, raw=False):
        sent_auxs = []
        for i in range(0,len(self)):
            try:
                if not raw and wc.is_auxiliary(self,i,AUX_LEMMAS,ALL_AUXILIARIES):
                    sent_auxs.append(Auxiliary(self.sentnum, self.words[i], self.lemmas[i], self.pos[i], i))

                elif raw and wc.is_auxiliary(self,i,AUX_LEMMAS,ALL_AUXILIARIES,raw=raw):
                    sent_auxs.append(RawAuxiliary(self.words[i], i, self.sentnum))

            except AuxiliaryHasNoTypeException:
                continue
        return sent_auxs

    def words_to_string(self):
        s = ''
        for w in self.words:
            s += w+' '
        return s

    def print_sentence(self):
        for w in self.words:
            print w,
        print

    def get_section(self):
        return int(self.file[4:6])

    def get_nltk_tree(self):
        return nt.maketree(self.tree_text[0])

    def chunked_dependencies(self, i, j, dep_names=('prep','adv','dobj','nsubj')):
        """Here we will return a list of all relevant dependency clusters between word i and word j."""
        deps = [dep for dep in self.dependencies if ((i <= dep.gov <= j) and (i <= dep.dependent <= j)
                                                                         and dep.name in dep_names)]
        # This makes it so that the gov is always the smaller one.
        for dep in deps:
            if dep.gov > dep.dependent:
                old_gov = dep.gov
                dep.gov = dep.dependent
                dep.dependent = old_gov

        deps.sort(key=attrgetter('gov'))

        # Here we are removing overlap by making earlier dependency chunks end by at most the index of the next start.
        last_start = -1
        last_stop = -1
        for c in range(len(deps)):
            dep = deps[c]

            if not dep.gov > last_stop:
                deps[c-1].dependent = dep.gov-1

            last_stop = dep.dependent

        # Here we get the dependency constituents for each desired dependency in dep_names.
        chunks = []
        for dep in deps:
            k, p = dep.gov, dep.dependent+1
            chunks.append( {'name':dep.name, 'sentdict':SubSentDict( self.words[k:p], self.pos[k:p],
                                                                     self.lemmas[k:p], start=k, end=p)} )

        if len(chunks) == 0:
            chunks.append( {'name':None, 'sentdict':SubSentDict( self.words[i-1:i], self.pos[i-1:i],
                                                                 self.lemmas[i-1:i], start=i-1, end=i)} )

        return chunks

def chunks_to_string(chunk):
    s = ''
    for w in chunk['sentdict'].words:
        s += w+' '
    return s

class Dependency:
    def __init__(self, name, gov, dependent):
        self.name=name
        self.gov=gov
        self.dependent=dependent

    def __repr__(self):
        return '[%s : (%s,%s)]'%(self.name, self.gov, self.dependent)

class Annotations:
    """ A class that contains all of the annotations. It is abstract enough to hold Nielson annotations. """
    def __init__(self):
        self.matrix = []

    def __iter__(self):
        for annotation in self.matrix:
            yield annotation

    def add_section(self, section):
        for annotation in section:
            self.matrix.append(annotation)

class AnnotationSection:
    def __init__(self, subdir, path):
        """ Takes the WSJ subdirectory name and its path to get the Bos & Spenader annotations file. """
        self.matrix = []

        ann_file = subdir[0:2]+'.ann'
        print '\nProcessing the annotations file: %s'%ann_file

        f = open(path+ann_file)
        lines = f.readlines()
        f.close()

        for line in lines:
            self.matrix.append(Annotation(line))

    def __iter__(self):
        for annotation in self.matrix:
            yield annotation

    def file_has_vpe(self, f):
        for annotation in self:
            if f == annotation.file:
                return True
        return False

    def get_anns_for_file(self, f):
        """ Sometimes there are multiple instances of VPE in a file. """
        ret = []
        for annotation in self:
            if annotation.file == f:
                ret.append(annotation)

        return sorted(ret, key=lambda x: x.vpe_offset_start)

class Annotation:
    """ A B&S annotation, should only be created by the AnnotationSection class. """
    def __init__(self, text):
        annotation_text = text.split(' ')
        self.file = annotation_text[0]
        self.vpe_offset_start = int(annotation_text[1])
        self.vpe_offset_end = int(annotation_text[2])
        self.ant_offset_start = int(annotation_text[3])
        self.ant_offset_end = int(annotation_text[4])
        self.trigger_type = annotation_text[5]
        self.vpe_type = annotation_text[6]
        self.pattern = annotation_text[7]
        del annotation_text

    def __repr__(self):
        return "File: %s, VPE Start,end: %d,%d"%(self.file,self.vpe_offset_start,self.vpe_offset_end)

""" ---- Auxiliary and Trigger classes. ---- """
class Auxiliaries:
    def __init__(self):
        self.auxs = []

    def __iter__(self):
        for aux in self.auxs:
            yield aux

    def __len__(self):
        return len(self.auxs)

    def get_aux(self,i):
        return self.auxs[i]

    def get_auxs(self):
        return self.auxs

    def add_aux(self, aux):
        self.auxs.append(aux)

    def add_auxs(self, lst, sentnum_modifier=0):
        try:
            for aux in lst:
                aux.sentnum += sentnum_modifier
                self.auxs.append(aux)
        except TypeError:
            pass

    def print_gold_auxiliaries(self):
        for aux in self:
            if aux.is_trigger:
                print aux

class Auxiliary:
    """ Any auxiliary that is encountered is an Auxiliary object. """
    def __init__(self, sentnum, word, lemma, pos, wordnum):
        self.type = self.get_type(lemma)
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.sentnum = sentnum
        self.wordnum = wordnum
        self.is_trigger = False
        self.gold_ant = None
        self.possible_ants = []

    def __repr__(self):
        if self.is_trigger:
            return 'TRIGGER - Type: %s, Lemma: %s, Word: %s, POS: %s, Sentnum: %d, Wordnum: %s'\
                  %(self.type,self.lemma,self.word,self.pos,self.sentnum,self.wordnum)
        return 'Not Trigger - Type: %s, Lemma: %s, Word: %s, POS: %s, Sentnum: %d, Wordnum: %s'\
                  %(self.type,self.lemma,self.word,self.pos,self.sentnum,self.wordnum)

    def equals(self, aux):
        """ Here we only need to compare the sentnum and wordnum because each auxiliary has a unique combination of these two. """
        return self.sentnum == aux.sentnum and self.wordnum == aux.wordnum
        # return self.wordnum == aux.wordnum and self.type == aux.type and self.word == aux.word and \
        #        self.lemma == aux.lemma and self.pos == aux.pos

    def set_antecedent(self, ant):
        self.gold_ant = ant

    def add_possible_ant(self, ant):
        self.possible_ants.append(ant)

    def get_type(self, lemma):
        if lemma in MODALS: return 'modal'
        elif lemma in BE: return 'be'
        elif lemma in HAVE: return 'have'
        elif lemma in DO: return 'do'
        elif lemma in TO: return 'to'
        elif lemma in SO: return 'so'
        else:
            raise AuxiliaryHasNoTypeException(lemma)


class RawAuxiliary:
    """ Only exists for extracting the annotations from the raw XML files. """
    def __init__(self, word, wordnum, sentnum):
        self.sentnum = sentnum
        self.word = word
        self.wordnum = wordnum

"""" ---- Antecedent classes. ---- """
class Antecedent:
    def __init__(self, sentnum, trigger, sub_sentdict, start, end, section=-1):
        """
            @type trigger: Auxiliary
            @type sub_sentdict: SubSentDict
            @type subtree: nt.nltk.ParentedTree
        """
        self.sentnum = sentnum
        self.subtree = None
        self.trigger = trigger
        self.sub_sentdict = sub_sentdict
        self.section = section
        self.score = None
        self.x = None # features
        self.start = start
        self.end = end

    def __repr__(self):
        ret = ''
        for w in self.sub_sentdict.words:
            ret += w+' '
        return ret

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.sentnum == other.sentnum and self.start == other.start and self.end == other.end

    def contains_trigger(self):
        if self.sentnum != self.trigger.sentnum:
            return False

        # If the antecedent comes after the trigger, delete it.
        if self.start >= self.trigger.wordnum or \
                (self.start <= self.trigger.wordnum <= self.end):
            return True

    def set_score(self, weights):
        self.score = dot(self.x, weights)

    def get_score(self):
        return self.score

    def word_pos_tuples(self):
        return zip(self.sub_sentdict.pos, self.sub_sentdict.words)

    def get_words(self):
        return self.sub_sentdict.words

    def get_head(self, idx=False, idx_in_subsentdict=False):
        """
            @type return: str
        """
        for i in range(len(self.sub_sentdict)):
            if wc.is_verb(self.sub_sentdict.pos[i]) and not wc.is_aux_lemma(self.sub_sentdict.lemmas[i]):
                if idx:
                    if idx_in_subsentdict:
                        return i
                    else:
                        return self.start+i
                else:
                    return self.sub_sentdict.words[i]

        if idx and not idx_in_subsentdict:
            return self.start
        elif idx_in_subsentdict:
            return 0
        else:
            try:
                return self.sub_sentdict.words[0]
            except IndexError:
                return ''

class RawAntecedent:
    """ Only exists for extracting the annotations from the raw XML files. """
    def __init__(self, words, start, end, sentnum):
        self.words = words
        self.start = start
        self.end = end
        self.sentnum = sentnum
