# This is the stage one program. We want to look at a sentence
# and determine whether or not it has VPE.
# @author Kian Kenyon-Dean
# @date May 27, 2015

import xml.etree.ElementTree as ET
from os import listdir
import time
from sys import argv
import nltktree


timestart = time.time()
### Directory Globals ###
# Mac
# FULL_PROJECT_DIR = '/Users/kian/Documents/HONOR/'
#
# XML_ANNOTATIONS_DIR = FULL_PROJECT_DIR+'xml_annotations/'
# XML_MRG = XML_ANNOTATIONS_DIR+'wsj_mrg_good_extraction/'
# XML_POS = XML_ANNOTATIONS_DIR+'wsj_pos/'
# XML_RAW_TOKENIZED = XML_ANNOTATIONS_DIR+'tokenized_raw/'
# VPE_ANNOTATIONS = FULL_PROJECT_DIR+'vpe_annotations/wsj/'
#
# LOG_LOCATIONS = FULL_PROJECT_DIR+'project/logs/'
# SLASH_CHAR = '/'

# PC
# FULL_PROJECT_DIR = 'C:\\Users\\Kian\\Sura\\'
#
# XML_ANNOTATIONS_DIR = FULL_PROJECT_DIR+'xml_annotations\\'
# XML_MRG = XML_ANNOTATIONS_DIR+'wsj_mrg\\'
# XML_POS = XML_ANNOTATIONS_DIR+'wsj_pos\\'
# XML_RAW_TOKENIZED = XML_ANNOTATIONS_DIR+'tokenized_raw\\'
# VPE_ANNOTATIONS = FULL_PROJECT_DIR+'vpe_annotations\\wsj\\'
#
# LOG_LOCATIONS = FULL_PROJECT_DIR+'project\\logs\\'
# SLASH_CHAR = '\\'

### ----------------- ###

### English Auxiliaries that trigger VPE ###

#VPE_TRIGGERS_IN_WSJ = ['do','be','have','can','could','may','must','might','will','would','shall','should']

MODALS = ['can','could','may','must','might','will','would','shall','should']
BE     = ['be']
HAVE   = ['have']
DO     = ['do']
TO     = ['to']
SO     = ['so','same','likewise','opposite']

ALL_CATEGORIES = [MODALS, BE, HAVE, DO, TO, SO]

VPE_TRIGGERS_IN_WSJ = MODALS+BE+HAVE+DO+TO+SO

### ----------------- ###


def isauxiliary(sentdict, auxidx):
    return (sentdict['lemmas'][auxidx] in VPE_TRIGGERS_IN_WSJ) and (isverb(sentdict['pos'][auxidx]) or (sentdict['lemmas'][auxidx] in TO+SO))

# This will take a single xml file, i.e. wsj_0001.mrg.xml, with its path
# It will return a matrix of sentence vectors.
# noinspection PyRedundantParentheses
def getdataMRG(xml_file,path):
    print 'Processing: %s' %(xml_file)
    parser = ET.XMLParser()
    tree = ET.parse(path+xml_file,parser=parser)
    root = tree.getroot()

    # For each <sentence> in the file...
    sentmatrix = []
    for sentence in root.iter('sentence'):
        sentparsetree = [tree.text for tree in sentence.iter('parse')]
        if len(sentparsetree) != 0: # For some reason it adds a lot of blank dictionaries.. Quickfix.
            sentwords     = ['ROOT']+[word.text for word in sentence.iter('word')]
            sentlemmas    = ['root']+[lemma.text for lemma in sentence.iter('lemma')]
            sentpostags   = ['root']+[pos.text for pos in sentence.iter('POS')]
            sentner       = ['root']+[ner.text for ner in sentence.iter('NER')]
            sentwordnums  = range(0,len(sentwords))
            sentdepnames  = []
            sentdepgovs   = []
            sentdepdeps   = []
            # for dep_type in sentence.iter('dependencies'):
            #     if dep_type.get('type') == "collapsed-ccprocessed-dependencies": #this only happens once
            #         sentdepnames = [dep.get('type') for dep in dep_type.iter('dep')]
            #         sentdepgovs  = [int(gov.get('idx')) for gov in dep_type.iter('governor')]
            #         sentdepdeps  = [int(d.get('idx')) for d in dep_type.iter('dependent')]

            sentdict = {'words': sentwords, 'lemmas': sentlemmas, 'pos': sentpostags, 'nums': sentwordnums,
                        'tree': sentparsetree, 'ner': sentner, 'dependencies': sentdepnames, 'govs': sentdepgovs,
                        'dependents': sentdepdeps}

            sentmatrix.append(sentdict)

    return sentmatrix

# Method to check if a word (based on its number in the sentence)
# is a governor in the dependency passed to it.
def isgovernor(wordnum, depnum, sentdict):
    return sentdict['nums'][sentdict['govs'][depnum]] == wordnum

# Check if a word is a dependent in the dependency passed to it.
def isdependent(wordnum, depnum, sentdict):
    return sentdict['nums'][sentdict['dependents'][depnum]] == wordnum

def isverb(wordpostag):
    verbtags = ['VB','VBD','VBG','VBN','VBP','VBZ','MD']
    return wordpostag in verbtags

def isnoun(tag):
    return tag in ['NN','NNS','NNP','NNPS','WP','PRP','PRP$']

def ispredicative(wordpostag):
    return wordpostag.endswith('PRD')

def isnounorprep(wordpostag):
    nountags = ['NN','NNS','NNP','NNPS','WP','PRP','PRP$','DT']
    return wordpostag in nountags

def isperiod(wordpostag):
    puncttags = ['.']
    return wordpostag in puncttags

def iscomma(wordpostag):
    return wordpostag == ','

def isdashorcolon(wordpostag):
    return wordpostag == ':'

def isconj(wordpostag):
    conjtags = ['CC']
    return wordpostag in conjtags

def isprep(wordpostag):
    preptags= ['IN']
    return wordpostag in preptags

def isadj(wordpostag):
    adjtags = ['JJ','JJR','JJS']
    return wordpostag in adjtags

def isdate(wordner):
    datenames = ['DATE']
    return wordner in datenames

def isquote(wordpostag):
    quotetags = ['"','\'']
    return wordpostag in quotetags

def ispunctuation(tag):
    return tag in ['\"', '\'', '.', ',', ':', ';','-','--']

def thesamecheck(sentence, auxidx):
    if len(sentence) > auxidx+2:
        return sentence[auxidx+1].lower() == 'the' and sentence[auxidx+2].lower() == 'same'
    return False

# Only use the preposition check rule if the auxiliary is a 'do'
def prepcheck(sentdict, idx):
    global DO, HAVE
    lst = DO
    return sentdict['words'][idx].lower() in lst

def islastword(sentdict, idx):
    return idx == getlastwordidx(sentdict)

def issecondtolastword(sentdict, idx):
    return idx == getlastwordidx(sentdict) - 1

def getlastwordidx(sentdict):
    i = len(sentdict['pos'])-1
    while not sentdict['pos'][i] == '.'and (not i == len(sentdict['words'])):
        i -= 1

    if sentdict['words'][i] == 'S': # Hack, need to get rid of all the fake words...
        i -= 1

    return i-1 # The last word is the one before the period

def geteachverbidx(sentdict):
    ret = []
    for i in range(0,len(sentdict['words'])):
        if isverb(sentdict['pos'][i]):
            ret.append(i)
    return ret

def getsurroundingstruct(sentdict, idx, preorpost, limit):
    words  = []
    lemmas = []
    pos    = []

    if preorpost == 'pre':
        for i in range(idx-limit, idx):
            try:
                words.append(sentdict['words'][i])
                lemmas.append(sentdict['lemmas'][i])
                pos.append(sentdict['pos'][i])
            except IndexError: continue

    elif preorpost == 'post':
        for i in range(idx, idx+limit):
            try:
                words.append(sentdict['words'][i])
                lemmas.append(sentdict['lemmas'][i])
                pos.append(sentdict['pos'][i])
            except IndexError: continue

    for w in words:
        if not w:
            w = w.lower()

    return {'words':words, 'lemmas':lemmas, 'pos':pos}

def equalitycheck(auxdict, vbdict, section=None, lax=None):

    def checksection(auxdict, vbdict, sect, lax):
        if not lax:
            cont = True
            for auxtoken,vbtoken in zip(auxdict[sect], vbdict[sect]):
                if cont:
                    if auxtoken!=vbtoken: cont = False
            return cont
        if lax:
            for auxtoken,vbtoken in zip(auxdict[sect], vbdict[sect]):
                if auxtoken==vbtoken: return True
            return False


    ret = False
    if not section:
        ret = checksection(auxdict, vbdict, 'words', lax)
        if ret: return ret
        ret = checksection(auxdict, vbdict, 'lemmas', lax)
        if ret: return ret
        ret = checksection(auxdict, vbdict, 'pos', lax)
        return ret
    else:
        return checksection(auxdict, vbdict, section)


# Methods for dealing with different types of auxiliaries:

def auxparallelism(sentdict, auxidx, preorpost, limit, lax=None):
    auxstruct = getsurroundingstruct(sentdict, auxidx, preorpost, limit)
    for verbidx in geteachverbidx(sentdict):
        if verbidx != auxidx:
            vbstruct = getsurroundingstruct(sentdict, verbidx, preorpost, limit)
            if equalitycheck(auxstruct, vbstruct, section=None, lax=lax):
                return True
    return False

# noinspection PyChainedComparisons
def isccommandedbycontinuationword(sentdict, auxidx, t, word_positions_in_tree):
    continuation_words = ['than', 'as', 'so']
    local_continuation_words = ['like']

    for i in range(0,len(sentdict['words'])):
        crtword = sentdict['words'][i].lower()
        if crtword in continuation_words and i >= auxidx - 5  and i < auxidx:
            if nltktree.ccommands(t, t[word_positions_in_tree[i-1]], t[word_positions_in_tree[auxidx-1]]):
                return True

    # localt = nltktree.generatelocalstructurefromsubtree(t, t[word_positions_in_tree[auxidx-1]])
    # local_word_subtrees = nltktree.getsmallestsubtrees(localt)
    # Checking for local c-command.
    # for subtree in local_word_subtrees:
    #     if subtree.leaves() in local_continuation_words:
    #         if nltktree.ccommands(localt, subtree, t[word_positions_in_tree[auxidx-1]]):
    #             return True

    return False

# Returns true if the auxiliary is c-commanded by a verb.
def auxisccommandedbyverb(sentdict, auxidx, t, word_positions_in_tree):
    subtrees = nltktree.getsmallestsubtrees(t)

    for subtree in subtrees:
        if isverb(subtree.label()):
            try:
                if nltktree.ccommands(t, subtree,  t[word_positions_in_tree[auxidx-1]]):
                    return True
            except IndexError:
                pass
    return False

def auxccommandsverb(sentdict, auxidx, t, word_positions_in_tree):
    subtrees = nltktree.getsmallestsubtrees(t)

    for subtree in subtrees:
        if isverb(subtree.label()):
            try:
                if nltktree.ccommands(t, t[word_positions_in_tree[auxidx-1]], subtree):
                    return True
            except IndexError:
                pass
    return False

# Returns true if the auxiliary is ASYMETRICALLY c-commanded by a verb in the local structure.
def auxislocallyccommandedbyverb(sentdict, auxidx, t, word_positions_in_tree):
    try:
        localt = nltktree.generatelocalstructurefromsubtree(t, t[word_positions_in_tree[auxidx-1]])
        local_word_subtrees = nltktree.getsmallestsubtrees(localt)

        for subtree in local_word_subtrees:
            if isverb(subtree.label()):
                if nltktree.ccommands(localt, subtree, t[word_positions_in_tree[auxidx-1]])\
                        and not nltktree.ccommands(localt, t[word_positions_in_tree[auxidx-1]], subtree):
                    return True
    except IndexError: pass
    return False

# Returns true if the auxiliary ASYMETRICALLY c-commands a verb.
def auxlocallyccommandsverb(sentdict, auxidx, t, word_positions_in_tree):
    try:
        localt = nltktree.generatelocalstructurefromsubtree(t, t[word_positions_in_tree[auxidx-1]])
        local_word_subtrees = nltktree.getsmallestsubtrees(localt)

        for subtree in local_word_subtrees:
            if isverb(subtree.label()):
                if nltktree.ccommands(localt, t[word_positions_in_tree[auxidx-1]], subtree)\
                        and not nltktree.ccommands(localt, subtree, t[word_positions_in_tree[auxidx-1]]):
                    return True
    except IndexError: pass
    return False

def auxccommandsverbthatcomesafter(sentdict, auxidx, tree, word_positions_in_tree):
    for i in range(auxidx+1, len(sentdict['pos'])):
        if isverb(sentdict['pos'][i]):
            if nltktree.ccommands(tree, tree[word_positions_in_tree[auxidx-1]], tree[word_positions_in_tree[i-1]]):
                return True
    return False

def countcommas(sentdict):
    count = 0
    for w in sentdict['words']:
        if w == ',': count+=1
    return count

def isfollowedbypunct(sentdict, auxidx, end=None):
    try:
        if not end:
            checkpuncttag = sentdict['pos'][auxidx+1]
            if isperiod(checkpuncttag) or iscomma(checkpuncttag) or isdashorcolon(checkpuncttag):
                return True

            try:
                checkpuncttag = sentdict['pos'][auxidx+2]
                if sentdict['lemmas'][auxidx+1] == 'not' and (isperiod(checkpuncttag) or iscomma(checkpuncttag) or isdashorcolon(checkpuncttag)):
                    return True
            except IndexError: pass
        else:
            checkpuncttag = sentdict['pos'][auxidx+1]
            if checkpuncttag in end: return True

            try:
                checkpuncttag = sentdict['pos'][auxidx+2]
                if sentdict['lemmas'][auxidx+1] == 'not' and checkpuncttag in end:
                    return True
            except IndexError: pass
    except IndexError:
        return False

    return False

def verbfollowsaux(sentdict, auxidx):
    for i in range(auxidx+1,len(sentdict['pos'])):
        if isverb(sentdict['pos'][i]):
            return True
    return False

# Old check - only uses the POS tagged sentence.
def auxiliarycheck(sentdict, auxidx, verbose):
    # The last word will always be a '.', so we have to go 1 before that.
    if verbose: print 'Word: '+sentdict['words'][auxidx]+'...',

    if islastword(sentdict, auxidx):
        return True

    # Here we are primitively checking what is dominated by the auxiliary.
    for i in range(auxidx+1, len(sentdict['pos'])):

        # This means that we hit punctuation before a noun, so there's no domination.
        crttag = sentdict['pos'][i]
        if verbose: print crttag+' ',
        if isperiod(crttag) or isconj(crttag) or (isprep(crttag) and prepcheck(sentdict,auxidx)):
            return True
        if (isnounorprep(crttag) and not isdate(sentdict['ner'][i])) or isverb(crttag) or isadj(crttag):
            return False

    if verbose: print '---No escape---.'
    return False


def nexttopunct(sentdict, auxidx, t, word_positions_in_tree):
    localt = nltktree.generatelocalstructurefromsubtree(t, t[word_positions_in_tree[auxidx-1]])
    local_word_subtrees = nltktree.getsmallestsubtrees(localt)
    try:
        checkpuncttag = sentdict['pos'][auxidx+1]
        if isperiod(checkpuncttag) or iscomma(checkpuncttag) or isdashorcolon(checkpuncttag):
            endbool = True

            # for subtree in local_word_subtrees:
            #     if isverb( subtree.label() ) and subtree != t[word_positions_in_tree[auxidx-1]]:
            #         if nltktree.ccommands(localt, subtree, t[word_positions_in_tree[auxidx-1]]):
            #             endbool = False
            #             break

            if endbool: return endbool
    except IndexError:
        return False

    try:
        checkpuncttag = sentdict['pos'][auxidx+2]
        if sentdict['lemmas'][auxidx+1] == 'not' and (isperiod(checkpuncttag) or iscomma(checkpuncttag) or isdashorcolon(checkpuncttag)):
            endbool = True

            # for subtree in local_word_subtrees:
            #     if isverb( subtree.label() ):
            #         if nltktree.ccommands(localt, subtree, t[word_positions_in_tree[auxidx-1]]):
            #             endbool = False
            #             break

            if endbool: return endbool
    except IndexError:
        return False

    return False

def modalcheck(sentdict, auxidx, tree, word_positions_in_tree, verbose=False):
    if toprecedesaux(sentdict, auxidx):
        return False
    if auxccommandsverbthatcomesafter(sentdict,auxidx,tree,word_positions_in_tree):
        return False
    return True

# TODO: make this better.   BE  BE  BE  BE  B E BEBEBEBEBEBEBEBEBEBBE
def becheck(sentdict, auxidx, tree, word_positions_in_tree ,verbose=False):
    badwords = ['being','been']
    endpunct = [',','.','--',':']

    if toprecedesaux(sentdict, auxidx):
        return False
    if sentdict['words'][auxidx-1].lower() == 'that' and sentdict['words'][auxidx].lower() == 'is':
        return False
    if isfollowedbypunct(sentdict, auxidx, end=['.']) and not sentdict['words'][auxidx] in badwords:
        return True

    # for i in range(auxidx+1, len(sentdict['pos'])):
    #     if sentdict['pos'][i] in endpunct:
    #         break
    #     if isverb(sentdict['pos'][i]) or isadj(sentdict['pos'][i]) or isnounorprep(sentdict['pos'][i]):
    #         return False

    return False

# TODO: write function to get the local S or SBAR phrase around the aux. DONE.
# noinspection PyChainedComparisons
def docheck(sentdict, auxidx, t, word_positions_in_tree, verbose=False):

    # We DO NOT want to consider 'do so' or 'do the same' sentences here!
    """try:
        if sentdict['lemmas'][auxidx+1] == 'so' or (sentdict['lemmas'][auxidx+1] == 'the' and sentdict['lemmas'][auxidx+2] == 'same'):
            return False
        # We can be POSITIVE that there is NO vpe if we have 'don't do ...' or 'x does do ...'
        if sentdict['lemmas'][auxidx-1] == 'do' or (sentdict['lemmas'][auxidx-2] == 'do' and sentdict['lemmas'][auxidx-1] == 'not'):
            return False
        if sentdict['lemmas'][auxidx+1] == 'do' or (sentdict['lemmas'][auxidx+1] == 'do' and sentdict['lemmas'][auxidx+2] == 'not'):
            return False
    except IndexError: pass"""

    try:
        if sentdict['lemmas'][auxidx+1] == 'that':
            return True
    except IndexError: pass

    if not auxislocallyccommandedbyverb(sentdict, auxidx, t, word_positions_in_tree):

        # # If 'do' locally c-commands a verb AND is locally c-commanded by a verb, we can be basically 100% sure that there is no VPE.
        # if auxislocallyccommandedbyverb(sentdict, auxidx, tree, word_positions_in_tree) and auxlocallyccommandsverb(sentdict, auxidx, tree, word_positions_in_tree):
        #     return False

        if toprecedesaux(sentdict, auxidx): return False

        localt = nltktree.generatelocalstructurefromsubtree(t, t[word_positions_in_tree[auxidx-1]])
        local_word_subtrees = nltktree.getsmallestsubtrees(localt)

        # Do at the end of sentence.
        checkpuncttag = sentdict['pos'][auxidx+1]
        if isperiod(checkpuncttag) or iscomma(checkpuncttag) or isdashorcolon(checkpuncttag):
            endbool = True

            for subtree in local_word_subtrees:
                if isverb( subtree.label() ) and subtree != t[word_positions_in_tree[auxidx-1]]:
                    if nltktree.ccommands(localt, subtree, t[word_positions_in_tree[auxidx-1]]):
                        endbool = False
                        break

            if endbool: return endbool

        # Don't at the end of sentence.
        try:
            checkpuncttag = sentdict['pos'][auxidx+2]
            if sentdict['lemmas'][auxidx+1] == 'not' and (isperiod(checkpuncttag) or iscomma(checkpuncttag) or isdashorcolon(checkpuncttag)):
                endbool = True

                for subtree in local_word_subtrees:
                    if isverb( subtree.label() ):
                        if nltktree.ccommands(localt, subtree, t[word_positions_in_tree[auxidx-1]]):
                            endbool = False
                            break

                if endbool: return endbool
        except IndexError:
            pass

        # Small increase in recall, decrease in precision from this.
        # numverbs = 0
        # for subtree in local_word_subtrees:
        #     if isverb(subtree.label()) or isnounorprep(subtree.label()):
        #         numverbs+=1
        # if numverbs == 1:
        #     return True

        if isccommandedbycontinuationword(sentdict ,auxidx, t, word_positions_in_tree):
            #if not auxlocallyccommandsverb(sentdict ,auxidx, tree, word_positions_in_tree): # 8% recall traded for 4% precision.
            #if (not isverb(sentdict['pos'][auxidx+1])) or (sentdict['lemmas'][auxidx+1]=='not' and not isverb(sentdict['pos'][auxidx+2])):
            return True

        # if not auxccommandsverbthatcomesafter(sentdict ,auxidx, tree, word_positions_in_tree):
        #     return True

        if verbfollowsaux(sentdict, auxidx):
            return False

        if isprep(sentdict['pos'][auxidx+1]) and sentdict['words'][auxidx] != 'done':
            return True

    return False

def havecheck(sentdict, auxidx, tree, word_positions_in_tree, verbose=False):
    global VPE_TRIGGERS_IN_WSJ

    if toprecedesaux(sentdict, auxidx):
        return False
    try:
        if sentdict['words'][auxidx+1] == 'a' or sentdict['words'][auxidx+1]=='to':
            return False
    except IndexError: pass

    if isfollowedbypunct(sentdict,auxidx):
        return True


    # auxidxs_in_sent = [i for i in range(0,len(sentdict['words'])) if sentdict['lemmas'][i] in VPE_TRIGGERS_IN_WSJ and i != auxidx]
    #
    # subtrees = nltktree.getsmallestsubtrees(t)
    # subtree_positions = nltktree.getsmallestsubtreepositions(t, subtree_list=subtrees)
    # localt   = nltktree.generatelocalstructurefromsubtree(t,subtree_positions[auxidx-1])

    # if auxislocallyccommandedbyverb(sentdict, auxidx, tree, word_positions_in_tree):
    #     return True

    return False
    # return isfollowedbypunct(sentdict, auxidx)
    #previouswordisasorsoorthan(sentdict['words'], auxidx)

# For to, we are just checking if what follows it is a punctuation, if this is the case, return true.
def tocheck(sentdict, auxidx, tree, word_positions_in_tree, verbose=False):

    speakinglemmas = ['say','said','says','acknowledge','acknowledged']

    try:
        if isperiod(sentdict['pos'][auxidx+1]): return True
    except IndexError: print sentdict['words']
    if len(sentdict['words']) > auxidx+3:
        # print 'Following to: '+sentdict['words'][auxidx+1] + ' ' + sentdict['lemmas'][auxidx+2]
        if sentdict['words'][auxidx+1] == ',':# and isquote(sentdict['lemmas'][auxidx+2]):
            for i in range(auxidx+2,len(sentdict['words'])):
                if sentdict['lemmas'][i] in speakinglemmas:
                    return True
    return False

def socheck(sentdict, auxidx, tree, word_positions_in_tree, verbose=False):
    if toprecedesaux(sentdict, auxidx):
        return False

    # 'do so/likewise'
    if sentdict['lemmas'][auxidx-1] == 'do' or sentdict['words'][auxidx-1] == 'be':
        if not isadj(sentdict['pos'][auxidx+1]):
            return True

    # 'do the same/opposite'
    if sentdict['lemmas'][auxidx-1] == 'the' and (sentdict['lemmas'][auxidx-2] == 'do' or sentdict['words'][auxidx-1] == 'be'):
        if not isnounorprep(sentdict['pos'][auxidx+1]):
            return True

    # if not isccommandedbycontinuationword(sentdict, auxidx, tree, word_positions_in_tree):
    #     if auxisccommandedbyverb(sentdict, auxidx, tree, word_positions_in_tree):
    #         # if not auxislocallyccommandedbyverb(sentdict, auxidx, tree, word_positions_in_tree):
    #         return auxccommandsverb(sentdict, auxidx, tree, word_positions_in_tree)

    return False

# Checks if the previous word is 'as' or 'so'. This is because we may have sentences like
# "So did Mr. Bush." where Mr. Bush is a direct object, but the 'so' signifies that we aren't
# dealing with a meaningful verb.
def previouswordisasorsoorthan(sentence, auxidx):
    if auxidx!=0:
        check = sentence[auxidx-1].lower()
        if check == 'so' or check =='as' or check == 'than':
            return True
    return False

def nextwordistoo(sentdict, auxidx):
    try:
        check = auxidx + 1
        if sentdict['lemmas'][check] == ',': check += 1
        return sentdict['lemmas'][check] == 'too'
    except IndexError: return False

def toprecedesaux(sentdict, auxidx):
    return sentdict['lemmas'][auxidx-1] == 'to'

# MAIN ALGORITHM. ----------------------------------------------------------------------------------------------------------
# Function that takes in a sentence dictionary and outputs if it has VPE. 
# TODO: add rules to this to determine VPE.
def myalghasvpe(sentdict, SENTENCENUMBER, tree, word_positions_in_tree, verbose, test_aux=None):
    global VPE_TRIGGERS_IN_WSJ, MODALS, BE, DO, HAVE, TO, SO
    # print 'TESTING SENTENCE NUMBER %d:'%SENTENCENUMBER
    # First test: if there is no auxiliary, there is no VPE.
    # Else, we will create a list containing all of the auxiliaries
    # by their corresponding word number in the sentence.
    modalidxs = []
    beidxs    = []
    doidxs    = []
    haveidxs  = []
    toidxs    = []
    soidxs    = []

    for i in sentdict['nums']: # Go through each index.
        if   sentdict['lemmas'][i] in MODALS and sentdict['pos'][i] == 'MD': modalidxs.append(i)
        elif sentdict['lemmas'][i] in BE:     beidxs.append(i)
        elif sentdict['lemmas'][i] in DO:     doidxs.append(i)
        elif sentdict['lemmas'][i] in HAVE:   haveidxs.append(i)
        elif sentdict['lemmas'][i] in TO:     toidxs.append(i)
        elif sentdict['lemmas'][i] in SO:     soidxs.append(i)

    auxiliaryidxs = modalidxs+beidxs+doidxs+haveidxs+toidxs+soidxs

    # If there is no auxiliary, there is no VPE.
    if not auxiliaryidxs:
        return [False,'']

    # Initialize the return values.
    auxbools = {}
    for auxidx in auxiliaryidxs:
        auxbools[auxidx] = False # auxparallelism(sentdict, auxidx, 'pre', 1, lax=None)


    if test_aux=='modal' or not test_aux:
        for modalidx in modalidxs: auxbools[modalidx] = modalcheck(sentdict, modalidx, tree, word_positions_in_tree, verbose)
    if test_aux=='be' or not test_aux:
        for beidx in beidxs: auxbools[beidx]          = becheck(sentdict, beidx, tree, word_positions_in_tree, verbose)
    if test_aux=='do' or not test_aux:
        for doidx in doidxs: auxbools[doidx]          = docheck(sentdict, doidx, tree, word_positions_in_tree, verbose)
    if test_aux=='have' or not test_aux:
        for haveidx in haveidxs: auxbools[haveidx]    = havecheck(sentdict, haveidx, tree, word_positions_in_tree, verbose)
    if test_aux=='to' or not test_aux:
        for toidx in toidxs: auxbools[toidx]          = tocheck(sentdict, toidx, tree, word_positions_in_tree, verbose)
    if test_aux=='so' or not test_aux:
        for soidx in soidxs: auxbools[soidx]          = socheck(sentdict, soidx, tree, word_positions_in_tree, verbose)


    true_cases = []
    for k in auxbools:
        if auxbools[k]:
            #print 'Sentence %d YES VPE.' %SENTENCENUMBER
            aux = sentdict['lemmas'][k]
            if aux in MODALS: aux = 'modal'
            if aux in SO: aux = 'so'
            true_cases.append([True,(auxidx, aux)])

    if len(true_cases) == 1:
        return true_cases[0]

    elif true_cases:
        return {'multiple':true_cases}

    #print 'Sentence %d NO VPE.' %SENTENCENUMBER
    return [False,'']
# MAIN ALGORITHM END -------------------------------------------------------------------------------------------------------	 

# This gets the data from the .ann file for a subdirectory which says where the VPEs are.
# There are 25 .ann files, 00.ann to 24.ann.
def getvpeannotationsdata(subdir, ann_path):
    # First: read the .ann file, this will get passed a dir 'XX/'
    ann_file = subdir[0:2]+'.ann'
    print '\nProcessing the annotations file: %s' %ann_file
    ann      = open(ann_path+ann_file)
    lines    = ann.readlines()
    ann.close()

    annotationmatrix = []
    for line in lines:
        annotation = line.split(' ')
        anndict = {'file': annotation[0], 'vpeoffsetstart': annotation[1], 'vpeoffsetend': annotation[2],
                   'antoffsetstart': annotation[3], 'antoffsetend': annotation[4], 'trigger': annotation[5],
                   'vpetype': annotation[6], 'pattern': annotation[7]}

        annotationmatrix.append(anndict)

    # for d in annotationmatrix:
    #     if d['file'] == 'wsj_0909':
    #         print d
    return annotationmatrix

def filehasvpe(xml_file,anndata):
    extensionlessfile = xml_file[0:8]

    for i in range(0,len(anndata)):
        if extensionlessfile == anndata[i]['file']:
            return True

    return False

# Check if a file has VPE, based on the annotations data matrix of files.
# Return [] if there is no VPE in the file, else return the indexes of
# the anndata list in which the file is referenced. WORKS.
def filevpelocations(xml_file, anndata):
    extensionlessfile = xml_file[0:8]
    #print 'Checking if file has VPE: %s' %extensionlessfile
    idxs     = []
    triggers = []
    for i in range(0,len(anndata)):
        if extensionlessfile == anndata[i]['file']:
            idxs.append(i)
            triggers.append(anndata[i]['trigger'])
    return [idxs,triggers]

# This will check if a sentence has VPE, according to the annotation data.
# It does this by matching the charoffsets in the tokenized_raw file. This
# file always has one more sentence than the regular files. This will return
# a list of the sentence numbers which have VPE in the mrg_xml_file. 
def goldstandardVPEsentences(xml_f, xml_raw_tok_path, subdir, anndata, mrgmatrix, get_words_as_sections=True, get_trig_patterns=False):
    global VPE_TRIGGERS_IN_WSJ,MODALS,SO

    VPEsentencenumbers = []
    VPEtriggerwords = [] # This contains tuples, (wordidx, auxiliary trigger)

    xml_file = xml_f[0:8]+'.xml'

    tmp = filevpelocations(xml_file, anndata)
    matrixfileidxs = tmp[0]
    triggers = tmp[1]

    if get_words_as_sections:
        for i in range(0,len(triggers)):
            if triggers[i] in MODALS: triggers[i] = 'modal'
            elif triggers[i] in SO: triggers[i] = 'so'

    if matrixfileidxs:
        sentencemap = maprawtomrg(xml_f, xml_raw_tok_path, subdir, mrgmatrix)

        parser = ET.XMLParser()
        tokenizedtree = ET.parse(xml_raw_tok_path + subdir + xml_file, parser=parser)
        root = tokenizedtree.getroot()

        for sentence in root.iter('sentence'):
            rawidx = int(sentence.get('id'))
            for token in sentence.iter('token'):
                for offsetstart in token.iter('CharacterOffsetBegin'):
                    # This for loop below goes through each dictionary corresponding to
                    # the file because sometimes a file has multiple instances of VPE.
                    # This is inefficient and should be changed, but it isn't too big of a deal.
                    for idx in matrixfileidxs:
                        if offsetstart.text == anndata[idx]['vpeoffsetstart']:
                            try:
                                VPEsentencenumbers.append(sentencemap[rawidx])
                                word = [w.text for w in token.iter('word')][0]

                                if not get_trig_patterns:
                                    VPEtriggerwords.append( (int(token.get('id')), word) )
                                else:
                                    VPEtriggerwords.append( (int(token.get('id')), word, anndata[idx]['pattern']) )

                            except KeyError:
                                print '------------ KEY ERROR -------------'
                                for word in sentence.iter('word'):
                                    print word.text,
                                print
                                print '---THE XML MAY NEED TO BE FIXED!!!!!---'
    # print VPEsentencenumbers
    # print triggers
    if len(VPEsentencenumbers) != len(set(VPEsentencenumbers)):
        print 'WARNING: MULTIPLE VPE IN A SENTENCE.'
        for i in range(0,len(VPEsentencenumbers)):
            if VPEsentencenumbers[i] in VPEsentencenumbers[i+1:len(VPEsentencenumbers)]:
                printsent(mrgmatrix, VPEsentencenumbers[i])
        print VPEsentencenumbers
        print triggers
    if len(VPEsentencenumbers) != len(triggers): print 'WARNING - LENGTH OF GOLD STANDARD TRIGGERS != LENGTH OF SENTS WITH VPE'
    return dict(zip(VPEsentencenumbers,triggers)),VPEtriggerwords

def printsent(mrgmatrix,sentnum):
    print '[',
    for word in mrgmatrix[sentnum]['words']:
        if word!='ROOT':
            print word,
    print ']'

# This get each sentence from the tokenized_raw xml and map each one to its
# corresponding sentence in the actual mrg xml. This is necessary because some
# of the sentences are not directly aligned as there may be slight parsing differences
# in the tokenized_raw versus the xml.
def maprawtomrg(xml_f, xml_raw_tok_path, subdir, mrgmatrix):
    xml_file = xml_f[0:8]+'.xml'

    sentencemap = {} # The keys will be the MRG sentence numbers, the values are the tok_raws.

    parser = ET.XMLParser()
    tokenizedtree = ET.parse(xml_raw_tok_path+subdir+xml_file,parser=parser)
    root = tokenizedtree.getroot()

    sents = []
    allwords = []

    for sentence in root.iter('sentence'):
        sentidx = int(sentence.get('id'))
        sents.append(sentidx)
        for word in sentence.iter('word'):
            if word.text != '.' and word.text!='START':
                allwords.append(word.text)

    del parser
    del tokenizedtree

    uniquewords = {}

    # Here we are grabbing a unique word from each sentence (this is slow but necessary).
    # This unique word will be used to distinguish the sentence
    # from each other sentence. It will also be used as a mapping
    # from the MRGs to the raw_tokenized xml files.
    # Returns a dict = {rawsentnum : mrgsentnum, ... }
    nonuniquesents = {}

    for sentence in root.iter('sentence'):
        sentidx = int(sentence.get('id'))
        hasunique = False
        if sentidx != 1: # This is the '.START' sentence
            for word in sentence.iter('word'):
                if allwords.count(word.text) == 1:
                    hasunique = True

                    # Here we are fixing the fractions by removing the space.
                    w = word.text
                    neww = ''
                    if '/' in w:
                        for char in w:
                            if ord(char) != 160: #I have no idea why the ascii code is 160...
                                neww+=char
                    else:
                        neww = w

                    uniquewords[neww] = {'raw':sentidx, 'mrg':-1}
                    break

            if not hasunique:
                # print '--------------------------'
                # print 'WARNING: this sentence has no unique word:'
                # print '[',
                # for word in sentence.iter('word'):
                # 	print word.text,
                # print ']'
                sent = []
                for word in sentence.iter('word'):
                    sent.append(word.text)
                nonuniquesents[sentidx] = sent

    # for w in uniquewords: print w+' ',
    # print
    # print '    %d sentences in this file with no identifying words.' %(len(nonuniquesents))
    # print '    These are the non-unique sentences:'
    # print '--------------------------------'
    # for s in nonuniquesents:
    #     print nonuniquesents[s]

    # Here we are mapping the raw to the mrgs. The initially unmapped things are non-unique sentences.
    for i in range(0,len(mrgmatrix)):
        fixed  = False
        mapped = False
        for word in mrgmatrix[i]['words']:
            if word in uniquewords:
                uniquewords[word]['mrg'] = i
                mapped = True
                fixed = True
                break # Out of this sentence.


        # This is to fix sentences with non-unique words, like "They didn't."
        if not mapped:
            for sentnum in nonuniquesents:
                # print '--------------------------------'
                # print 'This is the non-unique sentence: '
                # print nonuniquesents[sentnum]
                # print 'Comparing this sentence with: ',
                # print mrgmatrix[i]['words']
                if len(nonuniquesents[sentnum]) == len(mrgmatrix[i]['words']) - 1: #-1 for the added 'ROOT' word.
                    fail = False
                    for word in mrgmatrix[i]['words']:
                        if word not in nonuniquesents[sentnum] and word!='ROOT':
                            fail = True

                    if not fail:
                        # print 'Fixed: mapped the full sentence.'
                        uniquewords['fullsentence%d'%i] = {'raw':sentnum, 'mrg':i}
                        fixed = True
                        # print {'raw':sentnum, 'mrg':i}
                        # printsent(mrgmatrix, i)
                        # print '--------------------------'
                else: # This is to map sentences that are partial, i.e. "John went to the store. He Laughed." Might be one sentence in the other XML
                    can_resolve = True
                    for word in nonuniquesents[sentnum]:
                        if not word in mrgmatrix[i]:
                            can_resolve = False
                    if can_resolve:
                        uniquewords['fullsentence%d'%i] = {'raw':sentnum, 'mrg':i}
                # else: print 'ERROR: could not do a word-to-word sentence mapping.'

    sentencemap = {} # raw sent numbers are the keys to the corresponding mrg sent numbers.
    for k in uniquewords:
        if uniquewords[k]['mrg'] == -1:
            """if uniquewords[k]['raw'] >= len(sents)-2:
                # print '  Probably no need to worry, it was near the very end of the raw xml doc.'
                # print '--------------------------'
                print '',
            else:
                # print '--------------------------'
                # print 'POSSIBLE ERROR:'
                # print k+': ',
                # print uniquewords[k]
                # print '--------------------------'
                print '',"""
            pass
        else:
            sentencemap[uniquewords[k]['raw']] = uniquewords[k]['mrg']

    return sentencemap

def testaux(compare_str, test_aux_str):
    return compare_str is test_aux_str

def nextaux(mrgmatrix, crt_sentnum, crt_idx):

    if crt_sentnum >= len(mrgmatrix):
        return None

    elif crt_idx >= len(mrgmatrix[crt_sentnum]['lemmas']):
        if crt_sentnum < len(mrgmatrix):
            return nextaux(mrgmatrix, crt_sentnum+1, 0)
        else: return None

     #len(mrgmatrix[crt_sentnum]['lemmas'])):
    if isauxiliary(mrgmatrix[crt_sentnum], crt_idx):
        return crt_sentnum,crt_idx
    else:
        return nextaux(mrgmatrix, crt_sentnum, crt_idx+1)

def getauxs(sentdict):
    global VPE_TRIGGERS_IN_WSJ

    ret = []
    for i in range(0,len(sentdict['lemmas'])):
        if isauxiliary(sentdict, i):
            ret.append((i, sentdict['words'][i]))
    return ret

# This generates the confusion matrix for my algorithm's VPE detection, i.e.
# it calculates the true positives, false positives, true negatives, false negatives.
def f1(confusionmatrix):
    tp = confusionmatrix['tp']
    fn = confusionmatrix['fn']
    fp = confusionmatrix['fp']

    recall = 0
    if tp+fn!=0: recall = float(tp)/float(tp+fn)

    precision = 0
    if tp+fp!=0: precision = float(tp)/float(tp+fp)

    f1score = 0
    if precision + recall != 0: f1score = (2.0*precision*recall)/(precision+recall)

    return {'recall':recall, 'precision':precision, 'f1':f1score}

# Create a log file of what happened. Takes the big confusion matrix.
def logdata(dest, confusionmatrix, logname=None):
    filename = ''

    if logname is None:
        maxx = 0
        for f in listdir(dest):
            if f.endswith('log') and f.startswith('1'):
                if int(f[0:4]) > maxx: # '1000.log' > '1000' > 1000
                    maxx = int(f[0:4])

        filename = str(maxx+1)+'.log'

    else:
        filename = logname+'.log'


    print 'Writing new file: %s'%filename
    newf = open(dest+filename,'w')

    for auxkey in confusionmatrix:
        newf.write(auxkey+': ')
        total_instances = 0
        for key in confusionmatrix[auxkey]:
            newf.write('%s --> %d, '%(key,confusionmatrix[auxkey][key]))
            if not key == 'fp':
                total_instances+=confusionmatrix[auxkey][key]
        newf.write('\n')
        newf.write(auxkey+': total instances --> %d\n\n' %total_instances)

    newf.close()

def evaluation(mrgmatrix, confusionmatrix, myalgvpesents, myalgtriggers, gs_dict, verbose, test_aux=None):
    my_alg_dict = {}
    for i in range(0,len(myalgvpesents)): my_alg_dict[myalgvpesents[i]] = myalgtriggers[i]

    # This is to compare a tuple to a dictionary it may be in.
    def aboutequal(sentdict, idx_and_trigger, dictionary):
        ERROR_LEWAY = 3
        idx,trigger = idx_and_trigger

        for k in dictionary:
            dict_idx = dictionary[k][0]
            # Iterate from the idx to plus or minus a couple words (because improper original files...)
            if dict_idx in range(min(0,idx-ERROR_LEWAY), min(len(sentdict['lemmas']), idx+ERROR_LEWAY)):
                return True
            # TODO: COMPLETE THIS!!!!!!



        return False

    def evaluate(mrgmatrix, confusionmatrix, my_alg_dict, gs_dict, verbose, test_aux):
        for sentnum in my_alg_dict:
            if aboutequal(mrgmatrix[sentnum], my_alg_dict[sentnum], gs_dict):
                confusionmatrix[test_aux]['tp'] += 1

def evaluateresults(mrgmatrix, confusionmatrix, myalgvpesents, myalgtriggers, gs_dict, verbose, test_aux=None):
    # Here we are mapping the sentence numbers to their corresponding triggers (my alg sometimes thinks a sentence has multiple instances of VPE, so multiple triggers).
    my_alg_dict = {}
    for i in range(0,len(myalgvpesents)): my_alg_dict[myalgvpesents[i]] = myalgtriggers[i]

    if test_aux is None:
        for test_aux in confusionmatrix:
            for k in my_alg_dict:
                if test_aux in my_alg_dict[k]:
                    if k in gs_dict:
                        if gs_dict[k] in my_alg_dict[k]: # We need to have the right trigger detected.
                            confusionmatrix[test_aux]['tp'] += 1
                        else:
                            confusionmatrix[test_aux]['fp'] += 1
                            if verbose:
                                print '\nFalse positive:'
                                printsent(mrgmatrix, k)
                    else:
                        confusionmatrix[test_aux]['fp'] += 1
                        if verbose:
                             print '\nFalse positive:'
                             printsent(mrgmatrix, k)
            for k in gs_dict:
                if gs_dict[k] == test_aux:
                    if not k in my_alg_dict:
                        confusionmatrix[test_aux]['fn'] += 1
                        if verbose:
                            print '\nFalse negative:'
                            printsent(mrgmatrix, k)
    else:
        for k in my_alg_dict:
            if test_aux in my_alg_dict[k]:
                if k in gs_dict:
                    if gs_dict[k] in my_alg_dict[k]: # We need to have the right trigger detected.
                        confusionmatrix[test_aux]['tp'] += 1
                    else:
                        confusionmatrix[test_aux]['fp'] += 1
                        if verbose:
                             print '\nFalse positive:'
                             printsent(mrgmatrix, k)
                else:
                    confusionmatrix[test_aux]['fp'] += 1
                    if verbose:
                        print '\nFalse positive:'
                        printsent(mrgmatrix, k)
        for k in gs_dict:
            if gs_dict[k] == test_aux:
                if not k in my_alg_dict:
                    confusionmatrix[test_aux]['fn'] += 1
                    if verbose:
                        print '\nFalse negative:'
                        printsent(mrgmatrix, k)

    return confusionmatrix

#### TESTING ####

confusionmatrix  = {'modal': {'tp': 0, 'fp': 0, 'fn': 0}, 'be': {'tp': 0, 'fp': 0, 'fn': 0},
                    'have': {'tp': 0, 'fp': 0, 'fn': 0}, 'do': {'tp': 0, 'fp': 0, 'fn': 0},
                    'to': {'tp': 0, 'fp': 0, 'fn': 0}, 'so': {'tp': 0, 'fp': 0, 'fn': 0}}
"""
dnum = 0
test = True

TEST_AUX = None

for d in listdir(XML_MRG):
    subdir = d+SLASH_CHAR
    if subdir.startswith('.'): continue
    if dnum in range(0,1) and test:
        annotationmatrix = getvpeannotationsdata(subdir,VPE_ANNOTATIONS)

        for test_file in listdir(XML_MRG+subdir):
            if filehasvpe(test_file, annotationmatrix) and test: #and test_file.startswith('wsj_2232'):

                mrgmatrix = getdataMRG(test_file,XML_MRG+subdir)

                tmp = goldstandardVPEsentences(test_file, XML_RAW_TOKENIZED, subdir, annotationmatrix, mrgmatrix)
                gold_standard_vpe = tmp[0]
                gs_triggers = tmp[1]

                # noinspection PySimplifyBooleanCheck
                if gold_standard_vpe != {}:

                    verbose = False
                    myalgvpe = []
                    myalgtriggers = []

                    # For each sentence...
                    for i in range(0,len(mrgmatrix)):

                        t = nltktree.maketree(mrgmatrix[i]['tree'][0])
                        mrgmatrix[i]['tree'] = t

                        subtrees = nltktree.getsmallestsubtrees(t)
                        subtree_positions = nltktree.getsmallestsubtreepositions(t, subtree_list = subtrees)

                        tmp2 = myalghasvpe(mrgmatrix[i], i, t, subtree_positions, verbose, test_aux=TEST_AUX)
                        if True in tmp2: print tmp2
                        my_alg_has_vpe = False
                        triggers = []

                        if 'multiple' in tmp2:
                            # for each case my algorithm says there is VPE
                            for case in tmp2['multiple']:
                                triggers.append(case[1])
                            my_alg_has_vpe = True

                        else:
                            my_alg_has_vpe = tmp2[0]
                            triggers.append(tmp2[1])

                        if my_alg_has_vpe:
                            myalgvpe.append(i)
                            myalgtriggers.append(triggers)

                    # print 'ACTUAL: ',
                    # print gold_standard_vpe
                    # print 'My alg: ',
                    # print myalgvpe
                    # print
                    verbose = False
                    print myalgvpe
                    print gold_standard_vpe
                    # For each sentence number that truly has VPE...
                    confusionmatrix = evaluateresults(mrgmatrix, confusionmatrix, myalgvpe, myalgtriggers, gold_standard_vpe, verbose, test_aux=TEST_AUX)
                    test = True


                # for i in range(0,len(mrgmatrix)):
                # 	for word in mrgmatrix[i]['words']:
                # 		if word == 'NP' or word=='VP' or word.startswith('ADVP'):
                # 			print 'FUCK this file: %s' %test_file
                # 			printsent(mrgmatrix,i)
    dnum+=1

total_scores = {'fn':0, 'fp':0, 'tp':0}


for test_aux in confusionmatrix:
    print '\n%s:'%test_aux
    print confusionmatrix[test_aux]
    scores = f1(confusionmatrix[test_aux])
    for k in scores:
        print k.capitalize()+' : %0.2f' %scores[k]
    for k in confusionmatrix[test_aux]:
        total_scores[k] += confusionmatrix[test_aux][k]

print '\nTotal:'
print total_scores
scores = f1(total_scores)
for k in scores:
        print k.capitalize()+' : %0.2f' %scores[k]



try:
    logdata(LOG_LOCATIONS,confusionmatrix,argv[1])
except IndexError:
    logdata(LOG_LOCATIONS,confusionmatrix)

print '\nTime taken: ',
print time.time()-timestart
"""
