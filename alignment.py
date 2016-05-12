# Author: Kian Kenyon-Dean
# This file contains code that performs an alignment between the trigger's context and
# the context of each potential antecedent. This alignment is then turned into a feature marix that
# we pass to MIRA.
#
# The context is the list of chunked dependencies within the realm of the trigger/antecedent's
# nearest clause.
from nltktree import get_nearest_clause,lowest_common_subtree_phrases,get_phrases,getwordtreepositions
from copy import copy
from vpe_objects import chunks_to_string
import numpy as np
import antecedent_vector_creation as avc
import word2vec_functionality as w2v
import word_characteristics as wc

MAPPING_LENGTHS = []
MAX_SCORE = 4.0

def alignment_matrix(sentences, trigger, word2vec_dict, dep_names=('prep','adv','dobj','nsubj','nmod'), pos_tags=None):
    """
        Creates an alignment vector between the trigger and each of its potential antecedents.
        @type sentences: vpe_objects.AllSentences
        @type trigger: vpe_objects.Auxiliary
    """
    global MAPPING_LENGTHS
    ANT_CHUNK_LENGTHS = []

    trig_sentdict = sentences.get_sentence(trigger.sentnum)

    i,j = nearest_clause(trig_sentdict, trigger.wordnum-1) # WE NEED TO SUBTRACT BY ONE BECAUSE NO ROOT IN TREES
    trig_chunks = trig_sentdict.chunked_dependencies(i, j, dep_names=dep_names)
    remove_idxs(trig_chunks, trigger.wordnum, trigger.wordnum)

    for ant in trigger.possible_ants + [trigger.gold_ant]:

        if ant.get_words() == ['get', 'slightly', 'higher', 'yields'] or ant.get_words() == ['get', 'slightly', 'higher', 'yields','on','deposits']:
            pass

        ant_sentdict = sentences.get_sentence(ant.sentnum)

        k,l = nearest_clause(ant_sentdict, ant.start-1, end=ant.end-1)

        # if ant.sentnum == trigger.sentnum and k < l:
        #     l = min(l, i) # we don't want the nearest clause to include the trigger's clause.

        ant_chunks = ant_sentdict.chunked_dependencies(k, l, dep_names=dep_names)

        ANT_CHUNK_LENGTHS.append(len(ant_chunks))

        remove_idxs(ant_chunks, ant.start, ant.end)
        remove_idxs(ant_chunks, trigger.wordnum, trigger.wordnum)

        mapping, untrigs, unants = align(trig_chunks, ant_chunks, dep_names, word2vec_dict, threshold=0.15)

        ant.x = np.array([1] + alignment_vector(mapping, untrigs, unants, dep_names, word2vec_dict, verbose=False)
                             + relational_vector(trigger, ant)
                             + avc.ant_trigger_relationship(ant, trigger, sentences, pos_tags, word2vec_dict)
                             + hardt_features(ant, trigger, sentences, pos_tags))

    # print 'Avg mapping, trig_chunks, ant_chunks lengths: %0.2f, %d, %0.2f'\
    #       %(np.mean(MAPPING_LENGTHS), len(trig_chunks),np.mean(ANT_CHUNK_LENGTHS))

    ANT_CHUNK_LENGTHS = []
    MAPPING_LENGTHS = []
    return

def hardt_features(ant, trig, sentences, pos_tags):
    """
        This exists to add features that are based on what Hardt did in 1997.
        @type ant: vpe_objects.Antecedent
        @type trig: vpe_objects.Auxiliary
        @type sentences: vpe_objects.AllSentences
    """
    v = []
    sent_tree = sentences.get_sentence_tree(ant.sentnum)
    ant_sent = sentences.get_sentence(ant.sentnum)
    trig_sent = sentences.get_sentence(trig.sentnum)

    vp = sentences.nearest_vp(trig)
    vp_head = vp.get_head()
    vp_head_idx = vp.get_head(idx=True)

    ant_head = ant.get_head()
    ant_head_idx = ant.get_head(idx=True)

    v.append(1.0 if ant == vp else 0.0)
    v.append(1.0 if ant_head == vp_head else 0.0)
    v.append(1.0 if vp.start <= ant_head_idx <= vp.end else 0.0)
    v.append(1.0 if ant.start <= vp_head_idx <= ant.end else 0.0)
    v.append(ant.sentnum - vp.sentnum)
    v.append(ant.start - vp.start)
    v.append(ant.end - vp.end)

    # be-do form
    try:
        v.append(1.0 if wc.is_be(ant_sent.lemmas[ant.start-1]) or wc.is_be(ant_sent.lemmas[ant.start]) else 0.0)
        v.append(1.0 if trig.type == 'do' and v[-1]==1.0 else 0.0)
    except IndexError:
        v += [0.0, 0.0]

    # quotation features
    quote_start_trig, quote_end_trig = None,None
    for i,w in enumerate(trig_sent.lemmas):
        if w == "\"":
            if not quote_start_trig:
                quote_start_trig = i
            else:
                quote_end_trig = i
                break

    trig_in_quotes = False
    if quote_start_trig and quote_end_trig:
        trig_in_quotes = quote_start_trig <= trig.wordnum <= quote_end_trig
        v.append(1.0 if trig_in_quotes else 0.0)
    else:
        v.append(0.0)

    quote_start_ant, quote_end_ant = None,None
    for i,w in enumerate(ant_sent.lemmas):
        if w == "\"":
            if not quote_start_ant:
                quote_start_ant = i
            else:
                quote_end_ant = i
                break

    ant_in_quotes = False
    if quote_start_ant and quote_end_ant:
        ant_in_quotes = quote_start_ant <= ant.start <= quote_end_ant and quote_start_ant <= ant.end <= quote_end_ant
        v.append(1.0 if quote_start_ant <= ant.start <= quote_end_ant else 0.0)
        v.append(1.0 if quote_start_ant <= ant.end <= quote_end_ant else 0.0)
    else:
        v += [0.0,0.0]

    v.append(1.0 if trig_in_quotes and ant_in_quotes else 0.0)


    # Nielsen features
    v.append(1.0 if wc.is_aux_lemma(ant.sub_sentdict.lemmas[0]) else 0.0)
    v.append(1.0 if wc.is_aux_lemma(ant.sub_sentdict.lemmas[ant.get_head(idx=True, idx_in_subsentdict=True)]) else 0.0)
    for tag in pos_tags:
        v.append(1.0 if tag == ant.sub_sentdict.pos[0] else 0.0) # Sparse encoding of the pos tag of first word in ant
        v.append(1.0 if tag == ant.sub_sentdict.pos[-1] else 0.0) # Sparse encoding of the pos tag of last word in ant
        v.append(float(ant.sub_sentdict.pos.count(tag)) / len(ant.sub_sentdict)) # Frequency of the given pos tag in ant

    for fun in [wc.is_adverb, wc.is_verb, wc.is_adverb, wc.is_noun, wc.is_preposition, wc.is_punctuation, wc.is_predicative]:
        v.append(1.0 if fun(ant.sub_sentdict.pos[0]) else 0.0) # Sparse encoding of the identity of first word in ant
        v.append(1.0 if fun(ant.sub_sentdict.pos[-1]) else 0.0) # Sparse encoding of the identity of last word in ant
        v.append(float(len(map(fun,ant.sub_sentdict.pos))) / len(ant.sub_sentdict)) # Frequency of the given function in ant

    sent_phrases = get_phrases(sent_tree)
    ant_phrases = lowest_common_subtree_phrases(sent_tree, ant.get_words())

    v.append(float(len(ant_phrases)) / len(sent_phrases))
    for phrase in ['NP','VP','S','SINV','ADVP','ADJP','PP']:
        v.append(len(map(lambda s: s.startswith(phrase), ant_phrases)) / float(len(ant_phrases)))
        v.append(len(map(lambda s: s.startswith(phrase), sent_phrases)) / float(len(sent_phrases)))

    continuation_words = ['than','as','so']
    if ant.sentnum == trig.sentnum:
        v.append(1.0)
        for word in continuation_words:
            v.append(1.0 if word in ant_sent.words[ant.end : trig.wordnum] else 0.0)
    else:
        v.append(0.0)
        for word in continuation_words:
            v.append(0.0)
    try:
        v.append(1.0 if ant_sent.words[ant.start-1] == trig.word else 0.0)
        v.append(1.0 if ant_sent.lemmas[ant.start-1] == trig.lemma else 0.0)
        v.append(1.0 if ant_sent.lemmas[ant.start-1] == trig.type else 0.0)
        v.append(1.0 if ant_sent.pos[ant.start-1] == trig.pos else 0.0)
    except IndexError:
        v += [0.0, 0.0, 0.0, 0.0]

    # Theoretical linguistics features
    if ant.sentnum == trig.sentnum:
        word_positions = getwordtreepositions(sent_tree)

        v.append(1.0)
        v.append(1.0 if wc.ccommands(ant.start, trig.wordnum, sent_tree, word_positions) else 0.0)
        v.append(1.0 if wc.ccommands(trig.wordnum, ant.start, sent_tree, word_positions) else 0.0)
        v.append(1.0 if wc.ccommands(ant.end, trig.wordnum, sent_tree, word_positions) else 0.0)
        v.append(1.0 if wc.ccommands(trig.wordnum, ant.end, sent_tree, word_positions) else 0.0)

        # Check if a word in the antecedent c-commands the trig and vice versa.
        ant_word_ccommands,trig_ccommands = False,False
        for idx in range(ant.start, ant.end+1):
            if wc.ccommands(idx, trig.wordnum, sent_tree, word_positions):
                v.append(1.0)
                ant_word_ccommands = True

            if wc.ccommands(trig.wordnum, idx, sent_tree, word_positions):
                v.append(1.0)
                trig_ccommands = True

            if ant_word_ccommands and trig_ccommands: # speed boost of 0.02ms kek
                break

        if not ant_word_ccommands:
            v.append(0.0)

        if not trig_ccommands:
            v.append(0.0)
    else:
        v += [0.0 for _ in range(7)]

    return v

def relational_vector(trig, ant):
    """
        This creates a feature vector that represents the basic relationship between trigger and antecedent,
        i.e. word distance, sentence distance, etc.
    """
    v = []
    v += [1.0 if trig.sentnum == ant.sentnum else 0.0]
    v += [trig.sentnum - ant.sentnum]
    v += [(trig.wordnum - ant.start)*(1+trig.sentnum - ant.sentnum)]
    v += [(trig.wordnum - ant.end)*(1+trig.sentnum - ant.sentnum)]

    return v

def word2vec_alignment_features(word2vec_dict, mapping):
    return

def mapping_to_string(mapping):
    s=''
    for tup in mapping:
        try:
            s += '('+tup[0]['name']+', '+tup[1]['name'] +'): %0.2f - '%tup[2] + chunks_to_string(tup[0]) + '<-----> ' + chunks_to_string(tup[1]) + '\n'
        except TypeError:
            continue
    return s

def align(t_chunks, a_chunks, dep_names, word2vec_dict, threshold=0.15, verbose=False):
    """
    Creates an alignment between the chunks of a trigger context and antecedent context.
    :param t_chunks: List of chunks created for the trigger's context
    :param a_chunks: List of chunks created for the antecedent's context
    :param dep_names: Names of dependencies that we chunked for.
    :param threshold: Float minimum score for creating a mapping.
    :param verbose: Boolean.
    :return: 3 lists, mapping, un_mapped trig chunks, un_mapped ant chunks.
    """
    global MAPPING_LENGTHS

    mapping, un_mapped_trigs, un_mapped_ants = [], copy(t_chunks), copy(a_chunks)

    for i in range(len(t_chunks)):
        tchunk = t_chunks[i]
        best_score = 0.0
        best_achunk = None

        for j in range(len(a_chunks)):
            achunk = a_chunks[j]
            s = similarity_score(tchunk, achunk, word2vec_dict)#, words_weight=0.0, lemma_weight=1.0, pos_weight=0.0)

            if s > threshold:
                if tchunk in un_mapped_trigs:
                    un_mapped_trigs.remove(tchunk)

                if achunk in un_mapped_ants:
                    un_mapped_ants.remove(achunk)

                mapping.append((tchunk, achunk, s))

    # for chunk in a_chunks:
    #     print chunks_to_string(chunk)
    #
    # print mapping_to_string(mapping)

    MAPPING_LENGTHS.append(len(mapping))

    return mapping, un_mapped_trigs, un_mapped_ants

def alignment_vector(mapping, un_mapped_trigs, un_mapped_ants, dep_names, word2vec_dict, one_hot_length=5, verbose=False):
    """
        This creates an alignment between the dependency chunks of a trigger and potential antecedent,
        then returns the feature vector representing that alignment.
    """

    # Given that we have the mapping, make its feature vector:
    v = []
    mapping_length = len(mapping) if len(mapping) > 0 else 1

    # Easy vecs first: one hot encoding of number of unmapped chunks and the mapping.
    v += [1.0 if len(un_mapped_trigs) == i else 0.0 for i in range(one_hot_length)]
    v += [1.0 if not 1.0 in v[-one_hot_length:] else 0.0] # For encoding if we have more empty than the one-hot-length.
    v.append(len(un_mapped_trigs)/mapping_length)

    v += [1.0 if len(un_mapped_ants) == i else 0.0 for i in range(one_hot_length)]
    v += [1.0 if not 1.0 in v[-one_hot_length:] else 0.0]
    v.append(len(un_mapped_ants)/mapping_length)

    v += [1.0 if len(mapping) == i else 0.0 for i in range(one_hot_length)]
    v += [1.0 if not 1.0 in v[-one_hot_length:] else 0.0]
    v.append((len(un_mapped_trigs)+len(un_mapped_ants))/mapping_length)

    # Now encode the dependencies that have been mapped to.
    mapped_trig_deps = [tup[0]['name'] for tup in mapping]
    v += [1.0 if dep_name in mapped_trig_deps else 0.0 for dep_name in dep_names]
    v += [1.0 if not 1.0 in v[-len(dep_names):] else 0.0] # Encode if all the deps are missing

    mapped_ant_deps = [tup[1]['name'] for tup in mapping]
    v += [1.0 if dep_name in mapped_ant_deps else 0.0 for dep_name in dep_names]
    v += [1.0 if not 1.0 in v[-len(dep_names):] else 0.0]

    mapped_dep_tups = [(tup[0]['name'], tup[1]['name']) for tup in mapping]
    for d1 in dep_names:
        for d2 in dep_names:
            if (d1, d2) in mapped_dep_tups:
                v.append(1.0)
            else:
                v.append(0.0)

    a = [len(tup[0]['sentdict'].words) for tup in mapping]
    b = [len(tup[1]['sentdict'].words) for tup in mapping]
    c = [len(chunk['sentdict'].words) for chunk in un_mapped_ants]
    d = [len(chunk['sentdict'].words) for chunk in un_mapped_trigs]

    for l in [a,b,c,d]:
        if not l:
            v += [ 0.0, 0.0 ]
        else:
            v += [ float(min(l))/max(l), np.mean(l)/max(l) ]

    # Now encode the mean and standard deviation of the scores, min and max of scores:
    scores = [tup[2] for tup in mapping]
    if len(scores):
        v += [np.mean(scores), np.std(scores), min(scores), max(scores)]
    else:
        v += [0.0, 0.0, 0.0, 0.0]

    # Word2vec features.
    angles = []
    for tup in mapping:
        a = angle_between_chunks(tup[0], tup[1], word2vec_dict)
        if a:
            angles.append(a)

    if angles:
        v += [np.mean(angles), np.std(angles), min(angles), max(angles)]
    else:
        v += [0.0, 0.0, 0.0, 0.0]

    return v

def angle_between_chunks(c1, c2, word2vec_dict):
    c1_vec = w2v.average_vec_for_list(c1['sentdict'].words, word2vec_dict)
    c2_vec = w2v.average_vec_for_list(c2['sentdict'].words, word2vec_dict)

    if c1_vec and c2_vec:
        return w2v.angle_btwn_vectors(c1_vec, c2_vec)
    else:
        return None

def similarity_score(c1, c2, word2vec_dict, words_weight=0.2, pos_weight=0.5, lemma_weight=0.3, punish=False):
    """
        Scores the similarity between 2 chunks.
        If both chunks have the same dependency then they get a high score.
    """
    score = 0.0

    if c1['name'] == c2['name'] == 'nsubj':
        score += 1.5

    elif c1['name'] == c2['name']:
        score += 3.0

    score += f1_similarity(c1['sentdict'].words, c2['sentdict'].words) * words_weight
    score += f1_similarity(c1['sentdict'].pos, c2['sentdict'].pos) * pos_weight
    score += f1_similarity(c1['sentdict'].lemmas, c2['sentdict'].lemmas) * lemma_weight

    if punish:
        if f1_similarity(c1['sentdict'].words, c2['sentdict'].words) == 0.0:
            score -= words_weight

        if f1_similarity(c1['sentdict'].pos, c2['sentdict'].pos) == 0.0:
            score -= pos_weight

        if f1_similarity(c1['sentdict'].lemmas, c2['sentdict'].lemmas) == 0.0:
            score -= lemma_weight

    # a = angle_between_chunks(c1, c2, word2vec_dict)
    #
    # if a:
    #     score += (90.0-a) / 90.0

    return score

def f1_similarity(l1, l2):
    tp = float(len([v for v in l1 if v in l2]))
    fp = float(len([v for v in l1 if not v in l2]))
    fn = float(len([v for v in l2 if not v in l1]))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    try:
        return 1.0 - (2.0*precision*recall)/(precision+recall)
    except ZeroDivisionError:
        return 0.0

def remove_idxs(chunks, start_idx, end_idx):
    """This removes the trigger and antecedent from the chunks."""
    for chunk in chunks:
        indexes_to_remove = range(start_idx,end_idx+1)
        start_remove = None
        end_remove = 0
        for i in range(len(chunk['sentdict'])):
            # print i,i + chunk['sentdict'].start
            if i + chunk['sentdict'].start in indexes_to_remove:
                if start_remove == None:
                    start_remove = i
                end_remove = i
        if start_remove == None:
            continue
        # print 'Removing:'
        # print chunk['sentdict'].words[start_remove: end_remove+1]
        # print chunk['sentdict'].pos[start_remove: end_remove+1]
        # print chunk['sentdict'].lemmas[start_remove: end_remove+1]
        del chunk['sentdict'].words[start_remove: end_remove+1]
        del chunk['sentdict'].pos[start_remove: end_remove+1]
        del chunk['sentdict'].lemmas[start_remove: end_remove+1]

    rm = []
    for chunk in chunks:
        if len(chunk['sentdict']) == 0:
            rm.append(chunk)
    for chunk in rm:
        chunks.remove(chunk)
    return

def nearest_clause(s, start, end=None):
    clause = get_nearest_clause(s.get_nltk_tree(), start, end=end)
    return find_word_sequence(s.words, clause.leaves())

def find_word_sequence(words, targets):
    start,end,count = -1,-1,0

    for i in range(len(words)):
        if words[i] == targets[count]:
            if count == 0:
                start = i

            count += 1

            if count == len(targets):
                end = i
                break
        else:
            count = 0

    return start,end

"""
ADD FEATURE - NUMBER/PROPORTION OF UNMAPPED WORDS
GET rid of trigger's clause from antecedent chunks
"""
