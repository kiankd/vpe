# Author: Kian Kenyon-Dean
# This file contains code that performs an alignment between the trigger's context and
# the context of each potential antecedent. This alignment is then turned into a feature marix that
# we pass to MIRA.
#
# The context is the list of chunked dependencies within the realm of the trigger/antecedent's
# nearest clause.
from nltktree import get_nearest_clause
from copy import copy
from vpe_objects import chunks_to_string
import numpy as np
import antecedent_vector_creation as avc
import word2vec_functionality as w2v

MAPPING_LENGTHS = []

def alignment_matrix(sentences, trigger, word2vec_dict, dep_names=('prep','adv','dobj','nsubj','nmod'), pos_tags=None):
    """
        Creates an alignment vector between the trigger and each of its potential antecedents.
        @type sentences: vpe_objects.AllSentences
        @type trigger: vpe_objects.Auxiliary
    """
    global MAPPING_LENGTHS
    ANT_CHUNK_LENGTHS = []
    print '-----------'

    trig_sentdict = sentences.get_sentence(trigger.sentnum)
    print trig_sentdict

    i,j = nearest_clause(trig_sentdict, trigger.wordnum-1) # WE NEED TO SUBTRACT BY ONE BECAUSE NO ROOT IN TREES
    trig_chunks = trig_sentdict.chunked_dependencies(i, j, dep_names=dep_names)
    remove_idxs(trig_chunks, trigger.wordnum, trigger.wordnum)

    for chunk in trig_chunks:
        print '\t'+chunks_to_string(chunk)

    for ant in trigger.possible_ants + [trigger.gold_ant]:
        # print '\n---------------------MAPPING--------------------'
        # print 'SENTENCE:',trig_sentdict.words_to_string()
        # print trigger
        # print ant,'%d - (%d,%d)\n'%(ant.sentnum,ant.start,ant.end)

        ant_sentdict = sentences.get_sentence(ant.sentnum)
        # print ant_sentdict.words
        # print ant.sub_sentdict.words
        k,l = nearest_clause(ant_sentdict, ant.start-1, end=ant.end-1)
        ant_chunks = ant_sentdict.chunked_dependencies(k, l, dep_names=dep_names)

        ANT_CHUNK_LENGTHS.append(len(ant_chunks))

        # print chunks_to_string(ant_chunks)
        remove_idxs(ant_chunks, ant.start, ant.end)

        # print trigger
        # print ant#,'%d - (%d,%d)\n'%(ant.sentnum,ant.start,ant.end)

        mapping, untrigs, unants = align(trig_chunks, ant_chunks, dep_names, word2vec_dict)

        ant.x = np.array([1] + alignment_vector(mapping, untrigs, unants, dep_names, verbose=False))
                             # + relational_vector(trigger, ant)
                             # + avc.ant_trigger_relationship(ant, trigger, sentences, pos_tags))

    # if 'industrial-production index' in trig_sentdict.words_to_string():
    #     exit(0)

    print 'Avg mapping, trig_chunks, ant_chunks lengths: %0.2f, %d, %0.2f'%(np.mean(MAPPING_LENGTHS),
                                                                            len(trig_chunks),
                                                                            np.mean(ANT_CHUNK_LENGTHS))
    ANT_CHUNK_LENGTHS = []
    MAPPING_LENGTHS = []
    return

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

def align(t_chunks, a_chunks, dep_names, word2vec_dict, threshold=0.65, verbose=False):
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
            s = similarity_score(tchunk, achunk, word2vec_dict, abs(i-j)/len(t_chunks))

            if s > threshold:
                if tchunk in un_mapped_trigs:
                    un_mapped_trigs.remove(tchunk)

                if achunk in un_mapped_ants:
                    un_mapped_ants.remove(achunk)

                mapping.append((tchunk, achunk, s))

    for chunk in a_chunks:
        print chunks_to_string(chunk)
    print mapping_to_string(mapping)

    MAPPING_LENGTHS.append(len(mapping))

    # if verbose:
    #     print '--------------\nFrom this ant-chunks to this trig-chunks:'
    #     print a_chunks
    #     print t_chunks
    #     print "mapping:",mapping
    #     print 'Null trig chunks:'
    #     print un_mapped_trigs
    #     print 'Null ant chunks:'
    #     print un_mapped_ants

    return mapping, un_mapped_trigs, un_mapped_ants

def alignment_vector(mapping, un_mapped_trigs, un_mapped_ants, dep_names, one_hot_length=5, verbose=False):
    """
        This creates an alignment between the dependency chunks of a trigger and potential antecedent,
        then returns the feature vector representing that alignment.
    """

    # Given that we have the mapping, make its feature vector:
    v = []

    # Easy vecs first: one hot encoding of number of unmapped chunks and the mapping.
    v += [1.0 if len(un_mapped_trigs) == i else 0.0 for i in range(one_hot_length)]
    v += [1.0 if not 1.0 in v[-one_hot_length:] else 0.0] # For encoding if we have more empty than the one-hot-length.

    v += [1.0 if len(un_mapped_ants) == i else 0.0 for i in range(one_hot_length)]
    v += [1.0 if not 1.0 in v[-one_hot_length:] else 0.0]

    v += [1.0 if len(mapping) == i else 0.0 for i in range(one_hot_length)]
    v += [1.0 if not 1.0 in v[-one_hot_length:] else 0.0]

    # Now encode the dependencies that have been mapped to.
    mapped_trig_deps = [tup[0]['name'] for tup in mapping]
    v += [1.0 if dep_name in mapped_trig_deps else 0.0 for dep_name in dep_names]
    v += [1.0 if not 1.0 in v[-len(dep_names):] else 0.0] # Encode if all the deps are missing

    mapped_ant_deps = [tup[1]['name'] for tup in mapping]
    v += [1.0 if dep_name in mapped_ant_deps else 0.0 for dep_name in dep_names]
    v += [1.0 if not 1.0 in v[-len(dep_names):] else 0.0]


    # TODO: FRACTIONIZE THESE!!!!!
    a = [len(tup[0]['sentdict'].words) for tup in mapping]
    b = [len(tup[1]['sentdict'].words) for tup in mapping]
    c = [len(chunk['sentdict'].words) for chunk in un_mapped_ants]
    d = [len(chunk['sentdict'].words) for chunk in un_mapped_trigs]
    for l in [a,b,c,d]:
        if not l:
            v += [ 1.0, 0.0, 0.0 ]
        else:
            v += [ 0.0, float(min(l))/max(l), np.mean(l)/max(l) ]

    # Now encode the mean and standard deviation of the scores, min and max of scores:
    scores = [tup[2] for tup in mapping]
    if len(scores):
        v += [np.mean(scores), np.std(scores)]
        v += [min(scores), max(scores)]
    else:
        v += [0.0, 0.0, 0.0, 0.0]
    #
    # print t_chunks
    # print a_chunks
    # print mapping
    # print v
    # print '-----------------------'

    return v

def similarity_score(c1, c2, distance_cost, word2vec_dict, words_weight=0.5, pos_weight=0.3, lemma_weight=0.2):
    """
        Scores the similarity between 2 chunks.
        If both chunks have the same dependency then they get a high score.
    """
    score = 0.0
    if c1['name'] == c2['name']:
        score += 3.0
    score += f1_similarity(c1['sentdict'].words, c2['sentdict'].words) * words_weight
    score += f1_similarity(c1['sentdict'].pos, c2['sentdict'].pos) * pos_weight
    score += f1_similarity(c1['sentdict'].lemmas, c2['sentdict'].lemmas) * lemma_weight
    score -= distance_cost

    c1_vec = w2v.average_vec_for_list()

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
