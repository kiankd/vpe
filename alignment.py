# Author: Kian Kenyon-Dean
# This file contains code that performs an alignment between the trigger's context and
# the context of each potential antecedent. This alignment is then turned into a feature marix that
# we pass to MIRA.
#
# The context is the list of chunked dependencies within the realm of the trigger/antecedent's
# nearest clause.
from nltktree import get_nearest_clause
from copy import copy
import numpy as np
import antecedent_vector_creation as avc

def alignment_matrix(sentences, trigger, dep_names=('prep','adv','dobj','nsubj','nmod'), pos_tags=None):
    """
    Creates an alignment vector between the trigger and each of its potential antecedents.
    @type sentences: vpe_objects.AllSentences
    @type trigger: vpe_objects.Auxiliary
    """
    trig_sentdict = sentences.get_sentence(trigger.sentnum)
    i,j = nearest_clause(trig_sentdict, trigger.wordnum-1) # WE NEED TO SUBTRACT BY ONE BECAUSE NO ROOT IN TREES
    trig_chunks = trig_sentdict.chunked_dependencies(i, j, dep_names=dep_names)
    remove_idxs(trig_chunks, trigger.wordnum, trigger.wordnum)

    for ant in trigger.possible_ants + [trigger.gold_ant]:
        ant_sentdict = sentences.get_sentence(ant.sentnum)
        # print ant_sentdict.words
        # print ant.sub_sentdict.words
        k,l = nearest_clause(ant_sentdict, ant.start-1, end=ant.end-1)
        ant_chunks = ant_sentdict.chunked_dependencies(k, l, dep_names=dep_names)
        remove_idxs(ant_chunks, ant.start, ant.end)

        ant.x = np.array([1] + alignment_vector(trig_chunks, ant_chunks, dep_names)
                         + relational_vector(trigger, ant) + avc.ant_trigger_relationship(ant, trigger, sentences, pos_tags))
        # print len(ant.x), ant.x
    return

def relational_vector(trig, ant):
    """
        This creates a feature vector that represents the basic relationship between trigger and antecedent,
        i.e. word distance, sentence distance, etc.
    """
    v = []
    v += [1.0 if trig.sentnum == ant.sentnum else 0.0]
    v += [trig.sentnum - ant.sentnum]
    v += [ant.sentnum - trig.sentnum]
    v += [(trig.wordnum - ant.start)*(1+trig.sentnum - ant.sentnum)]
    v += [(trig.wordnum - ant.end)*(1+trig.sentnum - ant.sentnum)]
    v += [(ant.start - trig.wordnum)*(1+trig.sentnum - ant.sentnum)]
    v += [(ant.end - trig.wordnum)*(1+trig.sentnum - ant.sentnum)]

    return v

def alignment_vector(t_chunks, a_chunks, dep_names, threshold=0.25, one_hot_length=5):
    """
        This creates an alignment between the dependency chunks of a trigger and potential antecedent,
        then returns the feature vector representing that alignment.
    """
    # First we do a bipartite alignment mapping from t_chunks to a_chunks:
    mapping = []

    un_mapped_trigs = copy(t_chunks)
    un_mapped_ants = copy(a_chunks)
    for i in range(len(t_chunks)):
        tchunk = t_chunks[i]
        best_score = 0.0
        best_achunk = None

        for j in range(len(un_mapped_ants)):
            achunk = un_mapped_ants[j]
            s = similarity_score(tchunk, achunk, abs(i-j)/len(t_chunks))
            if s > best_score:
                best_achunk = achunk
                best_score = s

        if best_score > threshold:
            # print '\nAligning:',tchunk,best_achunk
            un_mapped_trigs.remove(tchunk)
            un_mapped_ants.remove(best_achunk)
            mapping.append((tchunk, best_achunk, best_score))
            # print mapping

    # print '\n---------------------MAPPING--------------------'
    # print mapping
    # print 'Null trig chunks:'
    # print un_mapped_trigs
    # print 'Null ant chunks:'
    # print un_mapped_ants

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

    # Now encode the mean and standard deviation of the scores, min and max of scores:
    scores = [tup[2] for tup in mapping]
    if len(scores):
        v += [np.mean(scores), np.std(scores)]
        v += [min(scores), max(scores)]
    else:
        v += [0.0, 0.0, 0.0, 0.0]

    return v

def similarity_score(c1, c2, distance_cost, words_weight=0.2, pos_weight=0.5, lemma_weight=0.3):
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
    # print score
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
        return 1.0

def remove_idxs(chunks, start_idx, end_idx):
    """This removes the trigger and antecedent from the chunks."""
    for chunk in chunks:
        indexes_to_remove = range(start_idx,end_idx+1)
        start_remove = None
        end_remove = 0
        for i in range(len(chunk['sentdict'])):
            # print i,i + chunk['sentdict'].start
            if i + chunk['sentdict'].start in indexes_to_remove:
                if start_remove==None:
                    start_remove = i
                end_remove = i
        if start_remove==None:
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
