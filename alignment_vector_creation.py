import nltktree as NT
import old.detectVPE as DV
import old.vectorcreation as VC
import numpy as np
from truth import *
import math

# @author Kian Kenyon-Dean
# Purpose: to generate feature vectors according to an alignment of comparing the environment of a possible antecedent to
# the surrounding environment of the trigger.

def make_alignment_vector(trig_sentdict, ant_sentdict, ant, trigger, POS_TAGS, AUX_WORDS, word2vec_dict, unique_patterns):
    vec = antecedent_description(trig_sentdict, ant_sentdict, ant, POS_TAGS)
    vec += trigger_description(trig_sentdict, ant_sentdict, trigger, POS_TAGS, AUX_WORDS)
    vec += alignment_comparison(trig_sentdict, ant_sentdict, ant, trigger, word2vec_dict)

    if unique_patterns:
        pattern_dict = {unique_patterns[i]:i for i in range(0,len(unique_patterns))}
        pattern_vec = [0 for _ in range(0,len(unique_patterns))]
        pattern_vec[pattern_dict[trigger.get_pattern()]] = 1
        vec += pattern_vec

    return vec

# Feature 1 is whether or not the antecedent is a constituent.
# Feature 2 is the length of the antecedent.
# Feature 3 is if the trigger is dominated by the antecedent's subtree.
# Feature set 4 is a list corresponding to the length of the Pos tags, it counts the occurences of each pos tag in the ant.
def antecedent_description(trig_sentdict, ant_sentdict, ant, POS_TAGS):
    vector = []

    ant_words = ant.get_words()
    subtree = ant.get_subtree()

    # Feature 1.
    vector.append(truth(len(subtree.leaves()) == len(ant_words)))

    # Feature 2.
    vector.append(len(ant_words))

    # Feature 3.
    vector.append(truth(NT.dominates(subtree.root(), subtree, ant.get_trigger().get_subtree())))

    # Features 4.
    pos_tags_dict = {}
    for tag in POS_TAGS:
        pos_tags_dict[tag] = 0

    idx = get_antecedent_head_index(ant.get_context(), ant)
    for tag in ant.get_context()['pos'][idx:len(ant_words)]:
        pos_tags_dict[tag] += 1

    vector += [pos_tags_dict[tag] for tag in pos_tags_dict]

    # Feature 5: if the antecedent starts with an auxiliary, verb, adj.
    vector.append(truth(DV.isauxiliary(ant_sentdict, idx)))
    vector.append(truth(DV.isverb(ant_sentdict['pos'][idx])))
    vector.append(truth(DV.isadj(ant_sentdict['pos'][idx])))

    return vector

def trigger_description(trig_sentdict, ant_sentdict, trigger, POS_TAGS, AUX_WORDS):
    vector = []

    trig_words = trigger.get_words()
    subtree = trigger.get_subtree()
    context_idx = 0

    for w in trigger.get_context()['words']:
        if w == trig_sentdict['words'][trigger.get_idx()]: break
        context_idx += 1

    # Features 1,2
    vector.append(truth(len(subtree.leaves()) == len(trig_words)))
    vector.append(len(trig_words))

    # Feature set 3.
    pos_tags_dict = {}

    for tag in POS_TAGS: pos_tags_dict[tag] = 0

    for tag in trigger.get_context()['pos'][context_idx:len(trig_words)]:
        pos_tags_dict[tag] += 1

    vector += [pos_tags_dict[tag] for tag in pos_tags_dict]

    # Feature sets 4,5,6. Description of the auxiliary.
    vector += VC.lemmacategoryvector(trig_sentdict['lemmas'][trigger.get_idx()])
    vector += VC.lemmavector(trig_sentdict['lemmas'][trigger.get_idx()])
    vector += VC.auxwordvector(trig_sentdict['words'][trigger.get_idx()], AUX_WORDS)

    return vector

# Feature 1: comparing if the auxiliary in the antecedent's context is in the trigger's context.
# Feature 2: if the antecedent comes after the trigger and then if it is the trigger.
# Features 3,4,5: number, percentage of words,lemmas,pos_tags in common
# Feature 6: distance antecedent head is from trigger.
# Feature 7: comparing the NPs if they are there. Using word2vec!
#TODO: add features that can differentiate between antecedents which share heads.
def alignment_comparison(trig_sentdict, ant_sentdict, ant, trigger, word2vec_dict):
    vector = []

    ant_context_sentdict = ant.get_context()
    trig_context_sentdict = trigger.get_context()

    ant_head_idx = get_antecedent_head_index(ant_sentdict, ant)

    # Feature 1.
    ant_auxs = []
    for i in range(0,len(ant_sentdict['words'])):
        if DV.isauxiliary(ant_sentdict, i):
            ant_auxs.append(ant_sentdict['lemmas'][i])

    found = False
    for aux in ant_auxs:
        if aux in trig_sentdict['lemmas']:
            vector.append(truth(True))
            found = True
            break

    if not found:
        vector.append(truth(False))

    # Feature 2.
    if ant.get_sentnum() == trigger.get_sentnum():
        vector.append(truth(ant_head_idx > trigger.get_idx()))
        vector.append(truth(ant_head_idx == trigger.get_idx()))
        vector.append(truth(ant_head_idx < trigger.get_idx()))
    else: vector += [0,0,0]

    # Features 3,4,5.
    for k in ['words','lemmas','pos']:
        total = len(ant_context_sentdict[k])+len(trig_context_sentdict[k])
        common = len(set(ant_context_sentdict[k]).intersection(trig_context_sentdict[k]))
        vector.append(common)
        vector.append((2.0*float(common))/float(total))

    # Feature 6 - number of words between trigger and antecedent.
    vector.append(ant.get_sentnum()-trigger.get_sentnum())
    if ant.get_sentnum() == trigger.get_sentnum(): vector.append(ant_head_idx - trigger.get_idx())
    else:
        crt_sentnum = trigger.get_sentnum()
        distance = ant_head_idx
        while crt_sentnum < ant.get_sentnum():
            distance += len(trig_sentdict['words'])
            crt_sentnum += 1
        vector.append(distance)

    # Feature 7.
    # First we get the vecs from the Ant NP and average them.
    blank_np = False

    ant_np_word2vec = []
    ant_np_location = ant.get_context()['np']

    if ant_np_location != (-1,-1):
        ant_np_word2vec = get_average_np_vec(word2vec_dict, ant_sentdict, ant_np_location[0], ant_np_location[1])
    else: blank_np = True

    # Next we do the same for the Trigger NP.
    trig_np_word2vec = []
    trig_np_location = trigger.get_context()['np']

    if trig_np_location != (-1,-1):
        trig_np_word2vec = get_average_np_vec(word2vec_dict, trig_sentdict, trig_np_location[0], trig_np_location[1])
    else: blank_np = True

    # Adding the angle of the vector between the trigger NP and antecedent NP.
    if not blank_np:
        ant_length = vector_length(ant_np_word2vec)
        trig_length = vector_length(trig_np_word2vec)
        try:
            angle = angle_btwn_vectors(ant_np_word2vec, trig_np_word2vec, v1_length=ant_length, v2_length=trig_length)
        except ValueError:
            angle = 90.0

        vector.append(angle)
        vector.append(truth(angle == 0.0))
    else:
        vector.append(90.0)
        vector.append(truth(90.0 == 0.0))

    if not ant_np_word2vec:
        vector += [0 for _ in range(0,WORD2VEC_LENGTH)]
    else:
        vector += ant_np_word2vec
    if not trig_np_word2vec:
        vector += [0 for _ in range(0,WORD2VEC_LENGTH)]
    else:
        vector += trig_np_word2vec

    # Now for what comes after the head.
    ant_head_idx = get_antecedent_head_index(ant_sentdict, ant)
    ant_post_head_w2vec = get_average_np_vec(word2vec_dict, ant_sentdict, ant_head_idx, len(ant_sentdict['words']))

    # if not ant_post_head_w2vec: vector += [0 for i in range(0,WORD2VEC_LENGTH)]
    # else: vector += ant_post_head_w2vec

    stop_idx = len(trig_sentdict['words'])
    for i in range(trigger.get_idx(), len(trig_sentdict['words'])):
        if DV.ispunctuation(trig_sentdict['lemmas'][i]):
            stop_idx = i
            break

    post_trig_w2vec = get_average_np_vec(word2vec_dict, trig_sentdict, trigger.get_idx(), stop_idx)

    # if not post_trig_w2vec: vector += [0 for i in range(0,WORD2VEC_LENGTH)]
    # else: vector += post_trig_w2vec

    if ant_post_head_w2vec and post_trig_w2vec:
        try:
            post_angle = angle_btwn_vectors(ant_post_head_w2vec, post_trig_w2vec)
        except ValueError: post_angle = 90.0
        vector.append(post_angle)
        vector.append(truth(post_angle == 0.0))
    else:
        vector.append(90.0)
        vector.append(truth(90.0 == 0.0))

    # Sentenial complement check.
    tree = NT.maketree(ant_sentdict['tree'][0])
    if NT.dominates(tree, ant.get_subtree(), trigger.get_subtree()):
        vector.append(truth( NT.has_phrases_between_trees(ant.get_subtree(), trigger.get_subtree(), NIELSON_SENTENIAL_COMPLEMENT_PHRASES)))
    else:
        vector.append(truth(False))

    # Features to account for the number of each phrase type between the antecedent and trigger.
    phrases_between = [0 for _ in ALL_PHRASES]

    if ant.get_sentnum() == trigger.get_sentnum():
        for i in range(0,len(phrases_between)):
            if NT.has_phrases_between_trees(ant.get_subtree(), trigger.get_subtree(), [ALL_PHRASES[i]]):
                phrases_between[i] += 1

    vector += phrases_between
    vector.append(sum(phrases_between))

    return vector

def nielson_features(trig_sentdict, ant_sentdict, ant, trigger):
    vector = []

    if NT.dominates(ant.get_subtree().root(), ant.get_subtree(), trigger.get_subtree()):
        print

    return vector

def vector_length(v1,v2=None):
    if v2:
        return math.sqrt(np.dot(v1,v2))
    else:
        return math.sqrt(np.dot(v1,v1))

def angle_btwn_vectors(v1, v2, v1_length=None, v2_length=None):
    if not v1_length:
        v1_length = vector_length(v1)

    if not v2_length:
        v2_length = vector_length(v2)

    return math.acos( np.dot(v1,v2) / (v1_length * v2_length)) * 360.0 / 2.0 / np.pi

def get_average_np_vec(word2vec_dict, sentdict, np_start, np_end):
    ant_np_word2vec = []
    np_vecs = []

    for i in range(np_start, min(np_end,len(sentdict['words']))):
        np_vecs.append(get_w2vec_vector(word2vec_dict,sentdict['words'][i]))

    div = 0.0
    for vec in np_vecs:
        if vec: div += 1.0

    if div == 0.0:
        return []

    for i in range(0, WORD2VEC_LENGTH): # Each vector will have the same length.
        my_sum = 0.0
        for vec in np_vecs:
            if vec: my_sum += vec[i]
        ant_np_word2vec.append(my_sum/div)

    return ant_np_word2vec

def get_w2vec_vector(word2vec_dict, w):
    try:
        return word2vec_dict[w]
    except KeyError:
        return []


