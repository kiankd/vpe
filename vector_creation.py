import nltktree as nt
import word_characteristics as wc
from old.detectVPE import modalcheck,becheck,havecheck,docheck,socheck,tocheck
import old.vectorcreation as old_vc

def bool_to_int(boole):
    if boole: return 1
    return 0

def int_to_bool(inte):
    return inte == 1

def get_all_features(old_rules=True):
    if old_rules: rules = 'old_rules'
    else: rules = 'my_rules'
    return ['words','pos','bigrams','my_features',rules,'square_rules','combine_aux_type']

def make_vector(sentdict, aux, features, aux_categories, aux_lemmas, aux_words, surrounding_words, pos_tags, pos_bigrams, make_old=False):

    vec = []
    if not make_old:
        vec += lemma_category_vector(aux, aux_categories)
        vec += lemma_vector(aux, aux_lemmas)
        vec += aux_vector(aux, aux_words)

        if 'words' in features:
            vec += aux_structure_vector(sentdict, aux, 'words', surrounding_words)
        if 'pos' in features:
            vec += aux_structure_vector(sentdict, aux, 'pos', pos_tags)
        if 'bigrams' in features:
            vec += aux_pos_bigrams_vector(sentdict, aux, pos_bigrams)

        vec += linguistic_features_vector(sentdict, aux, features)

    else:
        vec += old_vc.lemmacategoryvector(aux.lemma)
        vec += old_vc.lemmavector(aux.lemma)
        vec += old_vc.auxwordvector(aux.word, aux_words)
        vec += old_vc.myfeaturesvector(sentdict, aux.wordnum, features)

    return vec

""" Methods for categorizing the auxiliary. """
def lemma_category_vector(aux, categories):
    vector = []
    for category in categories:
        if aux.type in category:
            vector.append(bool_to_int(True))
        else:
            vector.append(bool_to_int(False))
    return vector

def lemma_vector(aux, lemmas):
    vector = []
    for lemma_type in lemmas:
        if aux.lemma == lemma_type:
            vector.append(bool_to_int(True))
        else:
            vector.append(bool_to_int(False))
    return vector

def aux_vector(aux, words):
    vector = []
    for aux_word in words:
        if aux.word == aux_word:
            vector.append(bool_to_int(True))
        else:
            vector.append(bool_to_int(False))
    return vector

""" Methods for representing the surrounding structure of an auxiliary. """
def aux_structure_vector(sentdict, aux, cat, data):
    vector = []
    for i in range(aux.wordnum-wc.WORD_DISTANCE_SEARCH, aux.wordnum+wc.WORD_DISTANCE_SEARCH+1):
        if i != aux.wordnum:
            vector += aux_index_struct_vector(i, sentdict, cat, data)
    return vector

def aux_index_struct_vector(idx, sentdict, cat, data):
    vector = []
    try:
        contender = sentdict[cat][idx]
    except IndexError:
        contender = None

    got = False or contender==None
    for i in range(0,len(data)):
        if got:
            vector.append(bool_to_int(False))
        elif contender == data[i]:
            got = True
            vector.append(bool_to_int(True))
        else:
            vector.append(bool_to_int(False))

    return vector

def aux_pos_bigrams_vector(sentdict, aux, pos_bigrams):
    vector = []
    for i in range(aux.wordnum-wc.WORD_DISTANCE_SEARCH, aux.wordnum+wc.WORD_DISTANCE_SEARCH):
        try:
            tag1 = sentdict.pos[i]
            tag2 = sentdict.pos[i+1]
        except IndexError:
            for _ in pos_bigrams:
                vector.append(bool_to_int(False))
            continue

        got = False
        for pair in pos_bigrams:
            if got:
                vector.append(bool_to_int(False))
            elif (tag1,tag2) == pair:
                got = True
                vector.append(bool_to_int(True))
            else:
                vector.append(bool_to_int(False))

    return vector

""" Methods for representing linguistic features. """
def linguistic_features_vector(sentdict, aux, features):
    vector = []
    tree = sentdict.get_nltk_tree()
    subtree_positions = nt.get_smallest_subtree_positions(tree)

    if 'my_features' in features:
        vector += my_features_vector(sentdict, aux, tree, subtree_positions)

    if 'my_rules' in features:
        vector += my_rules_vector(sentdict, aux, tree, subtree_positions)

    if 'old_rules' in features:
        vector += old_rules_vector(sentdict, aux, tree, subtree_positions)

    if 'square_rules' in features:
        vector_length = len(vector)
        for i in range(0, vector_length):
            for j in range(i+1, vector_length):
                vector.append(bool_to_int(int_to_bool(vector[i]) and int_to_bool(vector[j])))

    if 'combine_aux_type' in features:
        vector_length = len(vector)
        aux_type = aux.type
        bools = [aux_type == 'modal', aux_type == 'be', aux_type == 'have', aux_type == 'do', aux_type == 'to', aux_type == 'so']
        for i in range(0, vector_length):
            for b in bools:
                vector.append(bool_to_int(b and int_to_bool(vector[i])))

    return vector

def my_features_vector(sentdict, aux, tree, subtree_positions):
    vector = [bool_to_int(wc.aux_ccommands_verb(sentdict, aux, tree, subtree_positions)),
              bool_to_int(wc.aux_ccommands_verb_that_comes_after(sentdict, aux, tree, subtree_positions)),
              bool_to_int(wc.aux_ccommanded_by_verb(sentdict, aux, tree, subtree_positions)),
              bool_to_int(wc.aux_locally_ccommanded_by_verb(sentdict, aux, tree, subtree_positions)),
              bool_to_int(wc.aux_locally_ccommands_verb(sentdict, aux, tree, subtree_positions)),
              bool_to_int(wc.is_ccommanded_by_continuation_word(sentdict, aux, tree, subtree_positions)),
              bool_to_int(wc.next_to_punct(sentdict, aux)),
              bool_to_int(wc.previous_word_is_continuation_word(sentdict, aux)),
              bool_to_int(wc.the_same_check(sentdict, aux)),
              bool_to_int(wc.to_precedes_aux(sentdict, aux)),
              bool_to_int(wc.verb_follows_aux(sentdict, aux)),
              bool_to_int(wc.followed_by_too(sentdict, aux))]
    return vector

def my_rules_vector(sentdict, aux, tree, subtree_positions):
    aux_type = aux.type
    vector = [bool_to_int(aux_type == 'modal' and wc.modal_rule(sentdict, aux, tree, subtree_positions)),
              bool_to_int(aux_type == 'be' and wc.be_rule(sentdict, aux)),
              bool_to_int(aux_type == 'have' and wc.have_rule(sentdict, aux)),
              bool_to_int(aux_type == 'do' and wc.do_rule(sentdict, aux, tree, subtree_positions)),
              bool_to_int(aux_type == 'to' and wc.to_rule(sentdict, aux)),
              bool_to_int(aux_type == 'so' and wc.so_rule(sentdict, aux))]
    return vector

def old_rules_vector(sentdict, aux, tree, subtree_positions):
    aux_type = aux.type
    vector = [bool_to_int(aux_type == 'modal' and modalcheck(sentdict, aux.wordnum, tree, subtree_positions)),
              bool_to_int(aux_type == 'be' and becheck(sentdict, aux.wordnum, tree, subtree_positions)),
              bool_to_int(aux_type == 'have' and havecheck(sentdict, aux.wordnum, tree, subtree_positions)),
              bool_to_int(aux_type == 'do' and docheck(sentdict, aux.wordnum, tree, subtree_positions)),
              bool_to_int(aux_type == 'to' and tocheck(sentdict, aux.wordnum, tree, subtree_positions)),
              bool_to_int(aux_type == 'so' and socheck(sentdict, aux.wordnum, tree, subtree_positions))]
    return vector