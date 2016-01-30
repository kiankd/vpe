import detectVPE as DV
import nltktree as NT
from truth import *

# This is a file purely created for making the feature vectors for the SVM.
# @author Kian Kenyon-Dean

# For pinpointing the auxiliaries:
def lemmacategoryvector(lemma):
    category_vector = []
    for category in DV.ALL_CATEGORIES:
        if lemma in category: category_vector.append(truth(True))
        else: category_vector.append(truth(False))

    return category_vector
    
def lemmavector(lemma):
    vector = []
    for lemma_type in DV.VPE_TRIGGERS_IN_WSJ:
        if lemma == lemma_type: vector.append(truth(True))
        else: vector.append(truth(False))

    return vector

def auxwordvector(word, all_auxs):
    vector = []
    for aux in all_auxs:
        if word == aux: vector.append(truth(True))
        else: vector.append(truth(False))

    return vector

# For getting the surrounding structure.
def makestructvector(sentdict, idx, words, lemmas, postags, features):
    vector = []

    # This is the most general structure creator, it can take any list that corresponds to a key in
    # the sentdict. It then creates a very large vector of 0s, except there will be one 1, which indicates
    # what the word is, kind of like a universal dictionary where the kth entry represents the kth unique
    # word (or lemma, or postag, etc.) within a dataset.
    def makestructtypevector(sentdict, idx, key_in_sentdict, lst):
        vector = []

        try:
            test_val = sentdict[key_in_sentdict][idx]
            if key_in_sentdict == 'words': test_val = test_val.lower()

        except IndexError:
            test_val = '-~NONE~-'

        if test_val == '-~NONE~-':
            for val in lst: vector.append(truth(False))

        else:
            got = False
            for val in lst:
                if not got and val == test_val: vector.append(truth(True))
                else: vector.append(truth(False))

        return vector

    if 'words' in features: vector += makestructtypevector(sentdict, idx, 'words', words)
    # if not specific or specific=='lemmas': vector += makestructtypevector(sentdict, idx, 'lemmas', lemmas)
    if 'pos' in features: vector += makestructtypevector(sentdict, idx, 'pos', postags)

    return vector

def makeposbigramsvector(sentdict, auxidx, postags, combine=False):
    vector = []
    true_idxs = []

    for i in range(auxidx-3,auxidx+3):
        try:
            crtpos  = sentdict['pos'][i]
            nextpos = sentdict['pos'][i+1]

        except IndexError:
            for p1 in postags:
                for p2 in postags:
                    if not combine:
                        vector.append(truth(False))
            continue

        got = False
        count = 0
        for k in range(0,len(postags)):
            for j in range(0,len(postags)):
                count += 1
                if not got:
                    if crtpos==postags[k] and nextpos==postags[j]:
                        if not combine:
                            vector.append(truth(True))
                        else:
                            true_idxs.append(count)

                        got = True
                    else:
                        if not combine: vector.append(truth(False))
                else:
                    if not combine: vector.append(truth(False))

    if combine:
        length_of_bigrams_set = (len(postags)-1)**2
        for i in range(0, length_of_bigrams_set):
            if i in true_idxs:
                vector.append(truth(True) * true_idxs.count(i))
            else:
                vector.append(truth(False))

    return vector

def auxstruct(sentdict, auxidx, features):
    vector = []

    words,lemmas,postags = [],[],[]
    if 'words' in features: words = extract_data_from_file(EACH_UNIQUE_WORD_NEAR_AUX)
    # if not specific or specific=='lemmas': lemmas = extractdatafromfile(EACH_UNIQUE_LEMMA_FILE)
    if 'pos' in features: postags = extract_data_from_file(EACH_UNIQUE_POS_FILE)

    vector += makestructvector(sentdict, auxidx-3, words, lemmas, postags, features)
    vector += makestructvector(sentdict, auxidx-2, words, lemmas, postags, features)
    vector += makestructvector(sentdict, auxidx-1, words, lemmas, postags, features)
    vector += makestructvector(sentdict, auxidx+1, words, lemmas, postags, features)
    vector += makestructvector(sentdict, auxidx+2, words, lemmas, postags, features)
    vector += makestructvector(sentdict, auxidx+3, words, lemmas, postags, features)

    if 'bigrams' in features: vector += makeposbigramsvector(sentdict, auxidx, postags, combine='combine_bigrams' in features)

    return vector

# For the features I discovered on my own.
def myfeaturesvector(sentdict, idx, features):
    vector = []

    tree = NT.maketree(sentdict['tree'][0])
    subtrees = NT.getsmallestsubtrees(tree)
    subtree_positions = NT.getsmallestsubtreepositions(tree, subtree_list = subtrees)
    aux = sentdict['lemmas'][idx]

    if 'my_features' in features:
        vector.append(truth(DV.auxccommandsverb(sentdict, idx, tree, subtree_positions)))
        vector.append(truth(DV.auxccommandsverbthatcomesafter(sentdict, idx, tree, subtree_positions)))
        vector.append(truth(DV.auxisccommandedbyverb(sentdict, idx, tree, subtree_positions)))
        vector.append(truth(DV.auxislocallyccommandedbyverb(sentdict, idx, tree, subtree_positions)))
        vector.append(truth(DV.auxlocallyccommandsverb(sentdict, idx, tree, subtree_positions)))
        vector.append(truth(DV.isccommandedbycontinuationword(sentdict, idx, tree, subtree_positions)))
        vector.append(truth(DV.nexttopunct(sentdict, idx, tree, subtree_positions)))
        vector.append(truth(DV.isfollowedbypunct(sentdict, idx, end=['.'])))
        vector.append(truth(DV.previouswordisasorsoorthan(sentdict['words'], idx)))
        vector.append(truth(DV.thesamecheck(sentdict['words'], idx)))
        vector.append(truth(DV.toprecedesaux(sentdict, idx)))
        vector.append(truth(DV.verbfollowsaux(sentdict, idx)))

        # TODO: added this new feature!
        vector.append(truth(DV.nextwordistoo(sentdict, idx)))


    if 'my_rules' in features:
        vector.append(truth(aux in DV.MODALS and DV.modalcheck(sentdict, idx, tree, subtree_positions) ) )
        vector.append(truth(aux in DV.BE and DV.becheck(sentdict, idx, tree, subtree_positions) ) )
        vector.append(truth(aux in DV.HAVE and DV.havecheck(sentdict, idx, tree, subtree_positions) ) )
        vector.append(truth(aux in DV.DO and DV.docheck(sentdict, idx, tree, subtree_positions) ) )
        vector.append(truth(aux in DV.TO and DV.tocheck(sentdict, idx, tree, subtree_positions) ) )
        vector.append(truth(aux in DV.SO and DV.socheck(sentdict, idx, tree, subtree_positions) ) )

        # This adds a new layer of features by combining all of the ones I had.
    if 'square_rules' in features:
        size = len(vector)
        for i in range(0, size):
            for j in range(0, size):
                if i != j: vector.append(truth( untruth(vector[i]) and untruth(vector[j]) ))

    if 'combine_aux_type' in features:
        bools = [aux in DV.MODALS, aux in DV.BE, aux in DV.HAVE, aux in DV.DO, aux in DV.TO, aux in DV.SO]
        vec = [v for v in vector]
        for v in vec:
            for b in bools:
                vector.append(truth( untruth(v) and b ))


    return vector

def verblocativevector(sentdict, auxidx):
    vector = []        
    verb_locations = []
    num_auxiliaries = 0
    
    closest = 99
    for i in range(0, len(sentdict['pos'])):
        if DV.isverb(sentdict['pos'][i]) and not i==auxidx:
            verb_locations.append(i)
            closest = min(closest, abs(auxidx-i))
        if sentdict['lemmas'][i] in DV.VPE_TRIGGERS_IN_WSJ:
            num_auxiliaries += 1
    
    # The first feature is the distance between the Auxiliary and the closest verb.
    if closest!=99: vector.append(closest)
    else: vector.append(truth(False))
    
    # Distance between auxiliary and closest previous verb.    
    closest = 99    
    for idx in verb_locations:
        if idx < auxidx:
            closest = min(closest, abs(auxidx-i))
    if closest!=99: vector.append(closest)
    else: vector.append(truth(False))
            
    # Distance between auxiliary and closest following verb.    
    closest = 99    
    for idx in verb_locations:
        if idx > auxidx:
            closest = min(closest, abs(auxidx-i))
    if closest!=99: vector.append(closest)
    else: vector.append(truth(False))
    
    # This next feature is the number of verbs in the auxiliary's sentence.
    vector.append(len(verb_locations))
    
    # This feature is the number of auxiliary's in the sentence.
    vector.append(num_auxiliaries)    
    
    return vector

def word2vecvectors(sentdict, auxidx, word2vec_dict, average=False):
    vecs = []

    # Here we are getting all of the word2vec vectors from the file. (RIGHT NOW I AM INCLUDING THE AUXILILARY).
    for i in range(auxidx-3, auxidx+3):
        if i == i: # TODO: CHANGE THIS BACK TO NOT INCLUDE AUXIDX IF NO CHANGE
            try: crt_word = sentdict['words'][i].lower()
            except IndexError:
                vecs.append([])
                continue
            vecs.append(get_w2vec_vector(word2vec_dict, crt_word))

    # Here we are averaging the (at most) 6 vectors to return one single vector.
    return_vector = []

    if average:
        for i in range(0, WORD2VEC_LENGTH): # Each vector will have the same length.
            my_sum = 0.0
            for vec in vecs:
                if vec: my_sum += vec[i]
            return_vector.append(my_sum/float(len(vecs)))
    else:
        for i in range(0, len(vecs)):
            if not vecs[i]:
                return_vector += [0 for j in range(0, WORD2VEC_LENGTH)]
            else:
                for k in vecs[i]:
                    return_vector.append(k)

    return return_vector

def get_w2vec_vector(word2vec_dict, w):
    try:
        return word2vec_dict[w]
    except KeyError:
        return []

# This gets rid of unnecessary columns in the feature vector list by deleting column in the matrix
# such that there is never a 1 occupying it.
def cleanvectors(vectors, columns_to_delete):

    cleaned_vectors = []
    # We can do what is commented while creating the vectors.
    """
    columns_to_delete = set()

    first = True
    for vec in vectors:
        vec_empties = set()

        for i in range(0, len(vec)):
            if vec[i] == truth(False):
                vec_empties.add(i)

        if first:
            columns_to_delete = vec_empties
            first = False

        else:
            columns_to_delete.intersection_update(vec_empties)
    """

    for vec in vectors:
        cleaned_vectors.append([vec[i] for i in range(0, len(vec)) if not i in columns_to_delete])

    return cleaned_vectors

# Helpful analysis functions:
def counttruth(vector):
    count = 0
    for val in vector:
        if val == truth(True): count += 1
    return count
