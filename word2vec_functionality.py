import numpy as np
import math
from truth import WORD2VEC_LENGTH

def vector_length(v):
    """
    :param v: List of floats
    :return: Float, the L2 norm of the vector.
    """
    return math.sqrt(np.dot(v,v))

def angle_btwn_vectors(v1, v2):
    """
    :param v1: List of floats
    :param v2: List of floats
    :return: Float - angle between the two vectors.
    """
    v1_length = vector_length(v1)
    v2_length = vector_length(v2)

    try:
        return math.acos( np.dot(v1,v2) / (v1_length * v2_length)) * 360.0 / 2.0 / np.pi
    except ValueError:
        return 90.0

def get_vec(word, word2vec_dict):
    """
    :param word: String we want the vector for.
    :param word2vec_dict: Word2vec dictionary
    :return: List, either the vector or an empty one if no key.
    """
    try:
        assert len(word2vec_dict[word]) == WORD2VEC_LENGTH
        return word2vec_dict[word]

    except KeyError:
        return []

def average_vec_for_list(words, word2vec_dict):
    """
    :param words: List of words.
    :param word2vec_dict: Word2vec dictionary
    :return: List that is the average word2vec vector for each word in words.
    """

    avg = []
    word_vecs = []

    keys_present = 0
    for word in words:
        v = get_vec(word, word2vec_dict)
        if v:
            keys_present += 1
            word_vecs.append(v)

    if keys_present == 0:
        return []

    for i in range(WORD2VEC_LENGTH):
        summ = 0.0
        for vec in word_vecs:
            summ += vec[i]
        avg.append( summ / float(WORD2VEC_LENGTH) )

    return avg


