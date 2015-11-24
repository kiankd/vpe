# Author: Kian Kenyon-Dean
# This file contains code that performs an alignment between the trigger's context and
# the context of each potential antecedent. This alignment is then turned into a feature marix that
# we pass to MIRA.
#
# The context is the list of chunked dependencies within the realm of the trigger/antecedent's
# nearest clause.
from nltktree import get_nearest_clause

def alignment_matrix(sentences, trigger, dep_names=('prep','adv','dobj','nsubj','nmod')):
    """
    @type sentences: vpe_objects.AllSentences
    @type trigger: vpe_objects.Auxiliary
    """

    trig_sentdict = sentences.get_sentence(trigger.sentnum)
    i,j = nearest_clause(trig_sentdict, trigger.wordnum)
    trig_chunks = trig_sentdict.chunked_dependencies(i,j,dep_names=dep_names)

    m = []
    for ant in trigger.possible_ants:
        ant_sentdict = sentences.get_sentence(ant.sentnum)

        k,l = nearest_clause(ant_sentdict, ant.start, end=ant.end)
        ant_chunks = ant_sentdict.chunked_dependencies(k,l,dep_names=dep_names)

        m.append(alignment_vector(trig_chunks, ant_chunks))

    return

def alignment_vector(t_chunks, a_chunks):
    return

def similarity_score(c1, c2):
    """
        Scores the similarity between 2 chunks.
        If both chunks have the same dependency then they get a high score.
    """
    return

def remove_idxs(chunks, start_idx, end_idx):
    """This removes the trigger and antecedent from the chunks."""
    for chunk in chunks:
        indexes_to_remove = range(start_idx,end_idx)
        start_remove = None
        end_remove = 0
        for i in range(len(chunk['sentdict'])):
            if i + chunk['sentdict'].start in indexes_to_remove:
                if not start_remove:
                    start_remove = i
                end_remove = i
        if not start_remove:
            continue
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
    start,end = -1,-1
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
