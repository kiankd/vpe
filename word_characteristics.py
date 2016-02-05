import nltktree as nt

WORD_DISTANCE_SEARCH = 3
CONTINUATION_WORD_DISTANCE_SEARCH = 5

""" Basic Boolean functions. """
def in_to_or_so(w):
    TO = ['to']
    SO = ['so','same','likewise','opposite']
    return w in TO or w in SO

def is_auxiliary(sentdict, i, AUX_LEMMAS, ALL_AUXILIARIES, raw=False):
    if not raw:
        # return sentdict.lemmas[i] in AUX_LEMMAS and (is_verb(sentdict.pos[i]) or in_to_or_so(sentdict.words[i]))
        return is_aux_lemma(sentdict.lemmas[i]) and (is_verb(sentdict.pos[i]) or in_to_or_so(sentdict.words[i]))
    else:
        return sentdict.words[i] in ALL_AUXILIARIES

def is_aux_lemma(lemma):
    return is_modal(lemma) or is_be(lemma) or is_have(lemma) or is_do(lemma) or is_to(lemma) or is_so(lemma)

def is_verb(pos_tag):
    return pos_tag in ['VB','VBD','VBG','VBN','VBP','VBZ','MD']

def is_modal(lemma):
    return lemma in ['can','could','may','must','might','will','would','shall','should']

def is_be(lemma):
    return lemma == 'be'

def is_have(lemma):
    return lemma == 'have'

def is_do(lemma):
    return lemma == 'do'

def is_to(lemma):
    return lemma == 'to'

def is_so(lemma):
    return lemma in ['so','same','likewise','opposite']

def is_period(tag):
    return tag == '.'

def is_comma(tag):
    return tag == ','

def is_dash_or_colon(tag):
    return tag == ':'

def is_preposition(tag):
    return tag == 'IN'

def is_adjective(tag):
    return tag in ['JJ','JJR','JJS']

def is_predicative(tag):
    return 'PRD' in tag

def is_noun(tag):
    return tag in ['NN','NNS','NNP','NNPS','WP','PRP','PRP$','DT']

def is_adverb(tag):
    return tag == 'RB'

def is_determiner(tag):
    return tag == 'DT'

def is_punctuation(tag):
    return tag in ['.',',','-','--','\'','\"',':']

""" My features. """
def aux_ccommands_verb(sentdict, aux, tree, word_positions_in_tree):
    subtrees = nt.getsmallestsubtrees(tree)

    for subtree in subtrees:
        if is_verb(subtree.label()):
            try:
                if nt.ccommands(tree, tree[word_positions_in_tree[aux.wordnum-1]], subtree):
                    return True
            except IndexError:
                pass
    return False

def aux_ccommands_verb_that_comes_after(sentdict, aux, tree, word_positions_in_tree):
    for i in range(aux.wordnum+1, len(sentdict)):
        if is_verb(sentdict.pos[i]):
            if nt.ccommands(tree, tree[word_positions_in_tree[aux.wordnum-1]], tree[word_positions_in_tree[i-1]]):
                return True
    return False

def aux_ccommanded_by_verb(sentdict, aux, tree, word_positions_in_tree):
    subtrees = nt.getsmallestsubtrees(tree)

    for subtree in subtrees:
        if is_verb(subtree.label()):
            try:
                if nt.ccommands(tree, subtree, tree[word_positions_in_tree[aux.wordnum-1]]):
                    return True
            except IndexError:
                pass
    return False

def aux_locally_ccommanded_by_verb(sentdict, aux, tree, word_positions_in_tree):
    try:
        localt = nt.generatelocalstructurefromsubtree(tree, tree[word_positions_in_tree[aux.wordnum-1]])
        local_word_subtrees =  nt.getsmallestsubtrees(localt)

        for subtree in local_word_subtrees:
            if is_verb(subtree.label()):
                if nt.ccommands(localt, subtree, tree[word_positions_in_tree[aux.wordnum-1]])\
                        and not nt.ccommands(localt, tree[word_positions_in_tree[aux.wordnum-1]], subtree):
                    return True
    except IndexError: pass
    return False

def aux_locally_ccommands_verb(sentdict, aux, tree, word_positions_in_tree):
    try:
        localt = nt.generatelocalstructurefromsubtree(tree, tree[word_positions_in_tree[aux.wordnum-1]])
        local_word_subtrees = nt.getsmallestsubtrees(localt)

        for subtree in local_word_subtrees:
            if is_verb(subtree.label()):
                if nt.ccommands(localt, tree[word_positions_in_tree[aux.wordnum-1]], subtree)\
                        and not nt.ccommands(localt, subtree, tree[word_positions_in_tree[aux.wordnum-1]]):
                    return True
    except IndexError: pass
    return False

def is_ccommanded_by_continuation_word(sentdict, aux, tree, word_positions_in_tree):
    for i in range(max(0,aux.wordnum-CONTINUATION_WORD_DISTANCE_SEARCH),aux.wordnum):
        crt_word = sentdict.words[i].lower()
        if crt_word in ['than','as','so']:
            if nt.ccommands(tree, tree[word_positions_in_tree[i-1]], tree[word_positions_in_tree[aux.wordnum-1]]):
                return True
    return False

def next_to_punct(sentdict, aux):
    try:
        tag = sentdict.pos[aux.wordnum+1]
        if is_period(tag) or is_comma(tag) or is_dash_or_colon(tag):
            return True
    except IndexError:
        return False
    try:
        tag2 = sentdict.pos[aux.wordnum+2]
        if sentdict.lemmas[aux.wordnum+1] == 'not' and (is_period(tag2) or is_comma(tag2) or is_dash_or_colon(tag2)):
            return True
    except IndexError:
        return False

    return False

def previous_word_is_continuation_word(sentdict, aux):
    try:
        return sentdict.words[aux.wordnum-1].lower() in ['as','so','than']
    except IndexError:
        return False

def the_same_check(sentdict, aux):
    try:
        return sentdict.words[aux.wordnum+1].lower() == 'the' and sentdict.words[aux.wordnum+2].lower() == 'same'
    except IndexError:
        return False

def to_precedes_aux(sentdict, aux):
    try:
        return sentdict.words[aux.wordnum-1].lower() == 'to'
    except IndexError:
        return False

def verb_follows_aux(sentdict, aux):
    for i in range(aux.wordnum+1, len(sentdict)):
        if is_verb(sentdict.pos[i]):
            return True
    return False

def followed_by_too(sentdict, aux):
    try:
        return sentdict.words[aux.wordnum+1].lower() == 'too'
    except IndexError:
        return False

""" My Rules. """
def modal_rule(sentdict, aux, tree, word_positions_in_tree):
    if to_precedes_aux(sentdict, aux):
        return False
    if aux_ccommands_verb_that_comes_after(sentdict, aux, tree, word_positions_in_tree):
        return False
    return True

def be_rule(sentdict, aux):
    bad_words = ['being', 'been']
    if to_precedes_aux(sentdict, aux):
        return False
    try:
        if sentdict.words[aux.wordnum-1].lower() == 'that' and sentdict.words[aux.wordnum].lower() == 'is':
            return False
    except IndexError:
        pass

    try:
        if is_period(sentdict.lemmas[aux.wordnum+1]) and not sentdict.words[aux.wordnum] in bad_words:
            return True
    except IndexError:
        pass
    return False

def have_rule(sentdict, aux):
    if to_precedes_aux(sentdict, aux):
        return False
    try:
        if sentdict.words[aux.wordnum+1] == 'a' or sentdict.words[aux.wordnum+1]=='to':
            return False
    except IndexError:
        pass

    return next_to_punct(sentdict, aux)

def do_rule(sentdict, aux, tree, word_positions_in_tree):
    auxidx = aux.wordnum

    try:
        if sentdict.lemmas[auxidx+1] == 'that':
            return True
    except IndexError: pass

    if not aux_locally_ccommanded_by_verb(sentdict, aux, tree, word_positions_in_tree):
        if to_precedes_aux(sentdict, aux): return False

        localt = nt.generatelocalstructurefromsubtree(tree, tree[word_positions_in_tree[auxidx-1]])
        local_word_subtrees = nt.getsmallestsubtrees(localt)

        try:
            checkpuncttag = sentdict.pos[auxidx+1]
            if is_period(checkpuncttag) or is_comma(checkpuncttag) or is_dash_or_colon(checkpuncttag):
                endbool = True

                for subtree in local_word_subtrees:
                    if is_verb(subtree.label()) and subtree != tree[word_positions_in_tree[auxidx-1]]:
                        if nt.ccommands(localt, subtree, tree[word_positions_in_tree[auxidx-1]]):
                            endbool = False
                            break
                if endbool:
                    return endbool
        except IndexError:
            pass

        # Don't at the end of sentence.
        try:
            checkpuncttag = sentdict.pos[auxidx+2]
            if sentdict.lemmas[auxidx+1] == 'not' and (is_period(checkpuncttag) or is_comma(checkpuncttag) or is_dash_or_colon(checkpuncttag)):
                endbool = True
                for subtree in local_word_subtrees:
                    if is_verb(subtree.label()):
                        if nt.ccommands(localt, subtree, tree[word_positions_in_tree[auxidx-1]]):
                            endbool = False
                            break
                if endbool:
                    return endbool
        except IndexError:
            pass

        if is_ccommanded_by_continuation_word(sentdict ,aux, tree, word_positions_in_tree):
            return True

        if verb_follows_aux(sentdict, aux):
            return False

        try:
            if is_preposition(sentdict.pos[auxidx+1]) and sentdict.words[auxidx] != 'done':
                return True
        except IndexError:
            pass

    return False

def to_rule(sentdict, aux):
    speakinglemmas = ['say','acknowledge']
    auxidx = aux.wordnum
    try:
        if is_period(sentdict.pos[auxidx+1]):
            return True
    except IndexError:
        pass
    if len(sentdict) > auxidx+3:
        if sentdict.words[auxidx+1] == ',':
            for i in range(auxidx+2,len(sentdict)):
                if sentdict.lemmas[i] in speakinglemmas:
                    return True
    return False

def so_rule(sentdict, aux):
    auxidx = aux.wordnum
    if to_precedes_aux(sentdict, aux):
        return False

    try:
        if sentdict.lemmas[auxidx-1] == 'do' or sentdict.words[auxidx-1] == 'be':
            if not is_adjective(sentdict.pos[auxidx+1]):
                return True
    except IndexError:
        pass

    try:
        if sentdict.lemmas[auxidx-1] == 'the' and (sentdict.lemmas[auxidx-2] == 'do' or sentdict.words[auxidx-1] == 'be'):
            if not is_noun(sentdict.pos[auxidx+1]):
                return True
    except IndexError:
        pass

    return False

""" Structural functions. """
def pos_bigrams(pos_tags):
    """ Makes a list of all of the POS bigrams. """
    bigrams = []
    for tag1 in pos_tags:
        for tag2 in pos_tags:
            bigrams.append((tag1,tag2))
    return bigrams
