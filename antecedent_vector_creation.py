import nltktree as nt
import numpy
import word2vec_functionality as w2v

def build_feature_vector(ant, trigger, all_sentences, all_tags):
    """
    @type ant: vpe_objects.Antecedent
    @type trigger: vpe_objects.Auxiliary
    """
    vector = [1] #bias
    vector += describe_ant(ant, all_sentences, all_tags)
    vector += describe_trigger(trigger, all_sentences, all_tags)
    vector += ant_trigger_relationship(ant, trigger, all_sentences, all_tags)
    return numpy.array(vector)

def describe_ant(ant, sentences, all_tags):
    """
    @type ant: vpe_objects.Antecedent
    """
    v = []
    v += encode_pos_tag_frequencies(ant.sub_sentdict.pos, all_tags) # 46 features
    v += nearest_ant_np(ant, sentences, all_tags) # Another 46 features
    return v

def describe_trigger(trig, sentences, all_tags):
    """
    @type trig: vpe_objects.Auxiliary
    """
    v = []
    v += nearest_trig_np(trig, sentences, all_tags)
    return v

def ant_trigger_relationship(ant, trigger, sentences, all_tags, word2vec_dict):
    v = []

    # pos tags
    ant_np = numpy.array(nearest_ant_np(ant, sentences, all_tags))
    trig_np = numpy.array(nearest_trig_np(trigger, sentences, all_tags))

    v += list(abs(ant_np - trig_np))

    if not ant_np.any() or not trig_np.any():
        val = 0.0
    else:
        val = numpy.dot(ant_np, trig_np) / (numpy.linalg.norm(ant_np) * numpy.linalg.norm(trig_np))

    v.append(val)

    # word2vec words
    ant_np_embedding = w2v.average_vec_for_list(nearest_ant_np(ant, sentences, all_tags, get_words=True), word2vec_dict)
    trig_np_embedding = w2v.average_vec_for_list(nearest_trig_np(trigger, sentences, all_tags, get_words=True), word2vec_dict)

    if ant_np_embedding and trig_np_embedding:
        v.append(1.0)
        v.append(w2v.angle_btwn_vectors(ant_np_embedding, trig_np_embedding))
    else:
        v.append(0.0)
        v.append(0.0)


    # Comparing the word that comes before the antecedent to the trigger word.
    if ant.start != 0:
        v.append(1.0)
        v.append(1.0 if sentences[ant.sentnum].words[ant.start-1] == trigger.word else 0.0)
        v.append(1.0 if sentences[ant.sentnum].lemmas[ant.start-1] == trigger.lemma else 0.0)
        v.append(1.0 if sentences[ant.sentnum].pos[ant.start-1] == trigger.pos else 0.0)
    else:
        v += [0.0, 0.0, 0.0, 0.0, 0.0]

    # Comparing the word that comes before the trigger to the word that comes before the antecedent.
    if trigger.wordnum != 0 and ant.start != 0:
        v.append(1.0)

        v.append(1.0 if sentences[ant.sentnum].words[ant.start-1] ==
            sentences[trigger.sentnum].words[trigger.wordnum-1] else 0.0)

        v.append(1.0 if sentences[ant.sentnum].lemmas[ant.start-1] ==
            sentences[trigger.sentnum].lemmas[trigger.wordnum-1] else 0.0)

        v.append(1.0 if sentences[ant.sentnum].pos[ant.start-1] ==
            sentences[trigger.sentnum].pos[trigger.wordnum-1] else 0.0)
    else:
        v += [0.0, 0.0, 0.0, 0.0]

    return v

def nearest_trig_np(trig, sentences, all_tags, get_words=False):
    """
    @type trig: vpe_objects.Auxiliary
    @type sentences: vpe_objects.AllSentences
    """
    t = sentences.get_sentence_tree(trig.sentnum)
    tree_tuples = nt.pos_word_tuples(t)
    all_nps = nt.find_subtree_phrases(t, ['NP','NP-PRD'])

    trig_tup = (trig.pos, trig.word)
    trig_tup_idx = tree_tuples.index(trig_tup)

    closest_np_value = 99
    closest_np = None
    for NP in all_nps:
        last_np_word_idx = tree_tuples.index(nt.pos_word_tuples(NP)[-1])
        if abs(trig_tup_idx - last_np_word_idx) < closest_np_value:
            closest_np_value = trig_tup_idx - last_np_word_idx
            closest_np = NP
    if closest_np == None:
        closest_np = t

    if get_words:
        try:
            return closest_np.leaves()
        except AttributeError:
            return []

    np_pos = [subtree.label() for subtree in nt.getsmallestsubtrees(closest_np)]

    return encode_pos_tag_frequencies(np_pos, all_tags)

def nearest_ant_np(ant, sentences, all_tags, get_words=False):
    """
    @type ant: vpe_objects.Antecedent
    @type sentences: vpe_objects.AllSentences
    """
    t = sentences.get_sentence_tree(ant.sentnum)
    tree_tuples = nt.pos_word_tuples(t)
    all_nps = nt.find_subtree_phrases(t, ['NP','NP-PRD'])

    ant_tup = (ant.sub_sentdict.pos[len(ant.sub_sentdict)/2], ant.sub_sentdict.words[len(ant.sub_sentdict)/2])
    if ant.sentnum != ant.trigger.sentnum:
        ant_tup_idx = len(tree_tuples)
    else:
        ant_tup_idx = tree_tuples.index(ant_tup)

    closest_np_value = 99
    closest_np = None
    for NP in all_nps:
        last_np_word_idx = tree_tuples.index(nt.pos_word_tuples(NP)[-1])
        if abs(ant_tup_idx - last_np_word_idx) < closest_np_value:
            closest_np_value = ant_tup_idx - last_np_word_idx
            closest_np = NP

    if get_words:
        try:
            return closest_np.leaves()
        except AttributeError:
            return []

    try:
        np_pos = [subtree.label() for subtree in nt.getsmallestsubtrees(closest_np)]
    except AttributeError:
        np_pos = []

    return encode_pos_tag_frequencies(np_pos, all_tags)

def encode_pos_tag_frequencies(pos_list, all_tags):
    """
    This is normalized.
    @features: 46
    """
    if len(pos_list) == 0:
        return [0.0 for _ in all_tags]

    return [float(pos_list.count(tag))/len(pos_list) for tag in all_tags]
