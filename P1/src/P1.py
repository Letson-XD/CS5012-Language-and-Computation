from io import open
from conllu import parse_incr
from nltk import FreqDist, WittenBellProbDist, ngrams
import operator

#   The corpus repositories.
treebank = {}
treebank['en'] = 'UD_English-GUM/en_gum'
treebank['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
treebank['uk'] = 'UD_Ukrainian-IU/uk_iu'

def train_corpus(lang):
    return treebank[lang] + '-ud-train.conllu'

def test_corpus(lang):
    return treebank[lang] + '-ud-test.conllu'

# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]

# Generates the sentences for corpus
def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]

# Generates a list of sentences of words and a list of sentences or pos tags for the given list of sentences or tuples
# Adds the <s> and </s> based on delim
def generate_lists(sents, delim):
    word_list = []
    pos_tag_list = []
    for sent in sents:
        sent_word = []
        pos_sent = []
        if delim:
            sent_word.append('<s>')
            pos_sent.append('<s>')

        for token in sent:
            sent_word.append(token['form'])
            pos_sent.append(token['upos'])

        if delim:
            sent_word.append('</s>')
            pos_sent.append('</s>')

        word_list.append(sent_word)
        pos_tag_list.append(pos_sent)
    return word_list, pos_tag_list

# Builds ngrams based on the provided sentences 
def build_ngrams(sents):
    n_grams = []
    for sent in sents:
        bigrams = ngrams(sent, 2)
        n_grams.append(list(bigrams))
    return n_grams

# Gets the final tags the trellis for a given sentence
def getTagsFromTrellis(matrix, sent):
    finalTags=[]
    pointer = ""
    for i in range(1, len(sent)+1):
        max = 0
        maxID = 0
        if i == 1:
            for j in range(0, len(unique_tags)):
                if matrix[-i][j][0] > max:
                    max = matrix[-i][j][0]
                    maxID = j
            finalTags.append(unique_tags[maxID])
            pointer = matrix[-i][maxID][1]
        else:
            pointerID = unique_tags.index(pointer)
            finalTags.append(unique_tags[pointerID])
            pointer = matrix[-i][pointerID][1]
    finalTags.reverse()
    return finalTags

# Print the results based on the correct answers over the total
def printResults(finalTags, test_pos_tag_list, lang):
    correct = 0
    total = 0
    for s in range(0, len(test_pos_tag_list)):
        for t in range(0, len(test_pos_tag_list[s])):
            if test_pos_tag_list[s][t] == finalTags[s][t]:
                correct = correct+1
            total = total + 1
    print(lang + ":")
    print(correct/total)

# Runs the Viterbi algorithm to produce the trellis
def viterbi(test_word_list, unique_tags, tagMap, wordMap):
    finalTags=[]
    probMatrix = []

    for sentence in test_word_list:
        firstRun = True
        for word in sentence:
            col = []
            for tag in unique_tags:
                if firstRun:
                    pT = tagMap['<s>'].prob(tag)
                    pW = wordMap[tag].prob(word)
                    col.append([pW*pT, "q0"])
                else:
                    tag_Map = {}
                    for u_tag in range(0, len(unique_tags)):
                        pT = tagMap[unique_tags[u_tag]].prob(tag)
                        pW = wordMap[tag].prob(word)
                        tag_Map[unique_tags[u_tag]] = pT * pW * probMatrix[-1][u_tag][0]
                    prevBestTag = max(tag_Map.items(), key=operator.itemgetter(1))[0]
                    value = max(tag_Map.items(), key=operator.itemgetter(1))[1]
                    col.append([value, prevBestTag])
            firstRun = False
            probMatrix.append(col)
        finalTags.append(getTagsFromTrellis(probMatrix, sentence))
    return finalTags

# Helper function to get all the necessary for the Viterbi algorithm
def getData(train_corpus, test_corpus, lang):
    train_sents = conllu_corpus(train_corpus(lang))
    test_sents = conllu_corpus(test_corpus(lang))

    train_word_list, train_pos_tag_list = generate_lists(train_sents, True)
    test_word_list, test_pos_tag_list = generate_lists(test_sents, False)
    return train_word_list, train_pos_tag_list, test_word_list, test_pos_tag_list

# Builds the probability distribution for the Viterbi algorithm
def buildDist(unique_tags, train_pos_tag_n_grams, train_word_n_grams):
    tagMap = {}
    wordMap = {}

    for tag in unique_tags:
        tagList=[]
        wordList=[]
        for i in range(len(train_pos_tag_n_grams)):
            for j in range(len(train_pos_tag_n_grams[i])):
                if train_pos_tag_n_grams[i][j][0] == tag:
                    tagList.append(train_pos_tag_n_grams[i][j][1])
                    wordList.append(train_word_n_grams[i][j][0])
        tagMap[tag] = WittenBellProbDist(FreqDist(tagList), bins=1e5)
        wordMap[tag] = WittenBellProbDist(FreqDist(wordList), bins=1e5)
    return tagMap, wordMap

# GEts all the unique words seen within the given list
def getUniques(list):
    uniques = set()
    for item in list:
        for pos in item:
            uniques.add(pos)
    return uniques

# Gets the new tag for the unknown and hapax legomenon
def getNewTag(lang, word):
    if lang == "en":
        if word.endswith("ly"): # Adverb
            return "UNK-ly"
        if word.startswith("in"): # adjective, adverb or verb
            return "in-UNK"
        if word[0].isupper(): # proper pronoun
            return "C-UNK"
        if word.endswith("ed"): # verb or adjective
            return "UNK-ed"
        if word.endswith("ness"): # noun
            return "UNK-ness"
        if word.endswith("ing"): # verb or adjective
            return "UNK-ing"
        if word.endswith("ful"): # verb or adjective
            return "UNK-ful"
        if word.startswith("dis"): # verb or adjective
            return "dis-UNK"
        if word.startswith("mis"): # verb or adjective
            return "mis-UNK"
        if word.startswith("re"): # verb
            return "re-UNK"
        else:
            return "UNK"
    if lang == 'fr':
        if word.endswith("al"): # adjective or noun
            return "UNK-al"
        if word.endswith("ion"): # noun
            return "UNK-ion"
        if word.endswith("er"): # verb or noun
            return "UNK-er"
        if word.endswith("eur"): # noun
            return "UNK-eur"
        if word.endswith("ment"): # adverb
            return "UNK-ment"
        if word.endswith("e"): # adjective or verb
            return "UNK-e"
        else:
            return "UNK"
    if lang == 'uk':
        if word.endswith("ння"): # noun
            return "UNK-ння"
        if word.endswith("тва"): # noun
            return "UNK-тва"
        if word.endswith("ати"): # verb
            return "UNK-ати"
        if word.endswith("ити"): # verb
            return "UNK-ити"
        if word.endswith("ти"): # verb
            return "UNK-ти"
        if word.endswith("ють"): # verb
            return "UNK-ють"
        if word.endswith("ий"): # adjective
            return "UNK-ий"
        if word.endswith("о"): # adjective
            return "UNK-о"
        if word.endswith("я"): # adjective
            return "UNK-я"
        if word.endswith("и"): # adjective
            return "UNK-и"
        else:
            return "UNK"

# Finds the hapax legomenon for the given word list
def modifyHapaxLegomenon(word_list):
    flattened_words = {}
    for i in range(len(word_list)):
        for j in range(len(word_list[i])):
            if word_list[i][j] in flattened_words:
                flattened_words[word_list[i][j]] += 1
            else:
                flattened_words[word_list[i][j]] = 1

    for i in range(len(word_list)):
        sent_len = len(word_list[i])
        for j in range(sent_len):
            if flattened_words[word_list[i][j]] == 1:
                word_list[i][j] = getNewTag(lang, word_list[i][j])

langs = ['fr', 'en', 'uk']

for lang in langs:

    train_word_list, train_pos_tag_list, test_word_list, test_pos_tag_list = getData(train_corpus, test_corpus, lang)
    unique_tags = getUniques(train_pos_tag_list)
    unique_training_words = getUniques(train_word_list)
    unique_testing_words = getUniques(test_word_list)
    unknown_words = []
    for word in unique_testing_words:
        if word not in unique_training_words:
            unknown_words.append(word)
    modifyHapaxLegomenon(train_word_list)
    for i in range(len(test_word_list)):
        for j in range(len(test_word_list[i])):
            if test_word_list[i][j] in unknown_words:
                test_word_list[i][j] = getNewTag(lang, test_word_list[i][j])
    train_pos_tag_n_grams = build_ngrams(train_pos_tag_list)
    train_word_n_grams = build_ngrams(train_word_list)
    tagMap, wordMap = buildDist(unique_tags, train_pos_tag_n_grams, train_word_n_grams)
    unique_tags.remove('<s>')
    unique_tags.remove('</s>')
    unique_tags = list(unique_tags)
    finalTags = viterbi(test_word_list, unique_tags, tagMap, wordMap)
    printResults(finalTags, test_pos_tag_list, lang)
