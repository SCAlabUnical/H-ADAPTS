#!/usr/bin/env python
import random
import re
from pickle import dump
from pickle import load as pkload
import numpy as np
import spacy
import constants as c
import word_embedding_model as emb

URL_REGEX = r"@\w*|https?:?\/?\/?[\w.\/]*|https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
not_lemmatize = []
REMOVE_WORDS = ['rt', 'ht', 'htt', 'https', 'http', 'https t']
MINCOUNT = c.MINCOUNT
TRAIN_TEST_INPUT = c.TRAIN_TEST_INPUT
GUSE_PATH = c.GUSE_PATH


def cleaning(words_doc):
    txt = []
    is_hashtag = False
    for token in words_doc:
        if token.text == "#":
            is_hashtag = True
        elif not (token.is_stop or (len(token.text) < 2 and token.text not in not_lemmatize)):
            if is_hashtag:
                txt.append('#' + token.text)
                is_hashtag = False
            else:
                txt.append(token.text if token.text in not_lemmatize else token.lemma_)
    return ' '.join(txt)


def clean_and_phrase(corpus_tweets):
    # Removes non-alphabetic characters:
    corpus_cleaned = []
    for tweet in corpus_tweets:
        tweet_cleaned = []
        tweet = re.sub(URL_REGEX, '', ' '.join(tweet)).lower().strip()
        tweet = re.sub('[^#\\d\\w_]+', ' ', tweet).strip()
        tweet = tweet.split()
        for word in tweet:
            word_cleaned = word
            for r in REMOVE_WORDS:
                if r == word_cleaned:
                    word_cleaned = ''
            if word_cleaned != '':
                tweet_cleaned.append(word_cleaned)
        corpus_cleaned.append(' '.join(tweet_cleaned))
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])
    txt = [cleaning(words_doc) for words_doc in nlp.pipe(corpus_cleaned, batch_size=5000, n_threads=-1)]
    return txt


def store(corpus, file):
    """
        Store list of list sentences in text file

        :param1 file.
        :return: corpus represented in list of list of words
    """
    with open(file, 'w', encoding="utf8") as stream_out:
        for line in corpus:
            for word in line:
                stream_out.write(str(word) + ' ')
            stream_out.write('\n')


def load(file):
    """
        Load text file in list of list sentences

        :param1 file.
        :return: corpus represented in list of list of words
    """
    sentences = []
    with open(file, "r", encoding="utf8") as stream_in:
        for line in stream_in:
            sentences.append(line.split())
    return sentences


def preprocess_data(corpus_dataframe):
    """
        Read data, clean data, calculate bigrams and store in save_file for Word2Vec model
        :param1 corpus_dataframe.
        :param2 save_file.
    """
    tweet_corpus = corpus_dataframe['text']
    sentences = clean_and_phrase(tweet_corpus)
    return list(map(lambda x: x.split(), sentences))


def preprocess_data_for_sentence_embedding(corpus_dataframe):
    """
        Clean sentences and stores them in file for Google Universal Sentence Encoder

        :param1 file: specify input file in order to not use the whole input folder.
    """
    tweet_corpus = corpus_dataframe['text']
    corpus_cleaned = []
    for tweet in tweet_corpus:
        tweet_cleaned = []
        tweet = re.sub(URL_REGEX, '', ' '.join(tweet)).strip()
        tweet = re.sub(r'[^\x00-\x7f]', r' ', tweet).lower().strip()
        tweet = tweet.split()
        for word in tweet:
            word_cleaned = word
            for r in REMOVE_WORDS:
                if r == word_cleaned:
                    word_cleaned = ''
            if word_cleaned != '':
                tweet_cleaned.append(word_cleaned)
        corpus_cleaned.append(tweet_cleaned)
    store(corpus_cleaned, TRAIN_TEST_INPUT)


def generate_sample(perc):
    sents = load(c.TRAIN_TEST_INPUT)
    with open("sample_" + str(perc), 'w', encoding="utf8") as out:
        for s in sents:
            r = random.random()
            if r < perc:
                for word in s:
                    out.write(word + ' ')
                out.write('\n')


def load_without_not_relevant_hts(input):
    counter = dict()
    with open(input, 'r', encoding="utf8") as in_stream:
        for line in in_stream:
            l = line.split()
            for w in l:
                if '#' in w:
                    counter[w] = counter.get(w, 0) + 1
    result = []
    with open(input, 'r', encoding="utf8") as in_stream:
        for line in in_stream:
            l = line.split()
            res_line = []
            for w in l:
                if '#' in w:
                    if counter[w] > MINCOUNT:
                        res_line.append(w)
                else:
                    res_line.append(w)
            if len(res_line) > 0:
                result.append(res_line)
    return result


def prepare_train_test(perc_test):
    """
        Split corpus in train and test and store them

        :param1 perc_test: test corpus percentage in float.
    """
    corpus_with_bigrams = load_without_not_relevant_hts(TRAIN_TEST_INPUT)
    random.shuffle(corpus_with_bigrams)
    cleaned_corpus_with_bigrams = []
    # keeping tweets with at least one hashtag and one word
    for tweet in corpus_with_bigrams:
        bool_h = False
        bool_w = False
        for w in tweet:
            if w == 'rt':
                break  # each retweet starts with 'rt'
            if '#' in w:
                bool_h = True
            else:
                bool_w = True
        if bool_w and bool_h:
            cleaned_corpus_with_bigrams.append(tweet)
    store(cleaned_corpus_with_bigrams[int(len(cleaned_corpus_with_bigrams) * perc_test):], c.TRAIN_CORPUS)
    store(cleaned_corpus_with_bigrams[:int(len(cleaned_corpus_with_bigrams) * perc_test)], c.TEST_CORPUS)


def prepare_dataset(file):
    corpus_with_bigrams = load_without_not_relevant_hts(TRAIN_TEST_INPUT)
    cleaned_corpus_with_bigrams = []
    # keeping tweets with at least one hashtag and one word
    for tweet in corpus_with_bigrams:
        bool_h = False
        bool_w = False
        for w in tweet:
            if w == 'rt':
                break  # each retweet starts with 'rt'
            if '#' in w:
                bool_h = True
            else:
                bool_w = True
        if bool_w and bool_h:
            cleaned_corpus_with_bigrams.append(tweet)
    store(cleaned_corpus_with_bigrams, file)


def hashtags_list(tweet, model=None):
    ht_list = []
    if model is None:
        for w in tweet:
            word_cleaned = re.sub('[^#\\d\\w_]+', ' ', w).lower().strip()
            for word in word_cleaned.split():
                if word[0] == '#':
                    word_cleaned = word
                    break
            for r in emb.REMOVE_WORDS:
                if r == word_cleaned:
                    word_cleaned = ''
            if len(word_cleaned) > 1 and '#' in word_cleaned:
                ht_list.append(word_cleaned)
    else:
        for w in tweet:
            word_cleaned = re.sub('[^#\\d\\w_]+', ' ', w).lower().strip()
            for word in word_cleaned.split():
                if word[0] == '#':
                    word_cleaned = word
                    break
            for r in emb.REMOVE_WORDS:
                if r == word_cleaned:
                    word_cleaned = ''
            if len(word_cleaned) > 1 and '#' in word_cleaned and word_cleaned in model.wv.vocab:
                ht_list.append(word_cleaned)
    return ht_list


def count_words(tweets):
    counter = dict()
    for line in tweets:
        l = line.split()
        for w in l:
            counter[w] = counter.get(w, 0) + 1
    return counter


def remove_hashtags_from_sentences(tweets, hts, populate_dictionary=True):
    if c.SKIP_HASHTAG_REMOVING:
        result_tweets = []
        for tweet in tweets:
            tweet_string = ""
            for w in tweet:
                if w[0] == '#':
                    tweet_string = tweet_string + " " + w[1:]
                else:
                    tweet_string = tweet_string + " " + w
            result_tweets.append(tweet_string.strip())
        return result_tweets, hts

    if populate_dictionary:
        counter = count_words(tweets)
        dump(counter, open(c.H_REMOVING_DICT, 'wb'))
    else:
        counter = pkload(open(c.H_REMOVING_DICT, 'rb'))
    result_tweets = []
    result_hts = []
    for tweet, ht_list in zip(tweets, hts):
        norm_tweet = []
        tweet = tweet.split()
        for word in tweet:
            if word[0] == '#':
                no_ht_word = word[1:]
                if counter.__contains__(no_ht_word) and counter[no_ht_word] > MINCOUNT:
                    norm_tweet.append(no_ht_word)
            else:
                norm_tweet.append(word)
        if len(norm_tweet) > 0:
            result_tweets.append(" ".join(norm_tweet))
            result_hts.append(ht_list)
    return result_tweets, result_hts


def prepare_model_inputs_and_targets(w_emb):
    """
        Prepare train and test <X,Y> for neural network

        :param1 w_emb: Word2Vec model.
    """
    train = load(c.TRAIN_CORPUS)
    test = load(c.TEST_CORPUS)
    targets_train = []
    sentences_train = []
    targets_test = []
    sentences_test = []
    ht_lists = []
    for tweet in train:
        ht_list = hashtags_list(tweet, w_emb)
        h_embedding = emb.tweet_arith_embedding(w_emb, " ".join(ht_list))
        if h_embedding is not None:
            targets_train.append(np.array(h_embedding))
            sentences_train.append(tweet)
    for tweet in test:
        ht_list = hashtags_list(tweet, w_emb)
        h_embedding = emb.tweet_arith_embedding(w_emb, " ".join(ht_list))
        if h_embedding is not None:
            targets_test.append(h_embedding)
            sentences_test.append(tweet)
            ht_lists.append(ht_list)

    sentences_train_len = len(sentences_train)
    targets_train_len = len(targets_train)
    sentences = sentences_train
    sentences.extend(sentences_test)
    targets = targets_train
    targets.extend(targets_test)
    sentences, targets = remove_hashtags_from_sentences(sentences, targets)
    targets_train = np.array(targets[:targets_train_len])
    targets_test = np.array(targets[targets_train_len:])
    sentences_train = np.array(sentences[:sentences_train_len])
    sentences_test = np.array(sentences[sentences_train_len:])
    return sentences_train, sentences_test, targets_train, targets_test, ht_lists


def prepare_dataset_inputs_and_targets(w_emb, dataset_name):
    """
        Prepare train and test <X,Y> for neural network

        :param dataset_name: dataset name
        :param1 w_emb: Word2Vec model.
    """
    if dataset_name == 'train':
        dataset = load(c.TRAIN_CORPUS)
    else:
        dataset = load(c.TEST_CORPUS)
    targets = []
    sentences = []
    ht_lists = []
    for tweet in dataset:
        ht_list_test = hashtags_list(tweet, None)
        ht_list = hashtags_list(tweet, w_emb)
        h_embedding = emb.tweet_arith_embedding(w_emb, " ".join(ht_list))
        if h_embedding is not None:
            targets.append(np.array(h_embedding))
            sentences.append(tweet)
            ht_lists.append(ht_list_test)

    sentences, targets = remove_hashtags_from_sentences(sentences, targets)
    targets = np.array(targets)
    sentences = np.array(sentences)
    if dataset_name == 'train':
        return sentences, targets
    else:
        return sentences, targets, ht_lists
