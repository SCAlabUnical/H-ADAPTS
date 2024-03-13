import json
import os
import pickle as pkl
import shutil
import socket
import sys
import time
from os.path import join
import pandas as pd
import constants as c
import dataset_handler as dh
import evaluation as ev
import model
import word_embedding_model as emb

sys.stdout = open(join(c.TEST_FOLDER, "log.txt"), mode='w', buffering=1)


class ModelRunner:
    no_retweet = c.NO_RETWEETS

    def __init__(self):

        self.logic_window_length = c.LOGIC_WINDOW_LENGTH
        self.tumbling_window_length = c.TUMBLING_WINDOW_LENGTH
        self.ft_window_length = c.FT_WINDOW_LENGTH

        dataset_path = os.path.join(c.DATASETS_DIR, c.DATASET_NAME)

        # reading the dataset
        df = pd.read_json(dataset_path)

        if ModelRunner.no_retweet:
            df = df[df["is_retweet"] == False]

        # sort by date
        df = df.sort_values(by="date")[["date", "text"]]

        # split text for corpus
        df.loc[:, "text"] = df.loc[:, "text"].apply(lambda x: x.split())

        # datetime -> unix timestamp
        df.loc[:, "date"] = df.loc[:, "date"].apply(lambda x: x.timestamp())

        # get hashtags from text
        df['hashtags'] = df['text'].map(dh.hashtags_list)

        # split static dataset
        self.static_training_set = df[df["date"] < c.SPLIT_DATE]

        # save start_trending
        start_trending = {}
        for lista in self.static_training_set.loc[:, "hashtags"]:
            for hashtag in lista:
                if hashtag in start_trending:
                    start_trending[hashtag] += 1
                else:
                    start_trending[hashtag] = 1

        with open(c.JAVA_CONSTANTS, 'a') as f:
            f.write(" ".join(
                [key for key, v in sorted(start_trending.items(), key=lambda item: item[1], reverse=True)[:c.K]]))

        self.rt_dataset = df[df["date"] >= c.SPLIT_DATE]
        self.rt_dataset[["date", "hashtags"]].to_csv(os.path.join(c.DATASETS_DIR, "rt_dataset.csv"), index=False)

        # get first window for testing
        self.first_window = self.rt_dataset[
            self.rt_dataset["date"] < self.rt_dataset.date.iloc[0] + self.tumbling_window_length]

        # dataset for real-time training/testing
        if c.REALTIME:
            # dynamic handling through sliding windows
            self.rt_dataset = df
        else:
            self.rt_dataset = self.rt_dataset[
                self.rt_dataset["date"] >= self.rt_dataset.date.iloc[0] + self.tumbling_window_length]

    def static_training(self):

        # preprocessing data for Word2Vec model
        sentences = dh.preprocess_data(self.static_training_set)

        # Training Word2Vec model
        emb.create_and_train_Word2Vec_model(sentences, mincount=c.W2V_MINCOUNT)

        # Loading Word2Vec model for calculating targets
        w_emb = emb.load_Word2Vec_model()

        # Preprocessing data for sentence embeddings
        dh.preprocess_data_for_sentence_embedding(self.static_training_set)

        dh.prepare_train_test(c.PERC_TEST)

        # Preparing data for neural network training
        sentences_train, sentences_test, targets_train, targets_test, ht_lists = dh.prepare_model_inputs_and_targets(
            w_emb)

        with open(os.path.join(c.DATASETS_DIR, c.VALIDATION), 'wb') as file:
            pkl.dump({"sentences_test": sentences_test, "targets_test": targets_test, "ht_lists": ht_lists}, file)

        self.log_to_file("STATIC TRAINING STARTED")

        # Training MLP model
        model.static_training(sentences_train, sentences_test, targets_train, targets_test)
        self.log_to_file("STATIC TRAINING DONE")

    def test_on_validation(self):
        # Loading Word2Vec model for calculating targets
        w_emb = emb.load_Word2Vec_model()

        self.log_to_file("TEST ON VALIDATION STARTED")

        with open(os.path.join(c.DATASETS_DIR, c.VALIDATION), 'rb') as file:
            d = pkl.load(file)
            sentences_test = d["sentences_test"]
            targets_test = d["targets_test"]
            ht_lists = d["ht_lists"]

        # Loading MLP model and making predictions
        model.predict_hashtags_and_store_results(c.TEST_CORPUS, sentences_test, c.SPLIT_DATE)

        # Evaluate model
        if c.EXPANSION_STRATEGY == c.LOCAL_EXPANSION:
            # Local expansion evaluation
            ev.local_nhe_evaluation(ht_lists, sentences_test, model, c.SPLIT_DATE, c.MAX_EXPANSION_ITERATIONS)
        else:
            # Global expansion evaluation
            ev.global_nhe_evaluation(ht_lists, sentences_test, model, c.SPLIT_DATE, c.MAX_EXPANSION_ITERATIONS)

        self.log_to_file("TEST ON VALIDATION DONE")

        # Prepare test set
        dh.preprocess_data_for_sentence_embedding(self.first_window)
        dh.prepare_dataset(c.TEST_CORPUS)

        sentences_test, targets_test, ht_lists = dh.prepare_dataset_inputs_and_targets(w_emb, "test")
        # Loading MLP model and making predictions
        model.predict_hashtags_and_store_results(c.TEST_CORPUS, sentences_test,
                                                 self.first_window.date.iloc[0] + self.tumbling_window_length)

        # Evaluate model
        if c.EXPANSION_STRATEGY == c.LOCAL_EXPANSION:
            # Local expansion evaluation
            ev.local_nhe_evaluation(ht_lists, sentences_test, model,
                                    self.first_window.date.iloc[0] + self.tumbling_window_length,
                                    c.MAX_EXPANSION_ITERATIONS)
        else:
            # Global expansion evaluation
            ev.global_nhe_evaluation(ht_lists, sentences_test, model,
                                     self.first_window.date.iloc[0] + self.tumbling_window_length,
                                     c.MAX_EXPANSION_ITERATIONS)

    def log_to_file(self_, str):
        with open(c.LOG, 'a') as file:
            file.write(f"[{time.ctime()}] {str}\n")

    # return true if tweet_hts contains one or more hashtags from hts_list
    def filter_hashtags(self, tweet_hts, hts_list):
        for t in tweet_hts:
            if t not in hts_list:
                return True
        return False

    def process(self, tup):
        end_timestamp = tup[0]
        do_train = tup[1]

        # query (training set)
        mask = (end_timestamp - self.logic_window_length <= self.rt_dataset["date"]) & (
                self.rt_dataset["date"] < end_timestamp)
        train = self.rt_dataset[mask]

        # query (ft_set)
        mask = (end_timestamp - self.ft_window_length <= self.rt_dataset["date"]) & (
                self.rt_dataset["date"] < end_timestamp)
        ft = self.rt_dataset[mask]

        # query (test set)
        mask = (end_timestamp <= self.rt_dataset["date"]) & (
                self.rt_dataset["date"] < end_timestamp + self.tumbling_window_length)
        test = self.rt_dataset[mask]

        #           #
        #   TRAIN   #
        #           #

        do_train = not train.empty and not ft.empty and do_train
        do_test = not test.empty

        if do_train:

            # Preprocessing data for Word2Vec model
            sentences = dh.preprocess_data(train)

            # Training Word2Vec model
            emb.create_and_train_Word2Vec_model(sentences, mincount=c.W2V_MINCOUNT)

            # Loading Word2Vec model for calculating targets
            w_emb = emb.load_Word2Vec_model()

            # Prepare training set
            dh.preprocess_data_for_sentence_embedding(train)
            dh.prepare_train_test(c.PERC_TEST)

            # Preparing data for neural network training
            sentences_train, sentences_test, targets_train, targets_test, _ = dh.prepare_model_inputs_and_targets(w_emb)

            # Prepare ft set
            dh.preprocess_data_for_sentence_embedding(ft)
            dh.prepare_train_test(c.PERC_TEST)

            # Preparing data for ft training
            ft_sentences_train, ft_sentences_test, ft_targets_train, ft_targets_test, _ = dh.prepare_model_inputs_and_targets(
                w_emb)

            if len(sentences_train) != 0 and len(ft_sentences_train) != 0:
                self.log_to_file("training")
                # Fine tuning
                model.rt_training(sentences_train, sentences_test, targets_train, targets_test, ft_sentences_train,
                                  ft_sentences_test, ft_targets_train, ft_targets_test)
        else:
            # Loading Word2Vec model for calculating targets
            w_emb = emb.load_Word2Vec_model()

        if do_test:
            # Prepare test set
            dh.preprocess_data_for_sentence_embedding(test)
            dh.prepare_dataset(c.TEST_CORPUS)

            # Preparing data for testing
            sentences_test, targets_test, ht_lists = dh.prepare_dataset_inputs_and_targets(w_emb, "test")
            # Loading MLP model and making predictions
            if len(sentences_test) != 0:
                model.predict_hashtags_and_store_results(c.TEST_CORPUS, sentences_test,
                                                         end_timestamp + self.tumbling_window_length)

                # Evaluate model
                if c.EXPANSION_STRATEGY == c.LOCAL_EXPANSION:
                    # Local expansion evaluation
                    ev.local_nhe_evaluation(ht_lists, sentences_test, model,
                                            end_timestamp + self.tumbling_window_length, c.MAX_EXPANSION_ITERATIONS)
                else:
                    # Global expansion evaluation
                    ev.global_nhe_evaluation(ht_lists, sentences_test, model,
                                             end_timestamp + self.tumbling_window_length, c.MAX_EXPANSION_ITERATIONS)

    def static_testing(self):
        dates = self.rt_dataset["date"]
        test_list = [
            (d + self.tumbling_window_length, self.rt_dataset[(d <= dates) & (dates < d + self.tumbling_window_length)])
            for d in range(int(dates.iloc[0]), int(dates.iloc[-1]), self.tumbling_window_length)]
        for d, test in test_list:
            model_runner.log_to_file(f'Processing tuple with timestamp: {time.ctime(d)}')
            # Prepare test set
            dh.preprocess_data_for_sentence_embedding(test)
            dh.prepare_dataset(c.TEST_CORPUS)

            # Loading Word2Vec model for calculating targets
            w_emb = emb.load_Word2Vec_model()

            # Preparing data for testing
            sentences_test, targets_test, ht_lists = dh.prepare_dataset_inputs_and_targets(w_emb, "test")
            # Loading MLP model and making predictions
            if len(sentences_test) != 0:
                model.predict_hashtags_and_store_results(c.TEST_CORPUS, sentences_test, d)

                # Evaluate model
                if c.EXPANSION_STRATEGY == c.LOCAL_EXPANSION:
                    # Local expansion evaluation
                    ev.local_nhe_evaluation(ht_lists, sentences_test, model, d, c.MAX_EXPANSION_ITERATIONS)
                else:
                    # Global expansion evaluation
                    ev.global_nhe_evaluation(ht_lists, sentences_test, model, d, c.MAX_EXPANSION_ITERATIONS)
            model_runner.log_to_file('Tuple Processed')


if c.REALTIME:
    print("Trend shift detection enabled")
else:
    print("Static model")

model_runner = ModelRunner()

# STATIC TRAINING
if c.DO_STATIC_TRAIN:
    model_runner.static_training()
    for file in os.listdir(c.SAVE_FOLDER):
        shutil.copy(os.path.join(c.SAVE_FOLDER, file), c.STATIC_TRAIN_SAVE_FOLDER)
else:
    for file in os.listdir(c.STATIC_TRAIN_SAVE_FOLDER):
        shutil.copy(os.path.join(c.STATIC_TRAIN_SAVE_FOLDER, file), c.SAVE_FOLDER)

if c.REALTIME:
    model_runner.log_to_file("OPENING SOCKET")
    print("start storm topology")
    server_socket = socket.socket()
    server_socket.bind(("localhost", 8082))
    server_socket.listen()
    conn, _ = server_socket.accept()
    model_runner.log_to_file("CONNECTION STARTED")
    conn_file = conn.makefile()

    try:
        while True:
            storm_tuple = json.loads(conn_file.readline())
            model_runner.log_to_file(f'Processing tuple with timestamp: {time.ctime(storm_tuple[0])}')
            model_runner.process(storm_tuple)
            model_runner.log_to_file('Tuple Processed')
    except:
        print("EXECUTION ENDED")
else:
    model_runner.static_testing()
    print("EXECUTION ENDED")
