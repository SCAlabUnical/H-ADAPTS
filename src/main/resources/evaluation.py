import numpy as np
import constants as c
import time

NEUTRAL = -1
AMBIGUOUS = -2
results_file = c.EVALUATION_RESULTS
count = 0


def log_to_file(s):
    with open(results_file, "a") as f:
        f.write(f'{s}\n')


def recall_score(predicted, original):
    count_predicted = 0
    for i in range(0, len(predicted)):
        for j in range(0, len(original)):
            if predicted[i][0] == original[j]:
                count_predicted += 1
    return count_predicted / len(original)


def remove_duplicated(predicted_hashtag_list):
    result = []
    for hts in predicted_hashtag_list:
        result1 = []
        seen = []
        for tuple_hts in hts:
            h = tuple_hts[0]
            if h not in seen:
                seen.append(h)
                result1.append(tuple_hts)
        result.append(result1)
    return result


def global_nhe_evaluation(hashtags_test, sentences_test, model, timestamp, max_iterations=6):
    """
        Evaluates recall, precision and f1 score on each expansion iteration.
        Use global nearest hashtag expansion.

        param1: hashtags_test: original test set hashtag list of list.
        param2: sentences_test: input sentences for making model predictions.
        param3: model: mlp model.
        param4: max_iterations: max number of expansions.
    """
    global count
    count += 1
    predicted_hashtag_list = model.predict_top_k_hashtags(sentences_test, 100)
    predicted_hashtag_list = remove_duplicated(predicted_hashtag_list)
    log_to_file(f"RESULTS OF BATCH {time.ctime(timestamp)}\n")
    for n in range(0, max_iterations):
        log_to_file('RECALL VALUES FOR n: ' + str(n))
        predicted_hashtag_list_bounded = model.global_nhe(hashtags_test, predicted_hashtag_list, n)
        compute_scores(hashtags_test, predicted_hashtag_list_bounded)
        log_to_file("")


def local_nhe_evaluation(hashtags_test, sentences_test, model, timestamp, max_iterations=6):
    """
        Evaluates recall, precision and f1 score on each expansion iteration.
        Use local nearest hashtag expansion.

        param1: hashtags_test: original test set hashtag list of list.
        param2: sentences_test: input sentences for making model predictions.
        param3: model: mlp model.
        param4: max_iterations: max number of expansions.
    """
    global count
    count += 1
    predicted_hashtag_list = model.predict_top_k_hashtags(sentences_test, 50)
    log_to_file(f"RESULTS OF BATCH {time.ctime(timestamp)}\n")
    for n in range(0, max_iterations):
        log_to_file('VALUES FOR n: ' + str(n))
        predicted_hashtag_list_bounded = model.local_nhe(hashtags_test, predicted_hashtag_list, n)
        compute_scores(hashtags_test, predicted_hashtag_list_bounded)
        log_to_file("")


def compute_scores(hashtags_test, predicted_hashtag_list, counter=5):
    numberTweetsAnalyzed = [0, 0, 0, 0, 0, 0]
    recallSumPerTweet = [0, 0, 0, 0, 0, 0]
    precisionSumPerTweet = [0, 0, 0, 0, 0, 0]
    for indexInputTweet in range(0, len(predicted_hashtag_list)):
        real_hashtags = hashtags_test[indexInputTweet]
        k = len(real_hashtags)
        if k > counter:
            numberTweetsAnalyzed[counter] += 1
        else:
            numberTweetsAnalyzed[k - 1] += 1
        predicted_hts = predicted_hashtag_list[indexInputTweet]

        singleRecall = recall_score(predicted_hts, real_hashtags)
        if (singleRecall > 1.0):
            log_to_file("ERROR: " + str(predicted_hts) + ", " + str(real_hashtags))
        singlePrecision = singleRecall * len(real_hashtags) / len(predicted_hts)
        if k > counter:
            recallSumPerTweet[counter] += singleRecall
            precisionSumPerTweet[counter] += singlePrecision
        else:
            recallSumPerTweet[k - 1] += singleRecall
            precisionSumPerTweet[k - 1] += singlePrecision

    log_to_file("# OF TWEETS PER HASHTAG NUMBER")
    log_to_file(numberTweetsAnalyzed)

    recallPerHashtag = []
    precisionPerHashtag = []
    f1scorePerHashtag = []
    eps = 1e-13
    for i in range(0, 6):
        if numberTweetsAnalyzed[i] == 0:
            numberTweetsAnalyzed[i] = eps
        recall_hashtag_number = recallSumPerTweet[i] / numberTweetsAnalyzed[i]
        precision_hashtag_number = precisionSumPerTweet[i] / numberTweetsAnalyzed[i]
        recallPerHashtag.append(round(recall_hashtag_number, 3))
        precisionPerHashtag.append(round(precision_hashtag_number, 3))
        f1den = recall_hashtag_number + precision_hashtag_number
        if f1den == 0:
            f1den = eps
        f1scorePerHashtag.append(round(2 * recall_hashtag_number * precision_hashtag_number / f1den, 3))
    log_to_file("RECALL PER HASHTAG NUMBER")
    log_to_file(recallPerHashtag)
    log_to_file("PRECISION PER HASHTAG NUMBER")
    log_to_file(precisionPerHashtag)
    log_to_file("F1 SCORE PER HASHTAG NUMBER")
    log_to_file(f1scorePerHashtag)

    totalTweets = 0
    numberTweetsAnalyzed1 = numberTweetsAnalyzed[:len(numberTweetsAnalyzed) - 1]
    recallPerHashtag = recallPerHashtag[:len(recallPerHashtag) - 1]
    precisionPerHashtag = precisionPerHashtag[:len(precisionPerHashtag) - 1]
    f1scorePerHashtag = f1scorePerHashtag[:len(f1scorePerHashtag) - 1]

    for i in numberTweetsAnalyzed1:
        totalTweets += i
    weightedRecall = np.sum([(r * i) / totalTweets for r, i in zip(recallPerHashtag, numberTweetsAnalyzed1)])
    weightedPrecision = np.sum([(p * i) / totalTweets for p, i in zip(precisionPerHashtag, numberTweetsAnalyzed1)])
    weightedF1score = np.sum([(f * i) / totalTweets for f, i in zip(f1scorePerHashtag, numberTweetsAnalyzed1)])

    log_to_file("AVERAGE RECALL")
    log_to_file(np.round(weightedRecall, 3))
    log_to_file("AVERAGE PRECISION")
    log_to_file(np.round(weightedPrecision, 3))
    log_to_file("AVERAGE F1 SCORE")
    log_to_file(np.round(weightedF1score, 3))
    log_to_file("")
