import os
from os.path import join
import pandas as pd

GUSE = "GUSE"
BERT = "BERT"
# choose encoder
SENTENCE_ENCODING_MODEL = BERT  # Average pooling
# Word2Vec mincount parameter (suggested value for larger datasets: 5)
W2V_MINCOUNT = 40
# hashtag mincount in corpus (suggested value for large datasets: 10 or more)
MINCOUNT = 60
LATENT_SPACE_DIM = 150
WINDOW_SIZE = 5
# skip hashtag cleaning before sentence embedding parameter
SKIP_HASHTAG_REMOVING = True
# remove retweets
NO_RETWEETS = False
# Google Universal Sentence Encoder path
GUSE_PATH = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
PERC_TEST = 0.25
GLOBAL_EXPANSION = "global"
LOCAL_EXPANSION = "local"
# choose expansion strategy
EXPANSION_STRATEGY = GLOBAL_EXPANSION
MAX_EXPANSION_ITERATIONS = 6
# --------------------------
# neural network
BATCH_SIZE = 32
# log levels
SILENT = 0
PROGRESS = 1
ONE_LINE_PER_EPOCH = 2
LOG_LEVEL = 1
PATIENCE = 3
MAX_EPOCHS = 30
MAX_EPOCHS_FT = 12
DO_STATIC_TRAIN = False  # Choose if you want to train the model on the static dataset or load the weights from static_train_save_files
REALTIME = True  # Set to True if you want to enable real-time trend shift detection and model adaptation
RT_LR = 3e-5
# perc. of input layer neurons in the FC layer
SET_SEED = True  # set random seeds for repeatable experiments
TOPICS = []
TEST_FOLDER = "test"  # Set the output folder
DATASETS_DIR = "datasets"  # Set the dataset folder
DATASET_NAME = "[PUT YOUR DATASET NAME HERE]"
VALIDATION = "validation.pkl"
SAVE_FOLDER = join(TEST_FOLDER, "save_files")
STATIC_TRAIN_SAVE_FOLDER = join(TEST_FOLDER, "static_train_save_files")

# creating folder for save files
if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

# creating folder for save files for static train
if not os.path.exists(STATIC_TRAIN_SAVE_FOLDER):
    os.mkdir(STATIC_TRAIN_SAVE_FOLDER)

RESULTS_FOLDER = join(TEST_FOLDER, "results")

# creating folder for results and logs
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)

# results file
RESULTS = join(RESULTS_FOLDER, "results.txt")
# log file
LOG = join(RESULTS_FOLDER, "log.txt")
# evaluation results file
EVALUATION_RESULTS = join(RESULTS_FOLDER, "evaluation_results.txt")
open(RESULTS, 'w').close()
open(LOG, 'w').close()
open(EVALUATION_RESULTS, 'w').close()
W2V_SAVE_FILE_NAME = join(SAVE_FOLDER, "w2v_emb.model")
# name of the input text file for the model
TRAIN_TEST_INPUT = join(SAVE_FOLDER, 'model_sentences.txt')
H_REMOVING_DICT = join(SAVE_FOLDER, 'h_removing_dict.pkl')
MINMAX_SCALER_FILENAME = join(SAVE_FOLDER, 'minmax_scaler.pkl')
STD_SCALER_FILENAME = join(SAVE_FOLDER, 'std_scaler.pkl')
TRAIN_CORPUS = join(SAVE_FOLDER, "train_corpus.txt")
TEST_CORPUS = join(SAVE_FOLDER, "test_corpus.txt")

if SENTENCE_ENCODING_MODEL == GUSE:
    TL_MODEL_JSON_FILE_NAME = join(SAVE_FOLDER, "TLmodel.json")
    TL_MODEL_WEIGHTS_FILE_NAME = join(SAVE_FOLDER, "TLweights.h5")
    FT_MODEL_JSON_FILE_NAME = join(SAVE_FOLDER, "FTmodel.json")
    FT_MODEL_WEIGHTS_FILE_NAME = join(SAVE_FOLDER, "FTweights.h5")
else:
    TL_MODEL_JSON_FILE_NAME = join(SAVE_FOLDER, "BERT_TLmodel.json")
    TL_MODEL_WEIGHTS_FILE_NAME = join(SAVE_FOLDER, "BERT_TLweights.h5")
    FT_MODEL_JSON_FILE_NAME = join(SAVE_FOLDER, "BERT_FTmodel.json")
    FT_MODEL_WEIGHTS_FILE_NAME = join(SAVE_FOLDER, "BERT_FTweights.h5")

MODEL_JSON_FILE_NAME = FT_MODEL_JSON_FILE_NAME
MODEL_WEIGHTS_FILE_NAME = FT_MODEL_WEIGHTS_FILE_NAME
LOGIC_WINDOW_LENGTH = 14
FT_WINDOW_LENGTH = 4
TUMBLING_WINDOW_LENGTH = 1
K = 10
THRESHOLD = 0.9
JAVA_CONSTANTS = 'java_constants.txt'

with open(JAVA_CONSTANTS, 'w') as f:
    f.write(f'{TUMBLING_WINDOW_LENGTH}\n{K}\n{THRESHOLD}\n')

days = 60 * 60 * 24
LOGIC_WINDOW_LENGTH = LOGIC_WINDOW_LENGTH * days
TUMBLING_WINDOW_LENGTH = TUMBLING_WINDOW_LENGTH * days
FT_WINDOW_LENGTH = FT_WINDOW_LENGTH * days
# Bootstrap phase end date e.g., 2020-08-15 00:00:00
SPLIT_DATE = pd.Timestamp('[PUT YOUR SPLIT_DATE HERE]').timestamp()
