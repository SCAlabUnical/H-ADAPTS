import constants as c
import tensorflow as tf

if c.SET_SEED:
    from numpy.random import seed

    seed(42)
    tf.random.set_seed(43)

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
from transformers import DistilBertTokenizer
import tensorflow.python.keras.layers as layers
from transformers import TFDistilBertModel
from tensorflow.python.keras.layers.core import Dense
import tensorflow.python.keras as keras
import word_embedding_model
import numpy as np

SCALE_1 = 2 / 3


def cosine_distance(y_true, y_pred):
    return tf.compat.v1.losses.cosine_distance(tf.nn.l2_normalize(y_pred, 0), tf.nn.l2_normalize(y_true, 0), dim=0)


def static_training(sentences_train, sentences_test, targets_train, targets_test):
    tf.keras.backend.clear_session()
    print("BERT version")
    # transfer learning
    print("TRANSFER LEARNING STEP:")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    sequence_len = 100
    train_encodings = tokenizer(sentences_train.tolist(), truncation=True, padding='max_length',
                                max_length=sequence_len)
    test_encodings = tokenizer(sentences_test.tolist(), truncation=True, padding='max_length', max_length=sequence_len)

    input_ids = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='attention_mask')

    input_BERT = [input_ids, attention_mask]
    encoderlayer = TFDistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)

    encoderoutputs = encoderlayer(input_BERT)

    encoderoutput = encoderoutputs.last_hidden_state
    avg = layers.GlobalAveragePooling1D()(encoderoutput)
    n_hidden = avg.shape[1]

    h1 = Dense(int(n_hidden), activation="relu")(avg)
    h2 = Dense(int(n_hidden * SCALE_1), activation="relu")(h1)
    h3 = Dense(int(n_hidden * SCALE_1 * SCALE_1), activation="relu")(h2)

    out = Dense(len(targets_train[0]), activation='linear')(h3)

    model = keras.Model(inputs=input_BERT, outputs=out)
    model.summary()

    model.trainable = True
    for l in encoderlayer.layers[:]:
        l.trainable = False

    opt = tfa.optimizers.RectifiedAdam()

    # compile model
    model.compile(loss=cosine_distance, optimizer=opt, metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.TL_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True, save_weights_only=True)

    model.fit([np.array(train_encodings["input_ids"]), np.array(train_encodings["attention_mask"])],
              targets_train, validation_data=(
            [np.array(test_encodings["input_ids"]), np.array(test_encodings["attention_mask"])],
            targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])

    # fine tuning
    print("FINE TUNING STEP:")

    # Load weights into the new model
    model.load_weights(c.TL_MODEL_WEIGHTS_FILE_NAME)

    model.trainable = True
    opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)

    model.compile(loss=cosine_distance, optimizer=opt, metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.FT_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True, save_weights_only=True)

    model.fit([np.array(train_encodings["input_ids"]), np.array(train_encodings["attention_mask"])],
              targets_train, validation_data=(
            [np.array(test_encodings["input_ids"]), np.array(test_encodings["attention_mask"])],
            targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS_FT, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])


def rt_training(sentences_train, sentences_test, targets_train, targets_test, ft_sentences_train, ft_sentences_test,
                ft_targets_train, ft_targets_test):
    tf.keras.backend.clear_session()

    print("BERT version")
    # transfer learning
    print("TRANSFER LEARNING STEP:")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    sequence_len = 100
    train_encodings = tokenizer(sentences_train.tolist(), truncation=True, padding='max_length',
                                max_length=sequence_len)
    test_encodings = tokenizer(sentences_test.tolist(), truncation=True, padding='max_length', max_length=sequence_len)

    ft_train_encodings = tokenizer(ft_sentences_train.tolist(), truncation=True, padding='max_length',
                                   max_length=sequence_len)
    ft_test_encodings = tokenizer(ft_sentences_test.tolist(), truncation=True, padding='max_length',
                                  max_length=sequence_len)

    input_ids = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='attention_mask')

    input_BERT = [input_ids, attention_mask]
    encoderlayer = TFDistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)

    encoderoutputs = encoderlayer(input_BERT)

    encoderoutput = encoderoutputs.last_hidden_state
    avg = layers.GlobalAveragePooling1D()(encoderoutput)
    n_hidden = avg.shape[1]

    h1 = Dense(int(n_hidden), activation="relu", name="h1")(avg)
    h2 = Dense(int(n_hidden * SCALE_1), activation="relu", name="h2")(h1)
    h3 = Dense(int(n_hidden * SCALE_1 * SCALE_1), activation="relu", name="h3")(h2)

    out = Dense(len(targets_train[0]), activation='linear', name="h4")(h3)

    model = keras.Model(inputs=input_BERT, outputs=out)
    model.summary()

    MLP_weights = [[]]
    # save MLP weights
    for i in range(1, 5):
        MLP_weights.append(model.get_layer(f"h{i}").get_weights())

    # Load weights into the new model
    model.load_weights(c.FT_MODEL_WEIGHTS_FILE_NAME)

    for i in range(1, 5):
        model.get_layer(f"h{i}").set_weights(MLP_weights[i])

    model.trainable = True
    for l in encoderlayer.layers[:]:
        l.trainable = False

    opt = tfa.optimizers.RectifiedAdam()

    model.compile(loss=cosine_distance, optimizer=opt, metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.TL_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True, save_weights_only=True)

    model.fit([np.array(train_encodings["input_ids"]), np.array(train_encodings["attention_mask"])],
              targets_train, validation_data=(
            [np.array(test_encodings["input_ids"]), np.array(test_encodings["attention_mask"])],
            targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])

    # fine tuning
    print("FINE TUNING STEP:")

    # Load weights into the new model
    model.load_weights(c.TL_MODEL_WEIGHTS_FILE_NAME)

    model.trainable = True
    opt = tfa.optimizers.RectifiedAdam(learning_rate=c.RT_LR)

    model.compile(loss=cosine_distance, optimizer=opt, metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.FT_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True, save_weights_only=True)

    model.fit([np.array(ft_train_encodings["input_ids"]), np.array(ft_train_encodings["attention_mask"])],
              ft_targets_train, validation_data=(
            [np.array(ft_test_encodings["input_ids"]), np.array(ft_test_encodings["attention_mask"])],
            ft_targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS_FT, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])


def predict_top_k_hashtags(sentences, k):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 sentences: sentences.
        :param2 k: number of hashtags to predict for each sentence.
        :returns results: list of list of (hashtag, likelihood):
    """
    tf.keras.backend.clear_session()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    sequence_len = 100

    sentences_encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',max_length=sequence_len)
    # Model reconstruction
    input_ids = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='attention_mask')

    input_BERT = [input_ids, attention_mask]
    encoderlayer = TFDistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
    for l in encoderlayer.layers[:]:
        l.trainable = False
    encoderoutputs = encoderlayer(input_BERT)

    encoderoutput = encoderoutputs.last_hidden_state
    avg = layers.GlobalAveragePooling1D()(encoderoutput)
    n_hidden = avg.shape[1]

    h1 = Dense(int(n_hidden), activation="relu")(avg)
    h2 = Dense(int(n_hidden * SCALE_1), activation="relu")(h1)
    h3 = Dense(int(n_hidden * SCALE_1 * SCALE_1), activation="relu")(h2)

    out = Dense(c.LATENT_SPACE_DIM, activation='linear')(h3)

    model = keras.Model(inputs=input_BERT, outputs=out)
    model.summary()

    opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)

    model.compile(loss=cosine_distance, optimizer=opt, metrics=['cosine_proximity'])  # Load weights into the new model
    model.load_weights(c.MODEL_WEIGHTS_FILE_NAME)

    h_list = model.predict([np.array(sentences_encodings["input_ids"]), np.array(sentences_encodings["attention_mask"])])

    h_list = [np.reshape(h_vect, (len(h_vect),)) for h_vect in h_list]

    emb_model = word_embedding_model.load_Word2Vec_model()
    top_n_words = 1000
    result = [word_embedding_model.retain_hts(emb_model.wv.similar_by_vector(h_vect, topn=top_n_words))[:k] for h_vect in h_list]

    return result
