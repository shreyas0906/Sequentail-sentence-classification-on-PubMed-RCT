import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from src.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = DataLoader(batch_size=128)
EMBEDDING_LAYER_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
BERT_PUBMED_LAYER_URL = 'https://tfhub.dev/google/experts/bert/pubmed/2'
OUTPUT_DIM = 128
CHAR_OUTPUT_DIM = 25

tf_hub_embedding_layer = hub.KerasLayer(EMBEDDING_LAYER_URL,
                                        trainable=False,
                                        name="universal_sentence_encoder")

bert_layer = hub.KerasLayer(BERT_PUBMED_LAYER_URL,
                            trainable=False,
                            name='bert_model_layer')

char_vectorizer = tf.keras.layers.TextVectorization(max_tokens=data.NUM_CHAR_TOKENS,
                                                    output_sequence_length=data.output_sequences_char_length,
                                                    standardize="lower_and_strip_punctuation",
                                                    name="char_vectorizer")

char_vectorizer.adapt(data.train_chars)

char_embed = layers.Embedding(input_dim=data.NUM_CHAR_TOKENS,
                              output_dim=CHAR_OUTPUT_DIM,
                              mask_zero=False,
                              name='char_embed')

token_embedding = layers.Embedding(input_dim=data.output_sequences_len,
                                   output_dim=OUTPUT_DIM,
                                   mask_zero=False,
                                   name='token_embed')
NUM_CLASSES = 5


def get_data_for_training(dataset_type):
    if dataset_type == 'tribrid':
        return data.get_tribrid_model_input()
    elif dataset_type == 'token_and_chars':
        return data.get_only_char_and_token_data()


def only_tokens_model():
    inputs = layers.Input(shape=[], dtype=tf.string)
    pretrained_embedding = tf_hub_embedding_layer(inputs)
    x = layers.Dense(128, activation="relu")(pretrained_embedding)
    outputs = layers.Dense(5, activation="softmax")(x)
    token_model = tf.keras.Model(inputs=inputs,
                                 outputs=outputs)

    # Compile the model
    token_model.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])

    return token_model


def char_and_token_model():
    token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
    token_embeddings = bert_layer(token_inputs) #tf_hub_embedding_layer(token_inputs)
    token_dense_1 = layers.Dense(256, activation='relu')(token_embeddings)
    token_dense_2 = layers.Dense(128, activation='relu')(token_dense_1)
    token_model = tf.keras.Model(inputs=token_inputs,
                                 outputs=token_dense_2,
                                 name="token_model")

    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(char_embeddings)
    char_bi_lstm_2 = layers.Bidirectional(layers.LSTM(128))(char_bi_lstm)
    char_dense = layers.Dense(128, activation='relu')(char_bi_lstm_2)
    char_model = tf.keras.Model(inputs=char_inputs,
                                outputs=char_dense,
                                name="char_model")

    concat = layers.Concatenate(name="concatenation_layer")([token_model.output, char_model.output])
    combined_dropout = layers.Dropout(0.5)(concat)
    combined_dense = layers.Dense(128, activation='relu')(combined_dropout)
    output_layer = layers.Dense(NUM_CLASSES, activation='softmax')(combined_dense)

    model = tf.keras.Model(inputs=[token_inputs, char_inputs],
                           outputs=output_layer,
                           name="combined_model")

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def tribrid_model():
    """
    This model is trained on token, char and positional embeddings.
    :return: a compiled model which accepts char_embedding, token_embedding and positional_embedding.
    """

    text_vectorizer = layers.TextVectorization(max_tokens=data.MAX_TOKENS,
                                               output_sequence_length=data.output_sequences_len)

    text_vectorizer.adapt(data.train_sentences)

    token_inputs = layers.Input(shape=(1,), dtype=tf.string, name="token_inputs")
    token_vectors = text_vectorizer(token_inputs)
    token_embeddings = token_embedding(token_vectors)
    token_lstm_1 = layers.LSTM(256, return_sequences=True)(token_embeddings)
    token_lstm_2 = layers.LSTM(128)(token_lstm_1)
    dense_1 = layers.Dense(128, activation='relu')(token_lstm_2)
    dense_2 = layers.Dense(64, activation='relu')(dense_1)
    token_outputs = layers.Dense(64, activation='relu')(dense_2)
    token_model = tf.keras.Model(inputs=token_inputs,
                                 outputs=token_outputs)

    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name='char_inputs')
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm_1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(char_embeddings)
    char_bi_lstm_2 = layers.Bidirectional(layers.LSTM(64))(char_bi_lstm_1)
    char_model = tf.keras.Model(inputs=char_inputs,
                                outputs=char_bi_lstm_2)

    line_number_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_number_input")
    dense_3 = layers.Dense(64, activation='relu')(line_number_inputs)
    dense_4 = layers.Dense(64, activation='relu')(dense_3)

    line_number_model = tf.keras.Model(inputs=line_number_inputs,
                                       outputs=dense_4)

    total_line_inputs = layers.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
    dense_5 = layers.Dense(64, activation="relu")(total_line_inputs)
    dense_6 = layers.Dense(64, activation="relu")(dense_5)

    total_line_model = tf.keras.Model(inputs=total_line_inputs,
                                      outputs=dense_6)

    # 5. Combine token and char embeddings into a hybrid embedding
    combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")([token_model.output,
                                                                                  char_model.output])
    dense_7 = layers.Dense(256, activation="relu")(combined_embeddings)
    dense_8 = layers.Dense(128, activation="relu")(dense_7)
    drop_1 = layers.Dropout(0.2)(dense_8)

    # 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
    concat = layers.Concatenate(name="token_char_positional_embedding")([line_number_model.output,
                                                                         total_line_model.output,
                                                                         drop_1])

    # 7. Create output layer
    output_layer_0 = layers.Dense(128, activation="relu", name="output_layer_0")(concat)
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")(output_layer_0)

    # 8. Put together model
    tribrid_model = tf.keras.Model(inputs=[line_number_model.input,
                                           total_line_model.input,
                                           token_model.input,
                                           char_model.input],
                                   outputs=output_layer)

    tribrid_model.compile(loss="categorical_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["accuracy"])

    return tribrid_model
