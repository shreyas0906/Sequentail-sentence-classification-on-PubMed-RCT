import pandas as pd
import numpy as np
import string
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


def get_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def split_chars(text):
    """
    Converts a list of string to list of chars
    :param text: string input to be converted to list of chars.
    :return: list of chars.
    """
    return " ".join(list(text))


def preprocess_text_with_line_numbers(filename):
    """Returns a list of dictionaries of abstract line data.

    Takes in filename, reads its contents and sorts through each line,
    extracting things like the target label, the text of the sentence,
    how many sentences are in the current abstract and what sentence number
    the target line is.

    Args:
        filename: a string of the target text file to read and extract line data
        from.

    Returns:
        A list of dictionaries each containing a line from an abstract,
        the lines label, the lines position in the abstract and the total number
        of lines in the abstract where the line is from. For example:

        [{"target": 'CONCLUSION',
          "text": The study couldn't have gone better, turns out people are kinder than you think",
          "line_number": 8,
          "total_lines": 8}]
    """
    file = filename.split('/')[-1]
    input_lines = get_lines(filename)  # get all lines from filename
    abstract_lines = ""  # create an empty abstract
    abstract_samples = []  # create an empty list of abstracts

    # Loop through each line in target file
    for line in tqdm(input_lines, desc=f'processing {file}'):
        if line.startswith("###"):  # check to see if line is an ID line
            abstract_id = line
            abstract_lines = ""  # reset abstract string
        elif line.isspace():  # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines()  # split abstract into separate lines

            # Iterate through each line in abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}  # create empty dict to store data from line
                target_text_split = abstract_line.split("\t")  # split target label from text
                line_data["target"] = target_text_split[0]  # get target label
                line_data["text"] = target_text_split[1].lower()  # get target text and lower it
                line_data[
                    "line_number"] = abstract_line_number  # what number line does the line appear in the abstract?
                line_data["total_lines"] = len(
                    abstract_line_split) - 1  # how many total lines are in the abstract? (start from 0)
                abstract_samples.append(line_data)  # add line data to abstract samples list

        else:  # if the above conditions aren't fulfilled, the line contains a labelled sentence
            abstract_lines += line
    return abstract_samples


class DataLoader:

    def __init__(self, batch_size):
        self.DATA_DIR = "../pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
        self.MAX_TOKENS = 68000
        alphabets = string.ascii_lowercase + string.digits + string.punctuation
        self.NUM_CHAR_TOKENS = len(alphabets) + 2  # num of characters in alphabets + space + 'OOV' token
        self.BATCH_SIZE = batch_size
        # self.train_lines = self.get_lines(self.DATA_DIR + 'train.txt')

        self.train_samples = preprocess_text_with_line_numbers(self.DATA_DIR + 'train.txt')
        self.test_samples = preprocess_text_with_line_numbers(self.DATA_DIR + 'test.txt')
        self.val_samples = preprocess_text_with_line_numbers(self.DATA_DIR + 'dev.txt')

        self.train_df = pd.DataFrame(self.train_samples)
        self.val_df = pd.DataFrame(self.val_samples)
        self.test_df = pd.DataFrame(self.test_samples)

        self.train_sentences = self.train_df["text"].tolist()
        self.val_sentences = self.val_df["text"].tolist()
        self.test_sentences = self.test_df["text"].tolist()

        one_hot_encoder = OneHotEncoder(sparse=False)
        self.train_labels_one_hot = one_hot_encoder.fit_transform(self.train_df["target"].to_numpy().reshape(-1, 1))
        self.val_labels_one_hot = one_hot_encoder.transform(self.val_df["target"].to_numpy().reshape(-1, 1))
        self.test_labels_one_hot = one_hot_encoder.transform(self.test_df["target"].to_numpy().reshape(-1, 1))

        self.train_sentences_length = [len(sentence.split()) for sentence in self.train_sentences]
        self.average_train_sentences_length = np.mean(self.train_sentences_length)
        self.output_sequences_len = int(np.percentile(self.train_sentences_length, 95))

        self.train_chars = [split_chars(sentence) for sentence in self.train_sentences]
        self.valid_chars = [split_chars(sentence) for sentence in self.val_sentences]
        self.test_chars = [split_chars(sentence) for sentence in self.test_sentences]

        self.train_chars_length = [len(sentence) for sentence in self.train_sentences]
        self.mean_char_len = np.mean(self.train_chars_length)
        self.output_sequences_char_length = int(np.percentile(self.train_chars_length, 95))

        self.line_number = self.train_df['line_number']
        self.train_line_numbers_one_hot = tf.one_hot(self.line_number.to_numpy(), depth=15)
        self.val_line_numbers_one_hot = tf.one_hot(self.val_df['line_number'].to_numpy(), depth=15)
        self.test_line_numbers_one_hot = tf.one_hot(self.test_df['line_number'].to_numpy(), depth=15)

        self.train_total_lines_one_hot = tf.one_hot(self.train_df["total_lines"].to_numpy(), depth=20)
        self.val_total_lines_one_hot = tf.one_hot(self.val_df["total_lines"].to_numpy(), depth=20)
        self.test_total_lines_one_hot = tf.one_hot(self.test_df["total_lines"].to_numpy(), depth=20)

    def __get_train_char_dataset(self):
        """
        Generate a tf.data.Dataset input train_chars data pipeline
        :return: a train_char pipeline
        """
        train_char_dataset = tf.data.Dataset.from_tensor_slices((self.train_chars, self.train_labels_one_hot))
        train_char_dataset = train_char_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return train_char_dataset

    def __get_val_char_dataset(self):

        val_char_dataset = tf.data.Dataset.from_tensor_slices((self.valid_chars, self.val_labels_one_hot))
        val_char_dataset = val_char_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return val_char_dataset

    def __get_train_dataset(self):
        """
        Generate a tf.data.Dataset input train data pipeline
        :return: a training pipeline
        """
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_sentences, self.train_labels_one_hot))
        train_dataset = train_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return train_dataset

    def __get_validation_dataset(self):
        """
        Generate a tf.data.Dataset input validation data pipeline
        :return: a validation pipeline
        """
        valid_dataset = tf.data.Dataset.from_tensor_slices((self.val_sentences, self.val_labels_one_hot))
        valid_dataset = valid_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return valid_dataset

    def __get_test_dataset(self):
        """
        Generates a tf.data.Dataset input test data pipeline
        :return: a testing pipeline
        """
        test_dataset = tf.data.Dataset.from_tensor_slices((self.test_sentences, self.test_labels_one_hot))

        return test_dataset

    def get_only_tokens_data(self):
        return self.__get_train_dataset(), self.__get_validation_dataset()

    def get_only_char_and_token_data(self):

        train_char_token_data = tf.data.Dataset.from_tensor_slices((self.train_sentences, self.train_chars))
        train_char_token_labels = tf.data.Dataset.from_tensor_slices(self.train_labels_one_hot)
        train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels))

        train_chars_token_dataset = train_char_token_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        valid_char_token_data = tf.data.Dataset.from_tensor_slices((self.val_sentences, self.valid_chars))
        valid_char_token_labels = tf.data.Dataset.from_tensor_slices(self.val_labels_one_hot)
        valid_char_token_dataset = tf.data.Dataset.zip((valid_char_token_data, valid_char_token_labels))

        valid_chars_dataset = valid_char_token_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return train_chars_token_dataset, valid_chars_dataset

    def get_tribrid_model_input(self):
        """
        Make the data pipeline for tribrid model.
        :return: training dataset with positional embeddings, validation dataset with positional embeddings.
        """
        #training dataset.
        train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((self.train_line_numbers_one_hot,  # line numbers
                                                                        self.train_total_lines_one_hot,   # total lines
                                                                        self.train_sentences,             # train tokens
                                                                        self.train_chars))                # train chars
        train_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(self.train_labels_one_hot)       # train labels
        train_pos_char_token_dataset = tf.data.Dataset.zip((train_pos_char_token_data, train_pos_char_token_labels))  # combine data and labels
        train_pos_char_token_dataset = train_pos_char_token_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # turn into batches and prefetch appropriately

        # Validation dataset
        val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((self.val_line_numbers_one_hot,
                                                                      self.val_total_lines_one_hot,
                                                                      self.val_sentences,
                                                                      self.valid_chars))
        val_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(self.val_labels_one_hot)
        val_pos_char_token_dataset = tf.data.Dataset.zip((val_pos_char_token_data, val_pos_char_token_labels))
        val_pos_char_token_dataset = val_pos_char_token_dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # turn into batches and prefetch appropriately

        return train_pos_char_token_dataset, val_pos_char_token_dataset

