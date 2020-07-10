# Created by lan at 2020/6/17
from itertools import chain, product

import nltk as nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def csv_sim(transformed, original):
    """
    calculate the similarity between the transformed csv file and the original (ground truth) csv file.


    :param transformed: the csv file transformed from the input file by applying a set of operators
    :param original: the ground truth csv file that shows the desired result.
    :return: a similarity score between 0.0 (least similar) and 1.0 (the files are the same).
    """

    csv_sim_col, cpair_indices = csv_sim_c(transformed, original)
    csv_sim_row, rpair_indices = csv_sim_r(transformed, original)
    return (csv_sim_col + csv_sim_row) / 2, cpair_indices, rpair_indices


def csv_sim_c(transformed, original):
    t_data = np.array(transformed).astype(str)
    o_data = np.array(original).astype(str)
    assert t_data.ndim == 2 and o_data.ndim == 2

    # create columns from the transformed csv file
    t_cols = []
    t_col_size = t_data.shape[1]
    for i in range(t_col_size):
        column = t_data[:, i]
        unique_words = list(chain(*[nltk.word_tokenize(cell) for cell in column]))
        uw_seq = ' '.join([word for word in unique_words])
        t_cols.append(uw_seq)

    o_cols = []
    o_col_size = o_data.shape[1]
    for i in range(o_col_size):
        column = o_data[:, i]
        unique_words = list(chain(*[nltk.word_tokenize(cell) for cell in column]))
        uw_seq = ' '.join([word for word in unique_words])
        o_cols.append(uw_seq)

    m = t_col_size  # number of columns in the transformed file
    n = o_col_size  # number of columns in the original file

    merged = t_cols + o_cols

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
    vectors = vectorizer.fit_transform(raw_documents=merged).toarray()
    vocabulary = vectorizer.vocabulary_

    t_bow = vectors[:m]
    o_bow = vectors[-n:]

    # Todo: how to define 'least similar'?

    # the matrix that stores the dis-similarity between one column in the transformed csv and one in the original csv
    # row: columns in the transformed csv
    # column: columns in the original csv
    col_sim_matrix = cosine_similarity(t_bow, o_bow)
    col_diss_matrix = 1 - col_sim_matrix

    from scipy.optimize import linear_sum_assignment

    # If m < n, not every column in the original file can find a best match.
    # In this case, 0 (or 1 if we use (1 - similarity)) is set to every column in the original file that cannot find a match.
    row_indices, col_indices = linear_sum_assignment(col_diss_matrix)

    indices_sim_pairs = [(row_index, column_index) for row_index, column_index in zip(row_indices, col_indices)]
    csv_sim = col_sim_matrix[row_indices, col_indices].sum() / (n if n > m else m)

    return csv_sim, indices_sim_pairs


def csv_sim_r(transformed, original):
    t_data = np.array(transformed).astype(str)
    o_data = np.array(original).astype(str)
    assert t_data.ndim == 2 and o_data.ndim == 2

    # create rows from the transformed csv file
    t_rows = []
    t_row_size = t_data.shape[0]
    for i in range(t_row_size):
        row = t_data[i, :]
        unique_words = list(chain(*[nltk.word_tokenize(cell) for cell in row]))
        uw_seq = ' '.join([word for word in unique_words])
        t_rows.append(uw_seq)

    o_rows = []
    o_row_size = o_data.shape[0]
    for i in range(o_row_size):
        row = o_data[i, :]
        unique_words = list(chain(*[nltk.word_tokenize(cell) for cell in row]))
        uw_seq = ' '.join([word for word in unique_words])
        o_rows.append(uw_seq)

    m = t_row_size  # number of columns in the transformed file
    n = o_row_size  # number of columns in the original file

    merged = t_rows + o_rows

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
    vectors = vectorizer.fit_transform(raw_documents=merged).toarray()
    vocabulary = vectorizer.vocabulary_

    t_bow = vectors[:m]
    o_bow = vectors[-n:]

    # Todo: how to define 'least similar'?

    # the matrix that stores the dis-similarity between one column in the transformed csv and one in the original csv
    # row: columns in the transformed csv
    # column: columns in the original csv
    col_sim_matrix = cosine_similarity(t_bow, o_bow)
    col_diss_matrix = 1 - col_sim_matrix

    from scipy.optimize import linear_sum_assignment

    # If m < n, not every column in the original file can find a best match.
    # In this case, 0 (or 1 if we use (1 - similarity)) is set to every column in the original file that cannot find a match.
    row_indices, col_indices = linear_sum_assignment(col_diss_matrix)

    indices_sim_pairs = [(row_index, column_index) for row_index, column_index in zip(row_indices, col_indices)]
    normalization_factor = n if n > m else m
    csv_sim = col_sim_matrix[row_indices, col_indices].sum() / normalization_factor

    return csv_sim, indices_sim_pairs


# Todo: another metric is the the percentage of tuples (rows) in the target file that appear also in the constructed file.


if __name__=='__main__':
    # transformed = [['ab1 to test unique eu', '', '', ''], ['eu', '', '', ''], ['header1', 'header2', 'header3', 'header4'], ['', 31, 32, 2], ['', 32, 10, 2.2],
    #                  ['', 33, 6, 1.2]]
    # original = [['ab to test tokenization', '', 'exp', '', ''], ['EU', 'E', '', '', ''], ['header0', 'header1', 'header2', 'header3', 'header2'],
    #             ['', 21, 32, 2, 31], ['', 32, 10.5, 2, 32], ['', 30, 6, 1.2, 33]]
    # sim_package = csv_similarity(transformed=transformed, original=original)
    # print(sim_package[0])
    #
    # original = [['ab1 to test unique eu', '', '', ''], ['eu', '', '', ''], ['header1', 'header2', 'header3', 'header4'], ['', 31, 32, 2], ['', 32, 10, 2.2],
    #                  ['', 33, 6, 1.2]]
    # sim_package = csv_similarity(transformed=transformed, original=original)
    # print(sim_package[0])
    #
    # original = [[1, 21, 3], [4, 1, 22], [9, 1, 60], [12, 13, 11]]
    # sim_package = csv_similarity(transformed=transformed, original=original)
    # print(sim_package[0])
    pass
