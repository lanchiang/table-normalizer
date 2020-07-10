# Created by lan at 2020/6/15
import gzip
import json
from _csv import reader

import numpy as np
import pandas
from scipy.stats import mode

from tabnorm.experiment.metrics import csv_sim
from tabnorm.common.constants import CELL_METADATA, CELL_EMPTY, CELL_HEADER, CELL_GROUP_HEADER, CELL_DATA, CELL_NOTES, DELETE_ROW_OP, FILL_DOWN_OP, CELL_DERIVED

def loadfile(path):
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        data = list(csv_reader)
    return data


class Rule_based_transformer:
    def __init__(self, csv_data, cell_classes, verbose=False):
        self.csv_data = np.array(csv_data, dtype='object')

        self.cell_classes = np.array(cell_classes)

        assert self.csv_data.ndim == 2 and self.cell_classes.ndim == 2
        assert self.csv_data.shape[0] == self.cell_classes.shape[0]
        assert self.csv_data.shape[1] == self.cell_classes.shape[1]

        self.file_length = self.cell_classes.shape[0]
        self.file_width = self.cell_classes.shape[1]

        self.verbose = verbose

    def transform(self):
        self.delete_empty_rows()
        self.delete_empty_columns()
        self.delete_metadata_rows()
        self.delete_notes_rows()
        self.delete_derived_rows()
        self.delete_derived_columns()

        self.fill_right_header()
        self.fill_down_group()
        self.merge_adjacent_header_rows()

    def delete_empty_rows(self):
        is_empty_by_row = np.all(self.cell_classes == CELL_EMPTY, axis=1)
        indices = np.where(is_empty_by_row)[0]
        self.__delete_rows(indices)

    def delete_empty_columns(self):
        # print('Delete empty columns...')
        is_empty_by_column = np.all(self.cell_classes == CELL_EMPTY, axis=0)
        indices = np.where(is_empty_by_column)[0]
        self.__delete_columns(indices)

    def delete_metadata_rows(self):
        indices = self.metadata_row_indices()
        self.__delete_rows(indices)

    def delete_notes_rows(self):
        indices = self.notes_row_indices()
        self.__delete_rows(indices)

    def delete_derived_rows(self):
        is_derived_by_row = mode(self.cell_classes, axis=1)
        indices = np.where(is_derived_by_row[0] == CELL_DERIVED)[0]  # the indices of rows whose majority cell type is derived.
        self.__delete_rows(indices)

    def delete_derived_columns(self):
        is_derived_by_column = mode(self.cell_classes, axis=0)
        indices = np.where(is_derived_by_column[0] == CELL_DERIVED)[1]
        self.__delete_columns(indices)

    def fill_right_header(self):
        indices = self._header_row_indices()
        df = pandas.DataFrame(data=self.csv_data).replace(r'^\s*$', np.nan, regex=True)
        partial_df = df.iloc[indices]
        partial_df = partial_df.transpose().fillna(method='ffill').transpose()
        df = df.combine_first(partial_df)
        self.csv_data = df.to_numpy()

        df = pandas.DataFrame(data=self.cell_classes).replace('empty', np.nan, regex=False)
        partial_df = df.iloc[indices]
        partial_df = partial_df.transpose().fillna(method='ffill').transpose()
        df = df.combine_first(partial_df).replace(np.nan, 'empty', regex=False)
        self.cell_classes = df.to_numpy()

    # fill the empty cells below a group cell with the value of it, if a column shall be fill down
    def fill_down_group(self):
        has_group_by_column = np.any(self.cell_classes == CELL_GROUP_HEADER, axis=0)
        indices = np.where(has_group_by_column)[0]
        df = pandas.DataFrame(data=self.csv_data).replace(r'^\s*$', np.nan, regex=True)
        partial_df = df.iloc[:, indices]
        partial_df = partial_df.fillna(method='ffill')
        df[indices] = partial_df
        df = df.replace(np.nan, '', regex=False)
        self.csv_data = df.to_numpy()

        df = pandas.DataFrame(data=self.cell_classes).replace('empty', np.nan, regex=False)
        partial_df = df.iloc[:, indices]
        partial_df = partial_df.fillna(method='ffill')
        partial_df = partial_df.replace('group', 'data', regex=False)
        df[indices] = partial_df
        df = df.replace(np.nan, 'empty', regex=False)
        self.cell_classes = df.to_numpy()

    def merge_adjacent_header_rows(self):
        indices = self.adjacent_header_row_indices()
        indices = np.flip(indices)

        self.csv_data = pandas.DataFrame(data=self.csv_data).replace(np.nan, '', regex=False).to_numpy()
        for index in indices:
            concatenated = [' '.join([e1 if e1 is not np.nan else '', e2 if e2 is not np.nan else ''])
                            for e1, e2 in zip(self.csv_data[index], self.csv_data[index + 1])]
            # concatenated = list(map(' '.join, zip(self.csv_data[index], self.csv_data[index+1])))
            concatenated = np.array([s.strip() for s in concatenated])
            self.csv_data[index] = concatenated
            self.cell_classes[index] = self.__overwrite_header_types(target=self.cell_classes[index], source=self.cell_classes[index + 1], axis=1)
            self.__delete_rows(np.array([index + 1]))
        self.csv_data = pandas.DataFrame(data=self.csv_data).replace('', np.nan, regex=False).to_numpy()

    def adjacent_header_row_indices(self):
        indices = self._header_row_indices()
        index_diff = np.diff(indices)
        adjacent_indices = np.where(index_diff == 1)[0]  # the list of indices that are one before the next index in the list
        indices = indices[adjacent_indices]
        return indices

    def __delete_rows(self, row_indices):
        if row_indices.shape[0] != 0:
            self.csv_data = np.delete(self.csv_data, row_indices, axis=0)
            self.cell_classes = np.delete(self.cell_classes, row_indices, axis=0)

    def __delete_columns(self, column_indices):
        if column_indices.shape[0] != 0:
            self.csv_data = np.delete(self.csv_data, column_indices, axis=1)
            self.cell_classes = np.delete(self.cell_classes, column_indices, axis=1)

    def delete_columns(self, column_indices):
        self.csv_data = np.delete(self.csv_data, column_indices, axis=1)
        self.cell_classes = np.delete(self.cell_classes, column_indices, axis=1)

    def notes_row_indices(self):
        is_notes_by_row = np.all((self.cell_classes == CELL_NOTES) | (self.cell_classes == CELL_EMPTY), axis=1)
        is_empty_by_row = np.all(self.cell_classes == CELL_EMPTY, axis=1)
        indices = set(np.where(is_notes_by_row)[0])
        empty_indices = set(np.where(is_empty_by_row)[0])
        indices = np.array(list(indices - empty_indices))
        return indices

    def metadata_row_indices(self):
        is_metadata_by_row = np.all((CELL_METADATA == self.cell_classes) | (self.cell_classes == CELL_EMPTY), axis=1)
        is_empty_by_row = np.all(self.cell_classes == CELL_EMPTY, axis=1)
        indices = set(np.where(is_metadata_by_row)[0])
        empty_indices = set(np.where(is_empty_by_row)[0])
        indices = np.array(list(indices - empty_indices))
        return indices

    def _header_row_indices(self):
        is_header_by_row = np.all((self.cell_classes == CELL_HEADER) | (self.cell_classes == CELL_EMPTY), axis=1)
        is_empty_by_row = np.all(self.cell_classes == CELL_EMPTY, axis=1)
        indices = set(np.where(is_header_by_row)[0])
        empty_indices = set(np.where(is_empty_by_row)[0])
        indices = np.array(list(indices - empty_indices))
        return indices

    def __overwrite_header_types(self, target, source, axis):
        if axis == 0:  # column-wise
            raise NotImplementedError('No implementation for axis = 0 yet.')
        elif axis == 1:  # row-wise
            result = [sc if sc == CELL_HEADER else tc for tc, sc in zip(target, source)]
        else:
            raise RuntimeError('The specified axis is out of scope, Either 0 or 1 is valid.')

        return result


if __name__=='__main__':
    # saus_path = '/Users/lan/PycharmProjects/table-content-type-classification/prepared/saus.jl.gz'
    saus_path = '/Users/lan/PycharmProjects/table-content-type-classification/prepared/cius.jl.gz'
    with gzip.open(saus_path) as input_file:
        tables_dict = np.array([json.loads(line) for index, line in enumerate(input_file)])
        tables = [[table['table_array'], table['annotations'], table['file_name'], table['table_id']] for table in tables_dict
                  # if table['file_name'] == '10s0001.xls' and table['table_id'] == 'Data']
                  # if table['file_name'] == '10s0015.xls' and table['table_id'] == 'Data']
                  # if table['file_name'] == '10s0846.xls' and table['table_id'] == 'Data']
                  # if table['file_name'] == '10s1197.xls' and table['table_id'] == 'Data']
                  if table['file_name'] == '07arresttbl.xls' and table['table_id'] == 'Sheet1']
                  # if table['file_name'] == '07tbl10ar.xls' and table['table_id'] == 'Sheet1']

    # _matrix_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/matrix/10s0001.xls@Data.csv')
    # _relational_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/relational/10s0001.xls@Data.csv')

    # _matrix_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/matrix/10s0015.xls@Data.csv')
    # _relational_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/relational/10s0015.xls@Data.csv')

    # _matrix_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/matrix/10s0846.xls@Data.csv')
    # _relational_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/relational/10s0846.xls@Data.csv')

    # _matrix_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/matrix/10s1197.xls@Data.csv')
    # _relational_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/relational/10s1197.xls@Data.csv')

    _matrix_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/matrix/07arresttbl.xls@Sheet1.csv')
    _relational_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/relational/07arresttbl.xls@Sheet1.csv')

    # _matrix_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/matrix/07tbl10ar.xls@Sheet1.csv')
    # _relational_gs_path = loadfile('/Users/lan/Documents/hpi/manual-30/relational/07tbl10ar.xls@Sheet1.csv')

    data = tables[0][0]
    cell_classes = tables[0][1]

    transformer = Rule_based_transformer(data, cell_classes)
    transformer.transform()

    sim = csv_sim(transformer.csv_data, _matrix_gs_path)[0]
    print(sim)