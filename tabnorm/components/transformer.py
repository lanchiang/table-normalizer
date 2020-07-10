# Created by lan at 2020/6/15
import numpy as np
import pandas
from scipy.stats import mode

from ..common.constants import CELL_METADATA, CELL_EMPTY, CELL_HEADER, CELL_GROUP_HEADER, CELL_DATA, CELL_NOTES, DELETE_ROW_OP, FILL_DOWN_OP, CELL_DERIVED


class Checker:
    def __init__(self, transformer, verbose=False):
        self.transformer = transformer
        self.verbose = verbose

    def predicate_size(self):
        return 7

    def check(self):
        state_vector = []
        state_vector.append(0) if self.__has_empty_rows() else state_vector.append(1)
        state_vector.append(0) if self.__has_empty_columns() else state_vector.append(1)
        state_vector.append(0) if self.__has_adjacent_header_rows() else state_vector.append(1)
        state_vector.append(0) if self.__header_row_incomplete() else state_vector.append(1)
        state_vector.append(0) if self.__has_metadata_rows() else state_vector.append(1)
        state_vector.append(0) if self.__has_notes_rows() else state_vector.append(1)
        state_vector.append(0) if self.__has_group_cells() else state_vector.append(1)
        state_vector.append(0) if self.__has_derived_cells() else state_vector.append(1)
        state_vector.append(0) if not self.__has_only_header_data() else state_vector.append(1)
        return state_vector

    def __has_empty_rows(self):
        is_empty_by_row = np.all(self.transformer.cell_classes == CELL_EMPTY, axis=1)
        indices = np.where(is_empty_by_row)
        return True if len(indices[0]) > 0 else False

    def __has_empty_columns(self):
        is_empty_by_column = np.all(self.transformer.cell_classes == CELL_EMPTY, axis=0)
        indices = np.where(is_empty_by_column)
        return True if len(indices[0]) > 0 else False

    def __has_adjacent_header_rows(self):
        indices = self.transformer.adjacent_header_row_indices()
        return True if len(indices) > 0 else False

    def __header_row_incomplete(self):
        indices = self.transformer._header_row_indices()
        result = np.all(self.transformer.cell_classes[indices] == CELL_HEADER, axis=1)
        return True if not np.all(result) else False

    def __has_metadata_rows(self):
        indices = self.transformer.metadata_row_indices()
        return True if len(indices) > 0 else False

    def __has_notes_rows(self):
        indices = self.transformer.notes_row_indices()
        return True if len(indices) > 0 else False

    def __has_group_cells(self):
        indices = np.where(self.transformer.cell_classes == CELL_GROUP_HEADER)
        return True if len(indices[0]) > 0 else False

    def __has_derived_cells(self):
        indices = np.where(self.transformer.cell_classes == CELL_DERIVED)
        return True if len(indices[0]) > 0 else False

    def __has_only_header_data(self):
        if np.all((self.transformer.cell_classes == CELL_DATA) | (self.transformer.cell_classes == CELL_EMPTY)):
            return True

        result = np.all((self.transformer.cell_classes == CELL_HEADER) | (self.transformer.cell_classes == CELL_DATA) |
                        (self.transformer.cell_classes == CELL_EMPTY))
        if result:
            first_row = self.transformer.cell_classes[0]
            if np.all(first_row == CELL_HEADER):
                return True
            else:
                return False
        return False

        # header_row_indices = np.where(self.transformer.cell_classes == CELL_HEADER)[0]
        # empty_row_indices = np.where(self.transformer.cell_classes == CELL_EMPTY)[0]
        # coexistence_by_row = np.intersect1d(header_row_indices, empty_row_indices)
        # if coexistence_by_row.shape[0] > 0:
        #     return False
        # result = np.all((self.transformer.cell_classes == CELL_HEADER) | (self.transformer.cell_classes == CELL_DATA) | (self.transformer.cell_classes == CELL_EMPTY))
        # return result


class Transformer:
    def __init__(self, csv_data, cell_classes, verbose=False):
        self.csv_data = np.array(csv_data, dtype='object')

        self.cell_classes = np.array(cell_classes)

        assert self.csv_data.ndim == 2 and self.cell_classes.ndim == 2
        assert self.csv_data.shape[0] == self.cell_classes.shape[0]
        assert self.csv_data.shape[1] == self.cell_classes.shape[1]

        self.file_length = self.cell_classes.shape[0]
        self.file_width = self.cell_classes.shape[1]

        self.verbose = verbose

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
        indices = np.where(is_derived_by_row[0] == CELL_DERIVED)[0] # the indices of rows whose majority cell type is derived.
        self.__delete_rows(indices)

    def delete_derived_columns(self):
        is_derived_by_column = mode(self.cell_classes, axis=0)
        indices = np.where(is_derived_by_column[0] == CELL_DERIVED)[1]
        self.__delete_columns(indices)

    # fill the empty cells right to header cells with the value of the used header cells, if a column shall be fill right
    # only fill the header cells
    def fill_right_header(self):
        # obtain indices of header rows
        row_indices = self._header_row_indices()
        # if row_indices.shape[0] > 0:
        #     data_df = pandas.DataFrame(data=self.csv_data).replace(r'^\s*$', np.nan, regex=True)
        #     cell_class_df = pandas.DataFrame(data=self.cell_classes).replace('empty', np.nan, regex=False)
        #     data_target_rows = data_df.iloc[row_indices].transpose()
        #     cell_class_target_rows = cell_class_df.iloc[row_indices].transpose()
        #
        #     filled_data = data_target_rows.fillna(method='ffill')
        #     filled_cell_class = cell_class_target_rows.fillna(method='ffill')
        #     data_values = {}
        #     cell_class_values = {}
        #     # for each column, fill down only if the cell used to fill is a group header cell
        #     for column_name in cell_class_target_rows:
        #         conditions = [(cell_class_target_rows[column_name] == filled_cell_class[column_name]),
        #                       (filled_cell_class[column_name] == 'header')]
        #         choices = [cell_class_target_rows[column_name], filled_cell_class[column_name]]
        #         column = np.select(conditions, choices, default=np.nan)
        #         cell_class_values[column_name] = column
        #
        #         conditions = [(data_target_rows[column_name] == filled_data[column_name]),
        #                       (filled_cell_class[column_name] == 'header')]
        #         choices = [data_target_rows[column_name], filled_data[column_name]]
        #         column = np.select(conditions, choices, default=np.nan)
        #         data_values[column_name] = column
        #
        #     filled_data = pandas.DataFrame(data=data_values).transpose()
        #     data_df = data_df.combine_first(filled_data).replace(np.nan, '', regex=False)
        #     self.csv_data = data_df.to_numpy()
        #
        #     filled_cell_class = pandas.DataFrame(data=cell_class_values).transpose()
        #     cell_class_df = cell_class_df.combine_first(filled_cell_class).replace(np.nan, 'empty', regex=False)
        #     self.cell_classes = cell_class_df.to_numpy()

        df = pandas.DataFrame(data=self.csv_data).replace(r'^\s*$', np.nan, regex=True)
        partial_df = df.iloc[row_indices]
        partial_df = partial_df.transpose().fillna(method='ffill').transpose()
        df = df.combine_first(partial_df)
        self.csv_data = df.to_numpy()

        df = pandas.DataFrame(data=self.cell_classes).replace('empty', np.nan, regex=False)
        partial_df = df.iloc[row_indices]
        partial_df = partial_df.transpose().fillna(method='ffill').transpose()
        df = df.combine_first(partial_df).replace(np.nan, 'empty', regex=False)
        self.cell_classes = df.to_numpy()

    # fill the empty cells below a group cell with the value of it, if a column shall be fill down
    # only fill the group labels
    def fill_down_group(self):
        # obtain indices of columns that contain group header
        column_indices = np.where(np.any(self.cell_classes == CELL_GROUP_HEADER, axis=0))[0]
        if column_indices.shape[0] > 0:
            data_df = pandas.DataFrame(data=self.csv_data).replace(r'^\s*$', np.nan, regex=True)
            cell_class_df = pandas.DataFrame(data=self.cell_classes).replace('empty', np.nan, regex=False)
            data_target_columns = data_df.iloc[:, column_indices]
            cell_class_target_columns = cell_class_df.iloc[:, column_indices]

            filled_data = data_target_columns.fillna(method='ffill')
            filled_cell_class = cell_class_target_columns.fillna(method='ffill')
            data_values = {}
            cell_class_values = {}
            # for each column, fill down only if the cell used to fill is a group header cell
            for column_name in cell_class_target_columns:
                conditions = [(cell_class_target_columns[column_name] == filled_cell_class[column_name]),
                              (filled_cell_class[column_name] == 'group')]
                choices = [cell_class_target_columns[column_name], filled_cell_class[column_name]]
                column = np.select(conditions, choices, default=np.nan)
                cell_class_values[column_name] = column

                conditions = [(data_target_columns[column_name] == filled_data[column_name]),
                              (filled_cell_class[column_name] == 'group')]
                choices = [data_target_columns[column_name], filled_data[column_name]]
                column = np.select(conditions, choices, default=np.nan)
                data_values[column_name] = column

            filled_data = pandas.DataFrame(data=data_values)
            data_df[column_indices] = filled_data
            data_df = data_df.replace(np.nan, '', regex=False)
            self.csv_data = data_df.to_numpy()

            filled_cell_class = pandas.DataFrame(data=cell_class_values).replace('group', 'data', regex=False)
            cell_class_df[column_indices] = filled_cell_class
            cell_class_df = cell_class_df.replace(np.nan, 'empty', regex=False)
            self.cell_classes = cell_class_df.to_numpy()

    def merge_adjacent_header_rows(self):
        indices = self.adjacent_header_row_indices()
        indices = np.flip(indices)

        self.csv_data = pandas.DataFrame(data=self.csv_data).replace(np.nan, '', regex=False).to_numpy()
        for index in indices:
            concatenated = [' '.join([e1 if e1 is not np.nan else '', e2 if e2 is not np.nan else ''])
                            for e1, e2 in zip(self.csv_data[index], self.csv_data[index+1])]
            # concatenated = list(map(' '.join, zip(self.csv_data[index], self.csv_data[index+1])))
            concatenated = np.array([s.strip() for s in concatenated])
            self.csv_data[index] = concatenated
            self.cell_classes[index] = self.__overwrite_header_types(target=self.cell_classes[index], source=self.cell_classes[index+1], axis=1)
            self.__delete_rows(np.array([index+1]))
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
        if axis == 0: # column-wise
            raise NotImplementedError('No implementation for axis = 0 yet.')
        elif axis == 1: # row-wise
            result = [sc if sc == CELL_HEADER else tc for tc,sc in zip(target, source)]
        else: raise RuntimeError('The specified axis is out of scope, Either 0 or 1 is valid.')

        return result


if __name__=='__main__':
    verbose_csv_2 = [['ab1', '', '', '', ''], ['eu', '', '', '', ''], ['', '', '', '', ''], ['', 'header1', '', '', 'header2'], ['header1', 'header2', '', 'header3', 'header4'], ['', 31, '', 32, 2], ['', 32, '', 10, 2.2],
                     ['', 33, '', 6, 1.2], ['', 32, '', 1, 1.2]]
    cell_types_2 = [[CELL_METADATA, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY], [CELL_GROUP_HEADER, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
                    [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY], [CELL_EMPTY, CELL_HEADER, CELL_EMPTY, CELL_EMPTY, CELL_HEADER],
                    [CELL_HEADER, CELL_HEADER, CELL_EMPTY, CELL_HEADER, CELL_HEADER],
                    [CELL_EMPTY, CELL_DATA, CELL_EMPTY, CELL_DATA, CELL_DATA], [CELL_EMPTY, CELL_DATA, CELL_EMPTY, CELL_DATA, CELL_DATA],
                    [CELL_EMPTY, CELL_DATA, CELL_EMPTY, CELL_DATA, CELL_DATA], [CELL_EMPTY, CELL_DERIVED, CELL_EMPTY, CELL_DERIVED, CELL_DERIVED]]
    transformer = Transformer(verbose_csv_2, cell_types_2)
    transformer.merge_adjacent_header_rows()