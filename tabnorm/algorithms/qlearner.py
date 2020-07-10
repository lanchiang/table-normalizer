# Created by lan at 2020/6/30
import gzip
import itertools
import json
import warnings
from csv import reader, writer

import numpy as np

from tabnorm.components.transformer import Transformer, Checker
from tabnorm.data.loader import load_configuration_files
from tabnorm.experiment.metrics import csv_sim


def loadfile(path):
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        data = list(csv_reader)
    return data


def update_q(q, r, state, next_state, action, done, beta, gamma):
    """
    Update the q value in the quality matrix
    :param q: quality matrix
    :param r: reward matrix
    :param state: the current state
    :param next_state: the next state
    :param action: the operator applied to reach the next state
    :param beta: learning rate
    :param gamma: discount factor
    :return: the reward of applying the action at the current state
    """
    rsa = r[state, action]
    if done == 'Succeed':
        rsa = 1
    if done == 'Failed':
        rsa = -1
    qsa = q[state, action]

    new_q = qsa + beta * (rsa + gamma * max(q[next_state, :]) - qsa)

    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    # q[state] = (q[state] - q[state].min()) / (q[state].max() - q[state].min())
    # rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
    # q[state][q[state] > 0] = rn

    # return r[state, action]
    pass


class Qlearner():

    def __init__(self, dataset, cell_classes, ground_truth, goal, target_goal, target_prepare, verbose=False, file_name=None, threshold=None) -> None:
        self.dataset = dataset
        self.cell_classes = cell_classes
        self.goal = goal
        self.target_goal = target_goal
        self.target_prepare = target_prepare
        self.ground_truth = ground_truth
        self.verbose = verbose
        self.file_name = file_name
        self.threshold = threshold  # TODO: handle an array of thresholds

        self.r_matrix, self.q_matrix, self.n_states, self.n_actions, self.actions_name = self.__initialization()

    @property
    def __get_params(self):
        param_list = ['beta', 'gamma', 'epsilon', 'n_episodes']
        return param_list

    def set_params(self, **kwargs):
        """
        Set the values of the parameters used to run the learner.
        :param kwargs: key-worded parameters
        """
        for k, v in kwargs.items():
            if k not in self.__get_params:
                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`qlearner.get_params().keys()`")
            else:
                setattr(self, k, v)

    def __has_params(self):
        for param in self.__get_params:
            if not hasattr(self, param):
                return False
        return True

    def __initialization(self):
        """
        Initialize reward matrix and quality matrix
        :return: the reward matrix
        """
        r_matrix, actions_name = load_configuration_files()
        q_matrix = np.zeros_like(r_matrix)

        r_matrix = r_matrix[~np.all(r_matrix == -1, axis=1)]

        n_states = q_matrix.shape[0]
        # n_actions = q_matrix.shape[0]
        n_actions = actions_name.shape[0]

        return r_matrix, q_matrix, n_states, n_actions, actions_name

    def __construct_state_set(self):
        has_empty_rows = [0, 1]
        has_empty_columns = [0, 1]
        has_adjacent_header_rows = [0, 1]
        has_metadata = [0, 1]
        has_notes = [0, 1]
        has_group_header = [0, 1]
        has_derived = [0, 1]
        last_action = list(range(self.n_actions - 1))

        states = list(itertools.product(has_empty_rows, has_empty_columns, has_adjacent_header_rows, has_metadata, has_notes, has_group_header,
                                        has_derived, last_action))
        return np.array(states)

    def learning(self):
        terminate_criteria = [1] * 9

        r_matrix = np.full((self.n_actions, self.n_actions), 0.0)
        q_matrix = np.zeros_like(r_matrix)

        randomizer = np.random.RandomState()

        for episode in range(int(self.__getattribute__('n_episodes'))):
            # start a new episode
            transformer = Transformer(self.dataset, self.cell_classes)
            checker = Checker(transformer)
            switcher = {0: transformer.delete_empty_columns,
                        1: transformer.delete_empty_rows,
                        2: transformer.delete_metadata_rows,
                        3: transformer.delete_notes_rows,
                        4: transformer.delete_derived_rows,
                        5: transformer.fill_down_group,
                        6: transformer.fill_right_header,
                        7: transformer.delete_derived_columns,
                        8: transformer.merge_adjacent_header_rows}
            state_vector = checker.check()

            action_candidates = list(range(self.n_actions))
            randomizer.shuffle(action_candidates)
            # state_vector.append(action_candidates[0])
            current_state = action_candidates[0]
            last_state = None
            result = 'Unknown'

            max_steps = 20
            step = 0
            while state_vector != terminate_criteria:
                if step == max_steps:
                    result = 'Failed'
                    update_q(q_matrix, r_matrix, last_state, current_state, current_state, result,
                             self.__getattribute__('beta'), self.__getattribute__('gamma'))
                    break

                valid_actions = r_matrix[current_state] >= -1
                # exploration
                if randomizer.rand() < self.__getattribute__('epsilon'):
                    actions = np.array(list(range(self.n_actions)))
                    actions = actions[valid_actions]

                    if type(actions) is int:
                        actions = [actions]
                    randomizer.shuffle(actions)
                    action = actions[0]
                # exploitation
                else:
                    # if any q value of the current state is greater than zero, choose the biggest one
                    # if np.sum(q_matrix[current_state]) > 0:
                    if np.any(q_matrix[current_state] > 0):
                        action = np.argmax(q_matrix[current_state])
                    else:
                        actions = np.array(list(range(self.n_actions)))
                        actions = actions[valid_actions]
                        randomizer.shuffle(actions)
                        action = actions[0]

                # execute the transform, and observe the result
                # print('Execute action: ' + str(action))
                switcher[action]()

                state_vector = checker.check()

                next_state = action
                if state_vector == terminate_criteria:
                    result = 'Succeed'
                update_q(q_matrix, r_matrix, current_state, next_state, action, result,
                         self.__getattribute__('beta'), self.__getattribute__('gamma'))

                last_state = current_state
                current_state = next_state
                step += 1
            print(result)

        # use q matrix to construct the best sequence
        constructions = []
        for i in range(self.n_actions):
            transformer = Transformer(self.dataset, self.cell_classes)
            checker = Checker(transformer)
            switcher = {0: transformer.delete_empty_columns,
                        1: transformer.delete_empty_rows,
                        2: transformer.delete_metadata_rows,
                        3: transformer.delete_notes_rows,
                        4: transformer.delete_derived_rows,
                        5: transformer.fill_down_group,
                        6: transformer.fill_right_header,
                        7: transformer.delete_derived_columns,
                        8: transformer.merge_adjacent_header_rows}

            pipeline = []

            state_vector = checker.check()
            current_state = i
            max_steps = 20
            step = 0
            action = i
            while state_vector != terminate_criteria and step < max_steps:
                pipeline.append(action)
                action = np.argmax(q_matrix[current_state])
                switcher[action]()
                state_vector = checker.check()
                current_state = action
                step += 1
            pipeline.append(action)
            print(pipeline)
            csv_similarity = csv_sim(transformer.csv_data, self.ground_truth)[0]
            print(csv_similarity)
            print()
            constructions.append((pipeline, csv_similarity, transformer.csv_data, transformer.cell_classes))
        best_index = np.argmax([construction[1] for construction in constructions])
        best_construction = constructions[best_index]

        return best_construction

    # def learning(self):
    #     # construct state set.
    #     states = self.__construct_state_set()
    #     self.n_states = states.shape[0]
    #     self.n_actions = 9
    #
    #     r_matrix = np.full((states.shape[0], self.n_actions), -1)
    #
    #     terminate_states = np.array([i for i, state in enumerate(states) if np.all(state[:-1] == 1)])
    #     r_matrix[terminate_states] = 100
    #
    #     q_matrix = np.zeros_like(r_matrix)
    #     # q_matrix = q_matrix[~np.all(r_matrix == 100, axis=1)]
    #     # r_matrix = r_matrix[~np.all(r_matrix == 100, axis=1)]
    #
    #     randomizer = np.random.RandomState(1)
    #
    #     for episode in range(int(self.__getattribute__('n_episodes'))):
    #         transformer = Transformer(self.dataset, self.cell_classes)
    #         checker = Checker(transformer)
    #         switcher = {0: transformer.delete_empty_columns,
    #                     1: transformer.delete_empty_rows,
    #                     2: transformer.delete_metadata_rows,
    #                     3: transformer.delete_notes_rows,
    #                     4: transformer.delete_derived_rows,
    #                     5: transformer.fill_down_group,
    #                     6: transformer.fill_right_header,
    #                     7: transformer.delete_derived_columns,
    #                     8: transformer.merge_adjacent_header_rows}
    #
    #         # randomly choose a start state
    #         state_vector = checker.check()
    #
    #         action_candidates = list(range(self.n_actions))
    #         randomizer.shuffle(action_candidates)
    #         state_vector.append(action_candidates[0])
    #         current_state = np.where(np.all(states == state_vector, axis=1))[0][0]
    #
    #         # state_candidates = list(range(r_matrix.shape[0]))
    #         # randomizer.shuffle(state_candidates)
    #         # current_state = state_candidates[0]
    #
    #         while True:
    #             valid_actions = r_matrix[current_state] >= -1
    #             # exploration
    #             if randomizer.rand() < self.__getattribute__('epsilon'):
    #                 actions = np.array(list(range(self.n_actions)))
    #                 actions = actions[valid_actions]
    #
    #                 if type(actions) is int:
    #                     actions = [actions]
    #                 randomizer.shuffle(actions)
    #                 action = actions[0]
    #             # exploitation
    #             else:
    #                 # if any q value of the current state is greater than zero, choose the biggest one
    #                 if np.sum(q_matrix[current_state]) > 0:
    #                     action = np.argmax(q_matrix[current_state])
    #                 else:
    #                     actions = np.array(list(range(self.n_actions)))
    #                     actions = actions[valid_actions]
    #                     randomizer.shuffle(actions)
    #                     action = actions[0]
    #             # execute the transform, and observe the result
    #             print('Execute action: ' + str(action))
    #             switcher[action]()
    #
    #             state_vector = checker.check()
    #             state_vector.append(action)
    #
    #             next_state = np.where(np.all(states == state_vector, axis=1))[0][0]
    #
    #             update_q(q_matrix, r_matrix, current_state, next_state, action,
    #                      self.__getattribute__('beta'), self.__getattribute__('gamma'))
    #             if current_state in terminate_states:
    #                 break
    #
    #             current_state = next_state
    #         print()
    #
    #     # use q matrix to construct the best sequence
    #     for i in range(self.n_actions):
    #         transformer = Transformer(self.dataset, self.cell_classes)
    #         checker = Checker(transformer)
    #         switcher = {0: transformer.delete_empty_columns,
    #                     1: transformer.delete_empty_rows,
    #                     2: transformer.delete_metadata_rows,
    #                     3: transformer.delete_notes_rows,
    #                     4: transformer.delete_derived_rows,
    #                     5: transformer.fill_down_group,
    #                     6: transformer.fill_right_header,
    #                     7: transformer.delete_derived_columns,
    #                     8: transformer.merge_adjacent_header_rows}
    #
    #         pipeline = []
    #
    #         state_vector = checker.check()
    #         state_vector.append(i)
    #         current_state = np.where(np.all(states == state_vector, axis=1))[0][0]
    #         max_steps = 20
    #         step = 0
    #         action = i
    #         while current_state not in terminate_states and step < max_steps:
    #             # pipeline.append(current_state)
    #             pipeline.append(action)
    #             action = np.argmax(q_matrix[current_state])
    #             switcher[action]()
    #             state_vector = checker.check()
    #             state_vector.append(action)
    #             current_state = np.where(np.all(states == state_vector, axis=1))[0][0]
    #             step += 1
    #         pipeline.append(action)
    #         print(pipeline)
    #         print(csv_sim(transformer.csv_data, self.ground_truth)[0])
    #         print()
    #
    #     pass


if __name__ == '__main__':
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

    # data = [['Title', '', '', '', ''], ['', '', '', '', ''], ['', 'header1', '', '', 'header2'],
    #                  ['', 'header2', 'header5', 'header3', 'header4'], ['groups', '', '', '', ''], ['', 31, '', 32, 2], ['', 32, '', 10, 2.2],
    #                  ['', 33, '', 6, 1.2], ['', 32, '', 1, 1.2]]
    # cell_classes = [[CELL_METADATA, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
    #                 [CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
    #                 [CELL_EMPTY, CELL_HEADER, CELL_EMPTY, CELL_EMPTY, CELL_HEADER], [CELL_EMPTY, CELL_HEADER, CELL_HEADER, CELL_HEADER, CELL_HEADER],
    #                 [CELL_GROUP_HEADER, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY, CELL_EMPTY],
    #                 [CELL_EMPTY, CELL_DATA, CELL_EMPTY, CELL_DATA, CELL_DATA], [CELL_EMPTY, CELL_DATA, CELL_EMPTY, CELL_DATA, CELL_DATA],
    #                 [CELL_EMPTY, CELL_DATA, CELL_EMPTY, CELL_DATA, CELL_DATA], [CELL_EMPTY, CELL_DERIVED, CELL_EMPTY, CELL_DERIVED, CELL_DERIVED]]

    qlearner = Qlearner(data, cell_classes, _matrix_gs_path, None, None, None)
    # qlearner = Qlearner(data, cell_classes, data, None, None, None)
    # (beta = 0.1, gamma = 1, epsilon = 0.05) works
    qlearner.set_params(beta=0.1, gamma=1, epsilon=0.05, n_episodes=5E2)
    best_construction = qlearner.learning()

    construction_output = '/Users/lan/Documents/hpi/manual-30/constructions/'

    construction_output += '%s@%s.csv' % (tables[0][2], tables[0][3])

    with open(construction_output, 'w') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerows(best_construction[2])