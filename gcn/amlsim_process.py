import argparse
import datetime
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 1212


def load_transactions(file_path: str) -> pd.DataFrame:
    dtypes = {
        'tran_id': int,
        'orig_acct': int,
        'bene_acct': int,
        'tx_type': str,
        'base_amt': float,
        'tran_timestamp': str,
        'is_sar': int,
        'alert_id': int
    }

    df = pd.read_csv(file_path, dtype=dtypes, parse_dates=['tran_timestamp'])

    return df


def generate_feature_vectors(input_file_path: str, output_folder_path: str):
    feat_df = load_transactions(input_file_path)
    feat_df.drop(['tran_id', 'alert_id'], axis=1, inplace=True)

    categories = pd.get_dummies(feat_df['tx_type'], dtype=int)

    for col in reversed(categories.columns):
        feat_df.insert(loc=2, column=str.lower(col), value=categories.loc[:, col])
    feat_df.drop('tx_type', axis=1, inplace=True)

    feat_df['tran_timestamp'] = (feat_df['tran_timestamp'] -
                                 pd.Timestamp('1970-01-01 00:00:00+00:00')) // pd.Timedelta('1s')

    with open(output_folder_path + 'ind.amlsim.allx', 'wb') as f:
        pickle.dump(
            csr_matrix(feat_df.drop(['is_sar', 'orig_acct', 'bene_acct'], axis=1).values.tolist()),
            f)

    with open(output_folder_path + 'ind.amlsim.ally', 'wb') as f:
        ohe = OneHotEncoder(sparse=False, dtype=int)
        pickle.dump(ohe.fit_transform(feat_df['is_sar'].values.reshape(-1, 1)), f)

    feat_df_train_x, feat_df_test_x, feat_df_train_y, feat_df_test_y = train_test_split(
        feat_df.drop('is_sar', axis=1),
        feat_df['is_sar'],
        train_size=0.8,
        shuffle=True,
        random_state=SEED)

    feat_csr_train_x = csr_matrix(
        feat_df_train_x.drop(['orig_acct', 'bene_acct'], axis=1).values.tolist())
    feat_csr_test_x = csr_matrix(
        feat_df_test_x.drop(['orig_acct', 'bene_acct'], axis=1).values.tolist())

    with open(output_folder_path + 'ind.amlsim.x', 'wb') as f:
        pickle.dump(feat_csr_train_x, f)

    with open(output_folder_path + 'ind.amlsim.tx', 'wb') as f:
        pickle.dump(feat_csr_test_x, f)

    with open(output_folder_path + 'ind.amlsim.y', 'wb') as f:
        ohe = OneHotEncoder(sparse=False, dtype=int)
        pickle.dump(ohe.fit_transform(feat_df_train_y.values.reshape(-1, 1)), f)

    with open(output_folder_path + 'ind.amlsim.ty', 'wb') as f:
        ohe = OneHotEncoder(sparse=False, dtype=int)
        pickle.dump(ohe.fit_transform(feat_df_test_y.values.reshape(-1, 1)), f)

    with open(output_folder_path + 'ind.amlsim.test.index', 'w') as f:
        for v in feat_df_test_x['orig_acct']:
            f.write(str(v) + '\n')


def generate_graph(input_file_path: str, output_folder_path: str):
    df = load_transactions(input_file_path)

    graph = defaultdict()
    for i, row in df.iterrows():
        try:
            graph[row['orig_acct']].append(row['bene_acct'])
        except KeyError:
            graph[row['orig_acct']] = [row['bene_acct']]

    for v in graph.keys():
        graph[v] = list(set(graph[v]))

    with open(output_folder_path + 'ind.amlsim.graph', 'wb') as f:
        pickle.dump(graph, f)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='AMLSim transactions data input file path.')
    parser.add_argument('output', type=str, help='Transformed AMLSim data folder output path.')

    args = parser.parse_args()

    return args


def generate_amlsim_data():
    args = arg_parser()
    generate_feature_vectors(args.input, args.output)
    generate_graph(args.input, args.output)


if __name__ == '__main__':
    generate_amlsim_data()
