import gc
import os
import pickle

import numpy as np


def get_v_a_data(data_args, feats_path):
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Feature file not found: {feats_path}")
    with open(feats_path, "rb") as f:
        features = pickle.load(f)
    return padding_feats(data_args, features)


def padding(feature, max_length):
    feature = np.asarray(feature)
    if feature.ndim == 3:
        feature = feature.squeeze(1)
    if feature.shape[0] >= max_length:
        return feature[:max_length]
    padding_shape = (max_length - feature.shape[0], feature.shape[-1])
    zero_padding = np.zeros(padding_shape, dtype=feature.dtype)
    return np.concatenate((feature, zero_padding), axis=0)


def padding_feats(data_args, features):
    max_length = data_args["max_seq_len"]
    result = {}

    for split in list(features):
        split_features = features.pop(split)
        padded = []
        for feature in split_features:
            padded.append(padding(feature, max_length))
        result[split] = {"feats": padded}
        del split_features
        gc.collect()

    return result
