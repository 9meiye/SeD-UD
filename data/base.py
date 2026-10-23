import copy
import csv
import os

from . import benchmarks
from .mm_pre import MMDataset
from .text_pre import get_t_data
from .utils import get_v_a_data


class DataManager:
    def __init__(self, args):
        benchmark = benchmarks[args.dataset]
        lengths = benchmark["max_seq_lengths"]
        dims = benchmark["feat_dims"]

        args.text_seq_len = lengths["text"]
        args.video_seq_len = lengths["video"]
        args.audio_seq_len = lengths["audio"]
        args.text_feat_dim = dims["text"]
        args.video_feat_dim = dims["video"]
        args.audio_feat_dim = dims["audio"]
        self.mm_data = get_original_data(args)


def _resolve_feature_path(data_path, default_name, fallback_name):
    default_path = os.path.join(data_path, default_name)
    if os.path.exists(default_path):
        return default_path
    fallback_path = os.path.join(data_path, fallback_name)
    return fallback_path if os.path.exists(fallback_path) else default_path


def get_original_data(args):
    data_path = os.path.join(args.data_path, args.dataset)
    benchmark = benchmarks[args.dataset]
    labels = copy.deepcopy(benchmark["labels"])
    args.num_labels = len(labels)

    train_ids, train_labels = get_indexes_annotations(
        args, labels, os.path.join(data_path, "train.tsv")
    )
    dev_ids, dev_labels = get_indexes_annotations(
        args, labels, os.path.join(data_path, "dev.tsv")
    )
    test_ids, test_labels = get_indexes_annotations(
        args, labels, os.path.join(data_path, "test.tsv")
    )
    args.num_train_examples = len(train_ids)

    text_data = get_t_data(
        args,
        {
            "data_path": data_path,
            "train_data_index": train_ids,
            "dev_data_index": dev_ids,
            "test_data_index": test_ids,
            "max_seq_len": args.text_seq_len,
        },
    )

    video_path = _resolve_feature_path(
        data_path, "video_data/video_feats.pkl", "video_data/swin_feats.pkl"
    )
    video_data = get_v_a_data(
        {
            "train_data_index": train_ids,
            "dev_data_index": dev_ids,
            "test_data_index": test_ids,
            "max_seq_len": args.video_seq_len,
        },
        video_path,
    )

    audio_path = _resolve_feature_path(
        data_path, "audio_data/audio_feats.pkl", "audio_data/wavlm_feats.pkl"
    )
    audio_data = get_v_a_data(
        {
            "train_data_index": train_ids,
            "dev_data_index": dev_ids,
            "test_data_index": test_ids,
            "max_seq_len": args.audio_seq_len,
        },
        audio_path,
    )

    return {
        "train": MMDataset(
            train_labels, text_data["train"], video_data["train"], audio_data["train"]
        ),
        "dev": MMDataset(
            dev_labels, text_data["dev"], video_data["dev"], audio_data["dev"]
        ),
        "test": MMDataset(
            test_labels, text_data["test"], video_data["test"], audio_data["test"]
        ),
    }


def get_indexes_annotations(args, labels, read_file_path):
    label_map = {label: index for index, label in enumerate(labels)}
    indexes = []
    label_ids = []

    with open(read_file_path, "r", encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        for row in rows:
            if args.dataset == "MIntRec":
                indexes.append("_".join(row[:3]))
                label_ids.append(label_map[row[4]])
            elif args.dataset == "MELD-DA":
                indexes.append("_".join(row[:2]))
                label_ids.append(label_map[row[3]])
            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")

    return indexes, label_ids
