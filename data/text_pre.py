import csv
import os

from transformers import BertTokenizer


DEFAULT_TEXT_MODEL = "bert-base-uncased"
_TOKENIZER_CACHE = {}


def get_tokenizer(pretrained_model=None):
    model_name = pretrained_model or DEFAULT_TEXT_MODEL
    if model_name not in _TOKENIZER_CACHE:
        try:
            tokenizer = BertTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = tokenizer
    return _TOKENIZER_CACHE[model_name]


def get_t_data(args, data_args):
    tokenizer = get_tokenizer(args.text_pretrained_model)
    return {
        split: [
            _encode_text(text, tokenizer, args.text_seq_len)
            for text in _read_text_column(data_args["data_path"], split, args.dataset)
        ]
        for split in ("train", "dev", "test")
    }


def _read_text_column(data_path, split, dataset):
    text_column = 3 if dataset == "MIntRec" else 2
    path = os.path.join(data_path, f"{split}.tsv")
    with open(path, "r", encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        return [row[text_column] for row in rows]


def _encode_text(text, tokenizer, max_length):
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    return [encoded["input_ids"], encoded["attention_mask"]]
