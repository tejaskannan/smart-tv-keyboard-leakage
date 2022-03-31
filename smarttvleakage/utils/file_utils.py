import re
import os
import gzip
import json
import pickle
import codecs
from collections import OrderedDict
from typing import Any, Optional, Iterable


def make_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


def iterate_dir(folder: str, pattern: Optional[str] = None) -> Iterable[str]:
    # Set pattern to match any string if None
    if pattern is None:
        pattern = r'.'

    # Always sort files to ensure consistent retrieval
    for file_name in sorted(os.listdir(folder)):
        match = re.match(pattern, file_name)
        if match is not None:
            yield os.path.join(folder, file_name)


def save_jsonl_gz(data: Iterable[Any], file_path: str) -> None:
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines gzip file.'

    with gzip.GzipFile(file_path, 'wb') as f:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(f).write(json.dumps(element))
            writer(f).write('\n')


def read_jsonl_gz(file_path: str) -> Iterable[Any]:
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines gzip file.'

    reader = codecs.getreader('utf-8')
    with gzip.open(file_path) as f:
        for line in reader(f):
            yield json.loads(line, object_pairs_hook=OrderedDict)


def append_jsonl_gz(data: Any, file_path: str):
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines gzip file.'

    with gzip.GzipFile(file_path, 'ab') as f:
        writer = codecs.getwriter('utf-8')
        writer(f).write(json.dumps(data))
        writer(f).write('\n')


def save_pickle_gz(data: Any, file_path: str):
    assert file_path.endswith('.pkl.gz'), 'Must provide a pickle gzip file.'

    with gzip.GzipFile(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle_gz(file_path: str) -> Any:
    assert file_path.endswith('.pkl.gz'), 'Must provde a pickle gzip file.'

    with gzip.open(file_path) as f:
        return pickle.load(f)


def read_pickle(file_path: str) -> Any:
    assert file_path.endswith('.pkl'), 'Must provide a pickle file.'
    with open(file_path, 'rb') as f:
        return pickle.load(file_path)


def save_pickle(data: Any, file_path: str):
    assert file_path.endswith('.pkl'), 'Must provide a pickle file.'
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_json(file_path: str) -> OrderedDict:
    assert file_path.endswith('.json'), 'Must provide a json file.'
    with open(file_path, 'r') as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def save_json(data: Any, file_path: str):
    assert file_path.endswith('.json'), 'Must provide a json file.'
    with open(file_path, 'w') as f:
        json.dump(data, f)


def read_json_gz(file_path: str) -> Any:
    assert file_path.endswith('.json.gz'), 'Must provide a json gzip file.'

    with gzip.GzipFile(file_path, 'rb') as f:
        reader = codecs.getreader('utf-8')
        return json.load(reader(f), object_pairs_hook=OrderedDict)


def save_json_gz(data: Any, file_path: str):
    assert file_path.endswith('.json.gz'), 'Must provide a json gzip file.'

    with gzip.GzipFile(file_path, 'wb') as f:
        writer = codecs.getwriter('utf-8')
        writer(f).write(json.dumps(data))
