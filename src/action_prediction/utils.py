import os
import pickle
from filelock import FileLock
import dataclasses


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(file_name, data):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


class File:
    def __init__(self, file_name):
        self.file_name = file_name

    def load(self):
        if not os.path.exists(self.file_name):
            return None
        else:
            with FileLock(self.file_name + ".lock"):
                return load_pickle(self.file_name)

    def save(self, data):
        with FileLock(self.file_name + ".lock"):
            save_pickle(self.file_name, data)

    def update(self, id, item):
        with FileLock(self.file_name + ".lock"):
            if os.path.exists(self.file_name):
                data = load_pickle(self.file_name)
                data[id] = item
            else:
                data = {id: item}
            save_pickle(self.file_name, data)


class Embeddings(File):
    def __init__(self, path: str):
        file_name = os.path.join(path, f"knn_embeddings.pkl")
        super().__init__(file_name)


class Reflexion(File):
    def __init__(self, path: str):
        file_name = os.path.join(path, f"reflexion.pkl")
        super().__init__(file_name)


def convert_dict_to_dataclass(cls, data):
    if isinstance(data, dict):
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        return cls(
            **{k: convert_dict_to_dataclass(field_types[k], v) for k, v in data.items()}
        )
    return data
