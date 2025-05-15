import h5py
import numpy as np
import os


def flatten_and_convert(data, parent_key='', sep='#'):
    """ Recursively flattens a nested dictionary and converts lists to numpy arrays if numeric. """
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        # Recursively flatten dictionaries
        if isinstance(v, dict):
            items.extend(flatten_and_convert(v, new_key, sep=sep).items())

        # Handle lists
        elif isinstance(v, list):
            # If the list is numeric, convert to numpy array
            if all(isinstance(i, (int, float, np.integer, np.floating)) for i in v):
                v = np.array(v, dtype=np.float32)
                items.append((new_key, v))
            elif all(isinstance(i, np.ndarray) for i in v):
                # Keep lists of numpy arrays as they are
                items.append((new_key, v))
            else:
                print(f"Skipping list at key '{new_key}': Mixed or unsupported types.")

        # If already a numpy array or scalar
        elif isinstance(v, (int, float, np.integer, np.floating)):
            items.append((new_key, np.array([v], dtype=np.float32)))
        elif isinstance(v, np.ndarray):
            items.append((new_key, v))
        else:
            print(f"Skipping key '{new_key}': Unsupported type {type(v)}")

    return dict(items)


def save_training_data(run_id, result_dict):
    os.makedirs("trainings", exist_ok=True)
    file_path = os.path.join("trainings", f"{run_id}.h5")

    # Flatten and preprocess the result_dict
    result_dict = flatten_and_convert(result_dict)

    with h5py.File(file_path, "w") as f:
        for key, value in result_dict.items():
            print(f"Save {key}")
            if isinstance(value, list) and all(isinstance(v, np.ndarray) for v in value):
                group = f.create_group(key)
                for idx, emb in enumerate(value):
                    group.create_dataset(f"{idx}", data=emb)
            elif isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, (int, float)):
                f.create_dataset(key, data=np.array([value]))
            else:
                print(f"Skipping key '{key}': Unsupported data format: {type(value)}")


def unflatten_dict(flat_dict, sep='#'):
    """ Reconstruct nested dictionary from a flattened dictionary. """
    unflattened = {}
    for k, v in flat_dict.items():
        keys = k.split(sep)
        d = unflattened
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return unflattened


def load_training_data(run_id):
    file_path = os.path.join("trainings", f"{run_id}.h5")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    data = {}

    with h5py.File(file_path, "r") as f:
        def read_dataset(name, obj):
            """ Read datasets and groups recursively. """
            if isinstance(obj, h5py.Dataset):
                data[name] = np.array(obj)
            elif isinstance(obj, h5py.Group):
                # Handle groups as lists of datasets
                group_data = []
                for key in obj:
                    group_data.append(np.array(obj[key]))
                data[name] = group_data

        f.visititems(read_dataset)

    # Reconstruct the nested dictionary
    return unflatten_dict(data)