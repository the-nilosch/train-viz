import h5py
import numpy as np
import os

from helper.visualization import Animation, Run


def flatten_and_convert(data, parent_key='', sep='#'):
    """ Recursively flattens a nested dictionary and converts lists to numpy arrays if numeric, and keeps strings. """
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        # Recursively flatten dictionaries
        if isinstance(v, dict):
            items.extend(flatten_and_convert(v, new_key, sep=sep).items())

        # Handle lists
        elif isinstance(v, list):
            if all(isinstance(i, (int, float, np.integer, np.floating)) for i in v):
                v = np.array(v, dtype=np.float32)
                items.append((new_key, v))
            elif all(isinstance(i, list) and all(isinstance(x, (int, float)) for x in i) for i in v):
                # Convert list-of-lists of floats to list of np arrays
                array_list = [np.array(i, dtype=np.float32) for i in v]
                items.append((new_key, array_list))
            elif all(isinstance(i, np.ndarray) for i in v):
                items.append((new_key, v))
            else:
                print(f"Skipping list at key '{new_key}': Mixed or unsupported types.")


        # Scalars
        elif isinstance(v, (int, float, np.integer, np.floating)):
            items.append((new_key, np.array([v], dtype=np.float32)))
        elif isinstance(v, np.ndarray):
            items.append((new_key, v))
        elif isinstance(v, str):
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
            elif isinstance(value, str):
                # Store string as a special dataset with dtype 'S' (fixed ASCII) or 'utf-8'
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset(key, data=value, dtype=dt)
            else:
                print(f"Skipping key '{key}': Unsupported data format: {type(value)}")


def unflatten_dict(flat_dict, sep='#'):
    """ Reconstruct nested dictionary from a flattened dictionary. """
    unflattened = {}
    for k, v in flat_dict.items():
        if isinstance(v, bytes):
            v = v.decode('utf-8')
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
        def read_group(name, obj):
            if isinstance(obj, h5py.Group):
                # Read group ONCE as a list of its datasets
                group_data = []
                sorted_keys = sorted(obj.keys(), key=lambda x: int(x) if x.isdigit() else x)
                for key in sorted_keys:
                    group_data.append(np.array(obj[key]))
                data[name] = group_data
            elif isinstance(obj, h5py.Dataset):
                # Only add dataset if its parent is NOT in data
                parent = '/'.join(name.split('/')[:-1])
                if parent not in data:
                    val = obj[()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    data[name] = val

        f.visititems(read_group)

    return unflatten_dict(data)


def save_animation(ani: Animation):
    os.makedirs(f"trainings/{ani.run_id}", exist_ok=True)
    path = os.path.join("trainings", ani.run_id, f"{ani.title}.h5")
    with h5py.File(path, "a") as f:
        # clean previous
        if "projections" in f:
            del f["projections"]

        # projections
        pgrp = f.create_group("projections")
        for i, arr in enumerate(ani.projections):
            pgrp.create_dataset(str(i), data=np.asarray(arr), compression="gzip", shuffle=True)

        # CKA similarities (optional)
        if getattr(ani, "cka_similarities", None) is not None:
            cgrp = f.create_group("cka_similarities")
            for k, vals in ani.cka_similarities.items():
                cgrp.create_dataset(str(k), data=np.asarray(vals), compression="gzip", shuffle=True)

        # meta
        if hasattr(ani, "meta"):
            for k, v in ani.meta.items():
                if isinstance(v, str):
                    f.attrs[k] = v
                else:
                    f.attrs[k] = np.asarray(v)

def load_animation(run: Run, title: str):
    path = os.path.join("trainings", run.run_id, f"{title}.h5")
    with h5py.File(path, "r") as f:
        p = [np.array(f["projections"][k]) for k in sorted(f["projections"].keys(), key=int)]
        meta = {k: f.attrs[k] for k in f.attrs.keys()}

        cka_similarities = None
        if "cka_similarities" in f:
            cka_similarities = {
                int(k): np.array(f["cka_similarities"][k])
                for k in sorted(f["cka_similarities"].keys(), key=int)
            }

    ani = Animation(p, title, run)
    ani.meta = meta
    if cka_similarities is not None:
        ani.cka_similarities = cka_similarities

    return ani