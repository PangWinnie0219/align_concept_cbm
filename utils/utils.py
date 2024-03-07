import os
import time
import pickle
import time
import os
import json
import numpy as np
import yaml
import pandas as pd
import re
import hashlib

class LoopingIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def conf95(accs):
    return float(1.96 * np.std(accs) / np.sqrt(len(accs)))


def get_meanpm95conf_strlatex(accs):
    conf = float(1.96 * np.std(accs) / np.sqrt(len(accs)))
    mean = np.mean(accs)
    return f"${mean:.2f}\pm{conf:.2f}$"


def get_meanpm95conf_str(accs):
    conf = float(1.96 * np.std(accs) / np.sqrt(len(accs)))
    mean = np.mean(accs)
    return f"{mean:.2f}±{conf:.2f}"


def get_meanpm95conf_strlist(accs):
    return [f"{v:.2f}" for v in accs.dropna()]


def get_meanpm95conf_strall(accs):
    return (
        get_meanpm95conf_str(accs)
        + " ("
        + ",".join(get_meanpm95conf_strlist(accs))
        + ")"
    )


def setup_seed(seed):
    if seed < 0:
        if os.getenv("SATOSHI_SEED") is not None and seed == -2:
            seed = int(os.getenv("SATOSHI_SEED"))
            print("env seed used")
        else:
            import math

            seed = int(10**4 * math.modf(time.time())[0])
            seed = seed
    print("random seed", seed)
    return seed


def setup_savedir(
    prefix="", basedir="./experiments", args=None, append_args=[], add_time=True
):
    savedir = prefix
    if len(append_args) > 0 and args is not None:
        for arg_opt in append_args:
            arg_value = getattr(args, arg_opt)
            savedir += "_" + arg_opt + "-" + str(arg_value)
        if savedir.startswith("_"):
            savedir = savedir[1:]
    else:
        savedir += "exp"

    savedir = savedir.replace(" ", "").replace("'", "").replace('"', "")
    savedir = os.path.join(basedir, savedir)

    if add_time:
        now = time.localtime()
        d = time.strftime("%Y%m%d%H%M%S", now)
        savedir = savedir + "_" + str(d)

    # if exists, append _num-[num]
    i = 1
    savedir_ori = savedir
    while True:
        try:
            os.makedirs(savedir)
            break
        except FileExistsError:
            savedir = savedir_ori + "_num-%d" % i
            i += 1

    print("made the log directory", savedir)
    return savedir


def save_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
        print(path, "saved")


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_args(savedir, args, name="args.json", sort_keys=True, indent=4):
    # save args as "args.json" in the savedir
    path = os.path.join(savedir, name)
    with open(path, "w") as f:
        json.dump(vars(args), f, sort_keys=sort_keys, indent=indent)
    print("args saved as %s" % path)


def save_json(dict, path,sort_keys=False, indent=4):
    with open(path, "w") as f:
        json.dump(dict, f, sort_keys=sort_keys, indent=indent)
        print("log saved at %s" % path)


def gzip_file_and_remove_original(file_path):
    try:
        import gzip
    except ImportError:
        print("Cannot import gzip; please make sure it is installed.")
        return

    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return

    if not file_path.endswith(".gz"):
        gzipped_path = file_path + ".gz"
    else:
        gzipped_path = file_path

    with open(file_path, "rb") as original_file:
        with gzip.open(gzipped_path, "wb") as gzipped_file:
            gzipped_file.writelines(original_file)

    os.remove(file_path)
    print("File gzipped and original removed. Gzipped file saved as %s" % gzipped_path)


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_json_from_gzip(path):
    try:
        import gzip
    except ImportError:
        print("cannot import gzip ; try pip install gzip")
        assert False

    with gzip.open(path, "rb") as f:
        json_data = f.read().decode("utf-8")

    dict_data = json.loads(json_data)
    return dict_data


def get_unique_file_path(file_path):
    base, ext = os.path.splitext(file_path)
    num = 1
    new_file_path = f"{base}-{num}{ext}"
    while os.path.exists(new_file_path):
        num += 1
        new_file_path = f"{base}-{num}{ext}"
    return new_file_path


def check_gitstatus():
    import subprocess
    from subprocess import PIPE

    changed = "gitpython N/A"
    sha = None
    changed = None
    status = None
    # from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    # from https://stackoverflow.com/questions/31540449/how-to-check-if-a-git-repo-has-uncommitted-changes-using-python
    # from https://stackoverflow.com/questions/33733453/get-changed-files-using-gitpython/42792158
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            changed = [item.a_path for item in repo.index.diff(None)]
        except Exception as e:
            print(e)
    except ImportError:
        print("cannot import gitpython ; try pip install gitpython")

    if sha is None:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], stdout=PIPE, stderr=subprocess.PIPE
            )
            .stdout.decode("utf-8")
            .strip()
        )
    print("git hash", sha)

    if status is None:
        status = (
            subprocess.run(["git", "status"], stdout=PIPE, stderr=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )
    print("git status", status)

    return {"hash": sha, "changed": changed, "status": status}


def convert_value_to_prob(df_pred_value, attribute_encoders, label_smoothing=0.1):
    attribute_names = list(attribute_encoders.keys())
    num_attributes = len(attribute_names)

    df_pred_prob = []

    # For each row in df_pred_value
    for _, row in df_pred_value.iterrows():
        row_att_prob = {}
        # keep the columns except the attribute columns
        for col_name in row.index:
            if col_name not in attribute_names:
                row_att_prob[col_name] = row[col_name]

        # For each attribute
        for att_name in attribute_names:
            pred_att_value = row[att_name]
            num_values = len(attribute_encoders[att_name])

            # Initialize all probabilities with the smoothing value
            probs = [label_smoothing / num_values] * num_values

            # Assign 1 - label_smoothing to the predicted value
            pred_value_index = attribute_encoders[att_name][pred_att_value]
            probs[pred_value_index] = 1 - label_smoothing

            # Store the probabilities
            for att_value_idx, p_i in enumerate(probs):
                name = list(attribute_encoders[att_name].keys())[att_value_idx]
                name = name.replace(" ", "_")
                row_att_prob[f"{att_name}={name}"] = p_i

        df_pred_prob.append(row_att_prob)

    df_pred_prob = pd.DataFrame(df_pred_prob)

    return df_pred_prob

def convert_prob_to_value(df_pred_prob, attribute_encoders):
    attribute_names = list(attribute_encoders.keys())
    num_attributes = len(attribute_names)

    df_pred_value = []

    # For each row in df_pred_prob
    for _, row in df_pred_prob.iterrows():
        row_att_value = {}
        # Extract non-attribute columns
        non_attribute_cols = [col for col in row.index if '=' not in col]
        for col_name in non_attribute_cols:
            row_att_value[col_name] = row[col_name]

        # For each attribute
        for att_name in attribute_names:
            # Extract probabilities for the current attribute
            att_probs = [row[f"{att_name}={value.replace(' ', '_')}"] for value in attribute_encoders[att_name].keys()]
            
            # Get the index of the maximum probability value
            pred_value_index = np.argmax(att_probs)
            pred_att_value = list(attribute_encoders[att_name].keys())[pred_value_index]
            row_att_value[att_name] = pred_att_value.replace("_", " ")

        df_pred_value.append(row_att_value)

    df_pred_value = pd.DataFrame(df_pred_value)

    return df_pred_value

def remove_duplicate_spaces(input_string):
    return re.sub(r'\s+', ' ', input_string).strip()

def get_md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()