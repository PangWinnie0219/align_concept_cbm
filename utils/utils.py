import os
import time

import json
import numpy as np
import yaml



def conf95(accs):
    return float(1.96 * np.std(accs) / np.sqrt(len(accs)))


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



def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


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