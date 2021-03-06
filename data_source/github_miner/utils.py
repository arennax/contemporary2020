import pandas as pd
import os
import json
import pdb
import glob
import subprocess
import shutil


def get_repo_names(src="./data/data1.json"):
    with open(src) as f:
        data_file = json.load(f)
        ret = []
        for node in data_file["data"]["search"]["edges"]:
            url = node["node"]["url"]
            repo = url[url.index(".com") + 5:]
            ret.append(repo)
    return ret


def get_existing_results(src="./results/"):
    files = glob.glob(src + "*.csv")
    ret = []
    for file in files:
        ret.append(file.split("/")[-1].split("_")[0])
    return ret


def get_commits_from_clone_repo(src="jagregory/abrash-black-book"):
    os.makedirs("./tmp_folder", exist_ok=True)
    os.chdir("./tmp_folder")
    folder_name = src.split("/")[-1]

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    full_path = "https://github.com/" + src + ".git"
    subprocess.check_call(["git", "clone", full_path])

    os.chdir(f"{folder_name}")
    x = subprocess.check_output(["git", "log", "--pretty=%H_%cd_%s",
                                 "--date=local", "--date=iso-local", ])
    commits_list = x.decode("utf-8").split("\n")

    res = {}
    for comm in commits_list[:-1]:
        temp = comm.split("_")
        head = temp[0]
        res[head] = temp[1:]

    os.chdir(f"../..")
    shutil.rmtree(f"./tmp_folder/{folder_name}")
    return res


if __name__ == "__main__":
    get_commits_from_clone_repo()
    # get_repo_names()
    # get_existing_results()
