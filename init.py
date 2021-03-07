#%%
import os
import requests

DATA_DIR = 'data'
RESOURCES_DIR = 'resources'
MODELS_DIR = 'models'
STOP_WORDS_PATH = os.path.join(RESOURCES_DIR, 'stopwords.txt')

REPOSITORY_PATH = "https://github.com/hse-aml/natural-language-processing"
STOPWORDS_URL = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"


def download_file(url, file_path):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    try:
        with open(file_path, 'wb', buffering=16*1024*1024) as f:
            for chunk in r.iter_content(32 * 1024):
                f.write(chunk)
    except Exception:
        print("Download failed")
    finally:
        if os.path.getsize(file_path) != total_size:
            os.remove(file_path)
            print("Removed incomplete download")


def download_from_github(version, fn, target_dir, force=False):
    url = REPOSITORY_PATH + "/releases/download/{0}/{1}".format(version, fn)
    file_path = os.path.join(target_dir, fn)
    if os.path.exists(file_path) and not force:
        print("File {} is already downloaded.".format(file_path))
        return
    download_file(url, file_path)


def sequential_downloader(version, fns, target_dir, force=False):
    os.makedirs(target_dir, exist_ok=True)
    for fn in fns:
        download_from_github(version, fn, target_dir, force=force)


def download_project_resources(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    download_file(STOPWORDS_URL, STOP_WORDS_PATH)
    sequential_downloader(
        "project",
        [
            "dialogues.tsv",
            "tagged_posts.tsv",
        ],
        DATA_DIR,
        force=force
    )

download_project_resources(force=False)
# %%
