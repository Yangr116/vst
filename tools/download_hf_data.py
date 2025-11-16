import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_ETAG_TIMEOUT'] = '500'

from datasets import load_dataset
from tqdm import tqdm
import json
import time
from huggingface_hub import hf_hub_download, snapshot_download


def download_repo(repo_id, cache_dir, local_dir):
    try:
        snapshot_download(repo_id=repo_id, 
                        local_dir=local_dir,
                        cache_dir=cache_dir,
                        local_dir_use_symlinks=False,
                        repo_type="dataset",
                        max_workers=16)
        return True
    except Exception as e:
        print(e)
        print("Sleep 300s and try again.")
        time.sleep(300)
        return False

def download(repo_id, cache_dir, local_dir):
    print(f"Downloading...")
    success = download_repo(repo_id, cache_dir, local_dir)
    idx = 0
    while not success and idx < 5:
        success = download_repo(repo_id, cache_dir, local_dir)
        idx += 1
    
    if success:
        print(f"done!")

if __name__ == "__main__":
    repo_ids = [
        "lmms-lab/LLaVA-NeXT-Data"
        ]
    cache_dir = "/mnt/bn/ic-vlm/rayyang/cache/cache_hf"
    local_dir = "/mnt/bn/ic-vlm/rayyang/data/2D/"
    os.makedirs(cache_dir, exist_ok=True)
    for repo_id in repo_ids:
        _local_dir = os.path.join(local_dir, os.path.basename(repo_id))
        os.makedirs(_local_dir, exist_ok=True)
        download(repo_id, cache_dir=cache_dir, local_dir=_local_dir)
