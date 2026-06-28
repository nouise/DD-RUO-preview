#!/usr/bin/env python3
"""
DD-RUO ModelScope 上传脚本

仓库: yiping03/dd-ruo0 (public)
网页: https://www.modelscope.cn/models/yiping03/dd-ruo0
下载: snapshot_download('yiping03/dd-ruo0')

用法:
  MODELSCOPE_API_TOKEN=xxx python upload_modelscope.py <local_dir> <path_in_repo>
  例: python upload_modelscope.py /tmp/upload/DC/imagemeow checkpoints_release/DC/imagemeow
"""

import os, sys, time, traceback

TOKEN = os.environ.get('MODELSCOPE_API_TOKEN', '')
REPO = 'yiping03/dd-ruo0'

def upload_folder(local_dir, path_in_repo, max_workers=8, max_retries=6):
    from modelscope.hub.api import HubApi
    api = HubApi()
    for attempt in range(1, max_retries + 1):
        try:
            print(f"=== UPLOAD ATTEMPT {attempt}/{max_retries} ===", flush=True)
            print(f"local: {local_dir}  ->  repo: {path_in_repo}", flush=True)
            api.upload_folder(
                repo_id=REPO,
                folder_path=local_dir,
                path_in_repo=path_in_repo,
                commit_message=f'Upload {path_in_repo} (attempt {attempt})',
                token=TOKEN,
                repo_type='model',
                max_workers=max_workers,
            )
            print("UPLOAD_DONE_OK", flush=True)
            return True
        except Exception as e:
            print(f"ATTEMPT {attempt} FAILED: {e}", flush=True)
            traceback.print_exc()
            if attempt < max_retries:
                wait = 60 * attempt
                print(f"retry in {wait}s...", flush=True)
                time.sleep(wait)
    print("ALL_RETRIES_EXHAUSTED", flush=True)
    return False

if __name__ == '__main__':
    if not TOKEN:
        print("ERROR: 请先设置环境变量 MODELSCOPE_API_TOKEN", file=sys.stderr)
        sys.exit(2)
    if len(sys.argv) < 3:
        print("用法: python upload_modelscope.py <local_dir> <path_in_repo>")
        sys.exit(2)
    ok = upload_folder(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
