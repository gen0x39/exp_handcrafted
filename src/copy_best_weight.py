import os
from pathlib import Path
import shutil

CUR = os.getcwd()
PARENT = Path(CUR).parent
ROOT = os.path.join(PARENT, 'result')
files = os.listdir(ROOT)
dir_names = [f for f in files if os.path.isdir(os.path.join(ROOT, f))]
arch_dirs = [s for s in dir_names if 'arch' in s]

for ad in arch_dirs:
    data_path = os.path.join(ROOT, ad)
    output_path = os.path.join(PARENT, 'best_params', ad[0:8])
    if not os.path.exists(output_path):
       os.mkdir(output_path)

    files = os.listdir(data_path)
    files_file = [f for f in files if os.path.isfile(os.path.join(data_path, f))]
    best_files = [s for s in files_file if 'model_best' in s]

    for bf in best_files:
        pt_file_path = os.path.join(data_path, bf)
        if not os.path.exists(os.path.join(output_path, bf)):
            shutil.copy2(pt_file_path, output_path+'/')

    log_files = [s for s in files_file if 'log.txt' in s]
    for lf in log_files:
        log_file_path = os.path.join(data_path, lf)
        if not os.path.exists(os.path.join(output_path, lf)):
            shutil.copy2(log_file_path, output_path+'/')