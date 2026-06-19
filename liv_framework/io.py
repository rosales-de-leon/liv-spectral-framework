import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_liv_scans(scan_dir, BB_labels, model_names):
    
    scans = {model: [] for model in model_names}

    for bb in BB_labels:
        for model in model_names:
            
            file = Path(scan_dir) / f"LIV_scan_{bb}_{model}_v1.csv"
            
            if file.exists():
                df = pd.read_csv(file)
                scans[model].append(df)
            else:
                print(f"Missing file: {file}")

    return scans
