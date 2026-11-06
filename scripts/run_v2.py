"""Execute and validate the v2 notebook: python scripts/run_v2.py."""
import asyncio
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    protected = [root / name for name in tracked if name and '_v2' not in name and '/v2_' not in name]
    hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in protected if p.is_file()}
    path = root / 'notebooks/02_telco_churn_v2_xgboost.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    NotebookClient(notebook, timeout=1800, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    nbformat.write(notebook, path)
    metrics = pd.read_csv(root / 'reports/model_comparison_v2.csv')
    assert len(metrics) == 4 and metrics.model.is_unique
    assert metrics[['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'validation_threshold']].apply(
        lambda col: col.between(0, 1).all()).all()
    assert (metrics[['tn', 'fp', 'fn', 'tp']].sum(axis=1) == 1409).all()
    assert (metrics.tp + metrics.fn == 374).all()
    assert (metrics.contacted == metrics.tp + metrics.fp).all()
    assert np.allclose(metrics.f1, 2 * metrics.tp / (2 * metrics.tp + metrics.fp + metrics.fn))
    importance = pd.read_csv(root / 'reports/feature_importance_v2.csv')
    assert len(importance) == 19 and importance.feature.is_unique
    assert importance.notna().all().all()
    assert np.isclose(importance.gain_share.sum(), 1)
    for name in ['v2_04_model_curves.png', 'v2_05_feature_importance.png']:
        assert (root / 'reports/figures' / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert all(hashlib.sha256(p.read_bytes()).digest() == digest for p, digest in hashes.items())
    print('PASS: notebook executed, metrics and importance validated, existing files unchanged.')


if __name__ == '__main__':
    main()
