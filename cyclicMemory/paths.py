"""
paths.py
--------
Central path configuration for the cyclicMemory project.

DATA_DIR and RESULTS_DIR point to the project space on the Snellius cluster
so that models, datasets and plots do not consume home-space quota.
"""

import os

PROJECT_BASE = "/projects/prjs1487/python/cyclicMemory"

DATA_DIR    = os.path.join(PROJECT_BASE, "data")
RESULTS_DIR = os.path.join(PROJECT_BASE, "snellius_results")
