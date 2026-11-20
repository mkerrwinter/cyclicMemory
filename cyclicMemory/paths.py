"""
paths.py
--------
Central path configuration for the cyclicMemory project.

Cluster paths (DATA_DIR, RESULTS_DIR) point to the project space on Snellius
so that models, datasets and plots do not consume home-space quota.

Local paths (LOCAL_*) are used when running scripts (e.g. plotting.py) on
a local machine against a local copy of the cluster results.
"""

import os

# ---------------------------------------------------------------------------
# Cluster (Snellius project space)
# ---------------------------------------------------------------------------

# Set CYCLIC_MEMORY_PROJECT_BASE to the project directory on the cluster.
PROJECT_BASE = os.environ.get("CYCLIC_MEMORY_PROJECT_BASE", "/path/to/cyclicMemory")

DATA_DIR    = os.path.join(PROJECT_BASE, "data")
RESULTS_DIR = os.path.join(PROJECT_BASE, "snellius_results")

# ---------------------------------------------------------------------------
# Local laptop
# ---------------------------------------------------------------------------

LOCAL_PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCAL_DATA_DIR    = os.path.join(LOCAL_PROJECT_BASE, "data")
LOCAL_RESULTS_DIR = os.path.join(LOCAL_PROJECT_BASE, "snellius_results")
