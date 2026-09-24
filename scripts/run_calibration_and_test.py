# -*- coding: utf-8 -*-
"""Compatibility entry point for the current provenance-bound V6 pipeline.

The former V3 script could overwrite a calibration artifact without binding it
to the checkpoint or the selected data. Keep the historical command name, but
delegate to the maintained V6 implementation so it inherits the same
disjoint-split and SHA-256 checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_and_test_v6 import main


if __name__ == "__main__":
    main()
