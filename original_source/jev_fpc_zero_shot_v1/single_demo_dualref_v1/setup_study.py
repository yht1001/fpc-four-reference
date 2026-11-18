"""Bind the existing FPC request and inference code to this study's paths."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
FPC = ROOT.parent
DATA = FPC.parent / "datasets/FPCdataset2601"
DEMO = "192.168.1.64_01_20260106104620202_part000_480p"
BLOCKS = {"threshold": (32.0, 45.0), "temporal": (47.0, 60.0)}
BANKS = ("cycle_a", "cycle_b")
sys.path.insert(0, str(FPC / "bin02_dualref_v1"))

import bin02_env as backend

backend.R = ROOT
backend.REFS = {bank: ROOT / "references" / bank for bank in BANKS}
