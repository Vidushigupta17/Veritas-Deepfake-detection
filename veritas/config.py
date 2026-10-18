from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Config:
    # Step 3: band-pass for heart-rate (Hz)
    low_hz: float = 0.7
    high_hz: float = 3.0
    # Step 4/5: short-video windows
    window_sec: float = 10.0
    stride_sec: float = 5.0
    # Signal method: "green" | "chrom" | "pos"
    method: str = "green"
    # Cap frames processed per video (None = all)
    max_frames: int | None = None

REGIONS = ["forehead", "left_cheek", "right_cheek"]
PAIRS = [("forehead", "left_cheek"), ("forehead", "right_cheek"), ("left_cheek", "right_cheek")]
