"""ATR 自適應網格 · 台股零股每日建議系統。"""

__version__ = "1.0.0"

from .engine import BUY, HOLD, REVIEW, SELL, SKIP, Decision, evaluate, lot_size
from .fees import max_shares_for_min_fee

__all__ = [
    "BUY",
    "SELL",
    "HOLD",
    "REVIEW",
    "SKIP",
    "Decision",
    "evaluate",
    "lot_size",
    "max_shares_for_min_fee",
    "__version__",
]
