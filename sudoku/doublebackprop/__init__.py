"""
Double Backprop - Imitation Learning for XLM Framework

This package implements imitation learning with all necessary components:
- Model architecture (model.py)
- Loss function (loss.py)
- Predictor for inference (predictor.py)
- Data module (datamodule.py)
- Metrics computation (metrics.py)
- Type definitions (types.py)
- History tracking (history.py)
"""

from .model import RotaryTransformerModel, RotaryTransformerLoopholingModel
from .loss import MLMLoss, LoopholingBPTTPumaLoss
from .streaming_batch import StreamingBatch
from .predictor import ConfidenceBasedPredictor
from .datamodule import (
    PuzzleInfillPredCollator,
    PuzzleMLMCollator,
)
from .types import (
    MLMBatch,
    MLMSeq2SeqPredictionBatch,
    MLMUncondtionalPredictionBatch,
    MLMLossDict,
    MLMModel,
    MLMPredictionDict,
)
from .history import HistoryTopKPlugin
from .callbacks import ValidationThresholdSweepCallback

__all__ = [
    "RotaryTransformerModel",
    "RotaryTransformerLoopholingModel",
    "MLMLoss",
    "LoopholingBPTTPumaLoss",
    "StreamingBatch",
    "ConfidenceBasedPredictor",
    "PuzzleInfillPredCollator",
    "PuzzleMLMCollator",
    "MLMBatch",
    "MLMSeq2SeqPredictionBatch",
    "MLMUncondtionalPredictionBatch",
    "MLMLossDict",
    "MLMModel",
    "MLMPredictionDict",
    "HistoryTopKPlugin",
    "ValidationThresholdSweepCallback",
]
