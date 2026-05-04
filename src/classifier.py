"""Turkish NLP Text Classification Pipeline.

Supports:
  - Zero-shot classification (no training needed)
  - Fine-tuning on custom datasets
  - Batch inference with confidence scores
  - sklearn-compatible metrics report

Author: Umit Sencer | github.com/Umitsencer
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class ClassificationResult:
    text: str
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(label='{self.predicted_label}', "
            f"confidence={self.confidence:.3f})"
        )


# ------------------------------------------------------------------ #
# Core classifier
# ------------------------------------------------------------------ #

class TurkishTextClassifier:
    """Multi-class Turkish text classifier backed by HuggingFace Transformers.

    Modes:
        zero_shot  - classify against arbitrary labels, no training needed
        pretrained - load a fine-tuned sequence classification model
    """

    # Default zero-shot model with Turkish support
    _DEFAULT_ZERO_SHOT = "joeddav/xlm-roberta-large-xnli"

    def __init__(
        self,
        model_name: str = _DEFAULT_ZERO_SHOT,
        mode: str = "zero_shot",
        labels: Optional[List[str]] = None,
        device: int = -1,   # -1 = CPU, 0 = GPU
    ) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers paketi yuklenmemis: pip install transformers torch")
        self.model_name = model_name
        self.mode = mode
        self.labels = labels or []
        self.device = device
        self._pipe = None

    def load(self) -> "TurkishTextClassifier":
        """Load model pipeline (downloads model on first call)."""
        task = "zero-shot-classification" if self.mode == "zero_shot" else "text-classification"
        self._pipe = pipeline(task, model=self.model_name, device=self.device)
        print(f"[Classifier] Model yuklendi: {self.model_name} | mode={self.mode}")
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def classify(self, text: str, labels: Optional[List[str]] = None) -> ClassificationResult:
        """Classify a single text string."""
        if self._pipe is None:
            raise RuntimeError("Once load() cagirin.")
        effective_labels = labels or self.labels
        if not effective_labels:
            raise ValueError("labels parametresi bos olamaz.")

        if self.mode == "zero_shot":
            output = self._pipe(text, candidate_labels=effective_labels, multi_label=False)
            scores = dict(zip(output["labels"], output["scores"]))
            best_label = output["labels"][0]
            confidence = output["scores"][0]
        else:
            output = self._pipe(text)
            best_label = output[0]["label"]
            confidence = output[0]["score"]
            scores = {best_label: confidence}

        return ClassificationResult(
            text=text,
            predicted_label=best_label,
            confidence=confidence,
            all_scores=scores,
        )

    def classify_batch(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
    ) -> List[ClassificationResult]:
        """Classify a list of texts (shows progress)."""
        results = []
        total = len(texts)
        for i, text in enumerate(texts, 1):
            result = self.classify(text, labels=labels)
            results.append(result)
            if i % 10 == 0 or i == total:
                print(f"[Classifier] {i}/{total} islendi.")
        return results

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        texts: List[str],
        true_labels: List[str],
        labels: Optional[List[str]] = None,
    ) -> Dict:
        """Run batch inference and return sklearn classification report."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn yuklenmemis.")
        results = self.classify_batch(texts, labels=labels)
        predicted = [r.predicted_label for r in results]
        report = classification_report(true_labels, predicted, output_dict=True, zero_division=0)
        cm = confusion_matrix(true_labels, predicted)
        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predicted,
            "accuracy": report.get("accuracy", 0.0),
        }

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess(text: str) -> str:
        """Basic Turkish text preprocessing."""
        text = text.strip()
        text = " ".join(text.split())          # normalize whitespace
        text = text[:512]                       # BERT token limit safety
        return text

    def is_loaded(self) -> bool:
        return self._pipe is not None
