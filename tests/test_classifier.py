"""Tests for TurkishTextClassifier.

Run: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch

try:
    from src.classifier import TurkishTextClassifier, ClassificationResult
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

requires_deps = pytest.mark.skipif(not DEPS_AVAILABLE, reason="transformers not installed")


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

SAMPLE_LABELS = ["spor", "ekonomi", "siyaset", "teknoloji", "saglik"]

SAMPLE_TEXTS = [
    "Fenerbahce sezonu sampiyonlukla kapatti.",
    "Merkez bankasi faiz oranini yuzde 50 olarak belirledi.",
    "Secim kampanyasi tum hizyla devam ediyor.",
    "Yeni yapay zeka modeli rekor kirdi.",
]

SAMPLE_TRUE_LABELS = ["spor", "ekonomi", "siyaset", "teknoloji"]


# ------------------------------------------------------------------ #
# Unit tests — no model download needed
# ------------------------------------------------------------------ #

class TestClassificationResult:
    """Test the dataclass directly — always runs."""

    def test_repr_contains_label(self):
        r = ClassificationResult(
            text="test", predicted_label="spor", confidence=0.95
        )
        assert "spor" in repr(r)
        assert "0.950" in repr(r)

    def test_all_scores_default_empty(self):
        r = ClassificationResult(text="x", predicted_label="y", confidence=0.5)
        assert r.all_scores == {}

    def test_confidence_stored_correctly(self):
        r = ClassificationResult(text="x", predicted_label="y", confidence=0.87)
        assert r.confidence == pytest.approx(0.87)


@requires_deps
class TestTurkishTextClassifierUnit:
    """Mock-based tests — no model download."""

    def _make(self, labels=None):
        return TurkishTextClassifier(labels=labels or SAMPLE_LABELS)

    def test_is_loaded_false_before_load(self):
        clf = self._make()
        assert clf.is_loaded() is False

    def test_classify_raises_before_load(self):
        clf = self._make()
        with pytest.raises(RuntimeError, match="load()"):
            clf.classify("test metin")

    def test_classify_batch_raises_before_load(self):
        clf = self._make()
        with pytest.raises(RuntimeError, match="load()"):
            clf.classify_batch(["test"])

    def test_empty_labels_raises(self):
        clf = TurkishTextClassifier(labels=[])
        clf._pipe = MagicMock()  # simulate loaded
        with pytest.raises(ValueError, match="labels"):
            clf.classify("test", labels=[])

    def test_preprocess_strips_whitespace(self):
        result = TurkishTextClassifier.preprocess("  merhaba   dunya  ")
        assert result == "merhaba dunya"

    def test_preprocess_truncates_long_text(self):
        long_text = "a" * 1000
        result = TurkishTextClassifier.preprocess(long_text)
        assert len(result) <= 512

    def test_preprocess_empty_string(self):
        result = TurkishTextClassifier.preprocess("")
        assert result == ""

    def test_mode_default_is_zero_shot(self):
        clf = self._make()
        assert clf.mode == "zero_shot"

    def test_device_default_cpu(self):
        clf = self._make()
        assert clf.device == -1

    def test_classify_with_mocked_pipe(self):
        clf = self._make()
        mock_output = {
            "labels": ["spor", "ekonomi"],
            "scores": [0.92, 0.08],
        }
        clf._pipe = MagicMock(return_value=mock_output)
        result = clf.classify("Fenerbahce mac kazandi.")
        assert result.predicted_label == "spor"
        assert result.confidence == pytest.approx(0.92)
        assert result.all_scores["spor"] == pytest.approx(0.92)

    def test_classify_batch_with_mocked_pipe(self):
        clf = self._make()
        mock_output = {
            "labels": ["teknoloji", "spor"],
            "scores": [0.88, 0.12],
        }
        clf._pipe = MagicMock(return_value=mock_output)
        results = clf.classify_batch(SAMPLE_TEXTS[:2])
        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_is_loaded_true_after_mock_load(self):
        clf = self._make()
        clf._pipe = MagicMock()
        assert clf.is_loaded() is True
