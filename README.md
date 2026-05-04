# Turkish NLP Text Classifier

> **Multi-class Turkish text classification with HuggingFace Transformers**  
> Zero-shot inference · Fine-tuning ready · Batch prediction · sklearn metrics

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## Overview

This project provides a production-ready Turkish text classification pipeline using **multilingual BERT-based models** from HuggingFace. It supports:

- **Zero-shot classification** — classify text into arbitrary categories with no training data
- **Pretrained model inference** — load any fine-tuned `AutoModelForSequenceClassification`
- **Batch inference** with progress reporting
- **Evaluation** with `classification_report` + confusion matrix (sklearn-compatible)
- **Preprocessing utilities** for Turkish text normalization

---

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
```

### Zero-Shot Example

```python
from src.classifier import TurkishTextClassifier

clf = TurkishTextClassifier(
    labels=["spor", "ekonomi", "siyaset", "teknoloji", "saglik"]
)
clf.load()

result = clf.classify("Fenerbahce sezonu sampiyonlukla kapatti.")
print(result)
# ClassificationResult(label='spor', confidence=0.934)

print(result.all_scores)
# {'spor': 0.934, 'ekonomi': 0.031, 'siyaset': 0.019, ...}
```

### Batch Classification

```python
texts = [
    "Merkez bankasi faiz oranini yuzde 50 belirledi.",
    "Yeni yapay zeka modeli rekor kirdi.",
    "Secim kampanyasi hizla devam ediyor.",
]

results = clf.classify_batch(texts)
for r in results:
    print(f"{r.predicted_label:12} ({r.confidence:.2f}) — {r.text[:50]}")
```

### Expected Output

```
ekonomi      (0.91) — Merkez bankasi faiz oranini yuzde 50 belirledi.
teknoloji    (0.88) — Yeni yapay zeka modeli rekor kirdi.
siyaset      (0.79) — Secim kampanyasi hizla devam ediyor.
```

---

## Evaluation with sklearn Metrics

```python
true_labels = ["spor", "ekonomi", "siyaset", "teknoloji"]
texts = [...]  # same length

report = clf.evaluate(texts, true_labels)
print(f"Accuracy: {report['accuracy']:.2%}")
print(report["classification_report"])
```

### Sample Report

```
              precision    recall  f1-score   support

     ekonomi       0.91      0.94      0.92        50
    siyaset       0.78      0.82      0.80        50
        spor       0.96      0.93      0.94        50
   teknoloji       0.87      0.85      0.86        50

    accuracy                           0.89       200
   macro avg       0.88      0.89      0.88       200
weighted avg       0.88      0.89      0.88       200
```

---

## Project Structure

```
Turkish-NLP-Text-Classifier/
├── src/
│   ├── __init__.py
│   └── classifier.py          # Core classifier + ClassificationResult
├── tests/
│   └── test_classifier.py     # 12 unit tests (pytest + mocking)
├── demo.py                    # Quick demo script
├── requirements.txt
└── README.md
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Results

```
tests/test_classifier.py::TestClassificationResult::test_repr_contains_label           PASSED
tests/test_classifier.py::TestClassificationResult::test_all_scores_default_empty      PASSED
tests/test_classifier.py::TestClassificationResult::test_confidence_stored_correctly   PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_is_loaded_false_before_load PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_classify_raises_before_load PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_empty_labels_raises      PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_preprocess_strips_whitespace PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_preprocess_truncates_long_text PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_classify_with_mocked_pipe PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_classify_batch_with_mocked_pipe PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_is_loaded_true_after_mock_load PASSED

11 passed in 0.38s
```

---

## Supported Models

| Model | Size | Languages | Use Case |
|---|---|---|---|
| `joeddav/xlm-roberta-large-xnli` | 1.1GB | 100+ (incl. Turkish) | Zero-shot (default) |
| `savasy/bert-base-turkish-sentiment-cased` | 440MB | Turkish | Sentiment |
| `dbmdz/bert-base-turkish-cased` | 440MB | Turkish | Fine-tuning base |
| `google/flan-t5-base` | 250MB | Multi | Lightweight CPU |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `ClassificationResult` dataclass | Typed, serializable, repr-friendly output |
| Mock-based tests | CI runs in seconds, no 400MB model download |
| `preprocess()` static method | Reusable, testable, BERT token-limit safe |
| `multi_label=False` in zero-shot | Ensures mutually exclusive categories |
| Sklearn metrics | Industry-standard evaluation familiar to all ML engineers |

---

## Related Projects

- [Turkish-RAG-Assistant](https://github.com/Umitsencer/Turkish-RAG-Assistant) — Document QA with RAG
- [Finansal-Haber-NLP](https://github.com/Umitsencer/Finansal-Haber-NLP) — FinBERT trading signals
- [Smart-OCR-Extractor](https://github.com/Umitsencer/Smart-OCR-Extractor) — OCR for Turkish documents

---

## Author

**Umit Sencer** — Software Engineering Student, Kirklareli University  
TUBiTAK 2209-A Researcher | AI/ML | NLP | LLM

[GitHub](https://github.com/Umitsencer) · [LinkedIn](https://linkedin.com/in/umitsencer)
