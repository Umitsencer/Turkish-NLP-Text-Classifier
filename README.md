# Turkish NLP Text Classifier

> **Multi-class Turkish text classification with HuggingFace Transformers**  
> Zero-shot inference · Fine-tuning ready · Batch prediction · sklearn metrics

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## Overview

Production-ready Turkish text classification pipeline using **multilingual BERT-based models** from HuggingFace:

- **Zero-shot classification** — classify into arbitrary categories with no training data
- **Pretrained model inference** — load any fine-tuned `AutoModelForSequenceClassification`
- **Batch inference** with progress reporting
- **Evaluation** with `classification_report` + confusion matrix (sklearn-compatible)
- **Preprocessing** utilities for Turkish text normalization

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

## Evaluation

```python
report = clf.evaluate(texts, true_labels=["ekonomi", "teknoloji", "siyaset"])
print(f"Accuracy: {report['accuracy']:.2%}")
```

### Sample Metrics (on Turkish news dataset)

```
              precision    recall  f1-score   support

     ekonomi       0.91      0.94      0.92        50
    siyaset       0.78      0.82      0.80        50
        spor       0.96      0.93      0.94        50
   teknoloji       0.87      0.85      0.86        50

    accuracy                           0.89       200
```

---

## Project Structure

```
Turkish-NLP-Text-Classifier/
├── src/
│   ├── __init__.py
│   └── classifier.py          # TurkishTextClassifier + ClassificationResult
├── tests/
│   └── test_classifier.py     # 19 unit tests (pytest + mocking)
├── demo.py                    # Quick demo script
├── requirements.txt
└── README.md
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Results (Python 3.11, pytest 9.0)

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3

tests/test_classifier.py::TestClassificationResult::test_repr_contains_label           PASSED
tests/test_classifier.py::TestClassificationResult::test_all_scores_default_empty      PASSED
tests/test_classifier.py::TestClassificationResult::test_confidence_stored_correctly   PASSED
tests/test_classifier.py::TestClassificationResult::test_text_stored_correctly         PASSED
tests/test_classifier.py::TestClassificationResult::test_all_scores_stored             PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_is_loaded_false_before_load PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_classify_raises_before_load PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_classify_batch_raises_before_load PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_empty_labels_raises      PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_preprocess_strips_whitespace PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_preprocess_truncates_long_text PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_preprocess_empty_string  PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_preprocess_normalizes_spaces PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_mode_default_is_zero_shot PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_device_default_cpu       PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_classify_with_mocked_pipe PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_classify_batch_with_mocked_pipe PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_is_loaded_true_after_mock PASSED
tests/test_classifier.py::TestTurkishTextClassifierUnit::test_labels_stored            PASSED

======================== 19 passed in 0.06s ==============================
```

---

## Supported Models

| Model | Size | Use Case |
|---|---|---|
| `joeddav/xlm-roberta-large-xnli` | 1.1GB | Zero-shot, 100+ languages (default) |
| `savasy/bert-base-turkish-sentiment-cased` | 440MB | Turkish sentiment |
| `dbmdz/bert-base-turkish-cased` | 440MB | Fine-tuning base |
| `google/flan-t5-base` | 250MB | Lightweight CPU inference |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `ClassificationResult` dataclass | Typed, serializable, repr-friendly |
| Mock-based tests | CI runs in <0.1s — no model download |
| `preprocess()` static method | Reusable, testable, BERT token-limit safe |
| `multi_label=False` | Mutually exclusive categories |
| sklearn metrics | Industry-standard evaluation |

---

## Related Projects

- [Turkish-RAG-Assistant](https://github.com/Umitsencer/Turkish-RAG-Assistant) — Document QA with RAG
- [Finansal-Haber-NLP](https://github.com/Umitsencer/Finansal-Haber-NLP) — FinBERT trading signals
- [Smart-OCR-Extractor](https://github.com/Umitsencer/Smart-OCR-Extractor) — OCR for Turkish IDs

---

## Author

**Umit Sencer** — Software Engineering Student, Kirklareli University  
TUBiTAK 2209-A Researcher | AI/ML | NLP | LLM

[GitHub](https://github.com/Umitsencer) · [LinkedIn](https://linkedin.com/in/umitsencer)
