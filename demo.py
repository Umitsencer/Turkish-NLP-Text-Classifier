"""Quick demo — zero-shot classify Turkish news headlines."""
from src.classifier import TurkishTextClassifier

LABELS = ["spor", "ekonomi", "siyaset", "teknoloji", "saglik"]

HEADLINES = [
    "Fenerbahce sezonu sampiyonlukla kapatti.",
    "Merkez bankasi faiz oranini yuzde 50 olarak belirledi.",
    "Secim kampanyasi tum hizyla devam ediyor.",
    "Yeni yapay zeka modeli dil anlama testinde rekor kirdi.",
    "Saglık Bakanligi yeni asi takvimine iliskin aciklama yapti.",
]

TRUE_LABELS = ["spor", "ekonomi", "siyaset", "teknoloji", "saglik"]


def main():
    print("Turkish NLP Text Classifier — Demo\n" + "=" * 45)
    clf = TurkishTextClassifier(labels=LABELS)
    clf.load()

    print("\n[1] Tek Metin Siniflandirma:")
    result = clf.classify(HEADLINES[0])
    print(f"  Metin     : {HEADLINES[0]}")
    print(f"  Tahmin    : {result.predicted_label}")
    print(f"  Guven     : {result.confidence:.3f}")
    print(f"  Tum skorlar: {result.all_scores}")

    print("\n[2] Toplu Siniflandirma:")
    results = clf.classify_batch(HEADLINES)
    for r in results:
        print(f"  {r.predicted_label:12} ({r.confidence:.2f}) — {r.text[:55]}")

    print("\n[3] Degerlendirme Raporu:")
    report = clf.evaluate(HEADLINES, TRUE_LABELS)
    print(f"  Accuracy: {report['accuracy']:.2%}")


if __name__ == "__main__":
    main()
