#!/usr/bin/env python3
"""
Quick Test für die verbesserte Intent Detection
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.intent_detector_improved import intent_detector, IntentType


def test_problematic_cases():
    """Teste die ursprünglich problematischen Fälle"""

    print("🔬 TESTING IMPROVED INTENT DETECTION")
    print("=" * 50)

    test_cases = [
        ("Schreibe eine Geschichte über einen Roboter", IntentType.CREATIVE_WRITING),
        ("Fasse diesen Artikel zusammen", IntentType.SUMMARIZATION),
        ("Erstelle eine Python-Funktion zur Sortierung", IntentType.CODE_GENERATION),
        ("Wie kann ich meine Website optimieren?", IntentType.INFORMATION_SEARCH),
        ("Analysiere das Sentiment in diesem Text", IntentType.SENTIMENT_ANALYSIS),
        ("Übersetze diesen Text ins Englische", IntentType.TRANSLATION),
    ]

    success_count = 0
    total_count = len(test_cases)

    for i, (query, expected) in enumerate(test_cases, 1):
        detected_intent, confidence = intent_detector.detect_intent(query)
        is_correct = detected_intent == expected

        if is_correct:
            success_count += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{status} Test {i}: '{query}'")
        print(f"    Expected: {expected.value}")
        print(f"    Detected: {detected_intent.value} (confidence: {confidence:.3f})")

        if not is_correct:
            debug_info = intent_detector.get_intent_confidence_report(query)
            print(f"    All scores: {debug_info}")
        print()

    accuracy = success_count / total_count
    print(
        f"📊 RESULTS: {success_count}/{total_count} correct ({accuracy:.1%} accuracy)"
    )

    if accuracy >= 0.8:
        print("🎉 Intent Detection working well!")
    else:
        print("⚠️  Still needs improvement")


if __name__ == "__main__":
    test_problematic_cases()
