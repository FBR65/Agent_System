#!/usr/bin/env python3
"""
Test Script für die verbesserte Intent Detection
Direkte Tests ohne andere Abhängigkeiten
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.intent_detector import intent_detector, IntentType


async def test_problematic_cases():
    """Teste die ursprünglich problematischen Fälle"""

    print("🧪 TESTING IMPROVED INTENT DETECTION")
    print("=" * 50)

    test_cases = [
        {
            "query": "Schreibe eine Geschichte über einen Roboter",
            "expected": IntentType.CREATIVE_WRITING,
            "previous": "code_generation",
        },
        {
            "query": "Fasse diesen Artikel zusammen",
            "expected": IntentType.SUMMARIZATION,
            "previous": "creative_writing",
        },
        {
            "query": "Erstelle eine Python-Funktion zur Sortierung",
            "expected": IntentType.CODE_GENERATION,
            "previous": "should work",
        },
        {
            "query": "Korrigiere diesen Text: Das ist ein sehr schlechte Satz",
            "expected": IntentType.GRAMMAR_CORRECTION,
            "previous": "should work",
        },
        {
            "query": "Mache diesen Text professioneller",
            "expected": IntentType.TEXT_OPTIMIZATION,
            "previous": "should work",
        },
    ]

    success_count = 0
    total_count = len(test_cases)

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = case["expected"]
        previous = case["previous"]

        detected_intent, confidence = intent_detector.detect_intent(query)
        is_correct = detected_intent == expected

        if is_correct:
            success_count += 1
            status = "✅ FIXED" if "should work" not in previous else "✅ GOOD"
            print(f"{status} Test {i}")
        else:
            status = "❌ STILL BROKEN"
            print(f"{status} Test {i}")

        print(f"  Query: '{query}'")
        print(f"  Expected: {expected.value}")
        print(f"  Detected: {detected_intent.value} (confidence: {confidence:.3f})")

        if previous != "should work":
            print(f"  Previously: {previous}")

        if not is_correct:
            # Show debug info for failed cases
            debug_info = intent_detector.get_intent_confidence_report(query)
            print(f"  Debug scores: {debug_info}")

        print()

    accuracy = success_count / total_count
    print(
        f"📊 RESULTS: {success_count}/{total_count} correct ({accuracy:.1%} accuracy)"
    )

    if accuracy >= 0.8:
        print("🎉 Intent Detection significantly improved!")
    elif accuracy >= 0.6:
        print("⚡ Some improvement, but needs more work")
    else:
        print("❌ Intent Detection still needs work")

    return accuracy


async def test_all_intents():
    """Teste verschiedene Intent-Typen"""

    print("\n🎯 COMPREHENSIVE INTENT TESTING")
    print("=" * 50)

    test_cases = [
        # Creative Writing
        ("Schreibe eine Geschichte über Freundschaft", IntentType.CREATIVE_WRITING),
        ("Erzähle mir ein Märchen", IntentType.CREATIVE_WRITING),
        ("Erfinde eine Kurzgeschichte", IntentType.CREATIVE_WRITING),
        # Summarization
        ("Fasse den Artikel kurz zusammen", IntentType.SUMMARIZATION),
        ("Erstelle eine Zusammenfassung", IntentType.SUMMARIZATION),
        ("Gib mir einen Überblick", IntentType.SUMMARIZATION),
        # Code Generation
        ("Schreibe Python Code für eine API", IntentType.CODE_GENERATION),
        ("Programmiere eine Sortierfunktion", IntentType.CODE_GENERATION),
        ("Erstelle eine Klasse für Benutzer", IntentType.CODE_GENERATION),
        # Grammar Correction
        ("Korrigiere die Rechtschreibung", IntentType.GRAMMAR_CORRECTION),
        ("Prüfe den Text auf Fehler", IntentType.GRAMMAR_CORRECTION),
        ("Verbessere die Grammatik", IntentType.GRAMMAR_CORRECTION),
        # Text Optimization
        ("Mache den Text professioneller", IntentType.TEXT_OPTIMIZATION),
        ("Optimiere für eine E-Mail", IntentType.TEXT_OPTIMIZATION),
        ("Verbessere den Ton", IntentType.TEXT_OPTIMIZATION),
        # Sentiment Analysis
        ("Analysiere das Sentiment", IntentType.SENTIMENT_ANALYSIS),
        ("Wie ist die Stimmung?", IntentType.SENTIMENT_ANALYSIS),
        ("Bewerte die Emotionen", IntentType.SENTIMENT_ANALYSIS),
    ]

    correct = 0
    for query, expected in test_cases:
        detected, confidence = intent_detector.detect_intent(query)
        is_correct = detected == expected
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} '{query}' → {detected.value} ({confidence:.2f})")

    accuracy = correct / len(test_cases)
    print(f"\n📊 Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1%})")


async def main():
    """Hauptfunktion für den Intent Detection Test"""
    print("🔬 INTENT DETECTION VALIDATION")
    print("Testing the improved intent detection system")
    print()

    # Test the problematic cases first
    await test_problematic_cases()

    # Test comprehensive intent coverage
    await test_all_intents()

    print("\n" + "=" * 50)
    print("✅ INTENT DETECTION TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
