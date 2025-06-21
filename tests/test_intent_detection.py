#!/usr/bin/env python3
"""
Test Script für Enhanced Intent Detection
Testet die Genauigkeit der Intent-Erkennung mit verschiedenen Beispielen
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent_detector import intent_detector, IntentType


def test_intent_detection():
    """Teste Intent Detection mit problematischen Beispielen"""

    test_cases = [
        # Kreative Schreibaufgaben
        {
            "query": "Schreibe eine Geschichte über einen Roboter",
            "expected": IntentType.CREATIVE_WRITING,
            "description": "Klare Geschichte-Anfrage",
        },
        {
            "query": "Erzähle mir eine spannende Geschichte",
            "expected": IntentType.CREATIVE_WRITING,
            "description": "Story-Anfrage",
        },
        {
            "query": "Erfinde eine kurze Geschichte über Freundschaft",
            "expected": IntentType.CREATIVE_WRITING,
            "description": "Kreative Geschichte",
        },
        # Zusammenfassungen
        {
            "query": "Fasse diesen Artikel zusammen",
            "expected": IntentType.SUMMARIZATION,
            "description": "Artikel zusammenfassen",
        },
        {
            "query": "Erstelle eine kurze Zusammenfassung",
            "expected": IntentType.SUMMARIZATION,
            "description": "Zusammenfassung erstellen",
        },
        {
            "query": "Gib mir einen Überblick über den Text",
            "expected": IntentType.SUMMARIZATION,
            "description": "Textüberblick",
        },
        # Code-Generierung
        {
            "query": "Schreibe Python Code für eine Funktion",
            "expected": IntentType.CODE_GENERATION,
            "description": "Code schreiben",
        },
        {
            "query": "Programmiere mir einen Algorithmus",
            "expected": IntentType.CODE_GENERATION,
            "description": "Algorithmus programmieren",
        },
        # Grammatikkorrektur
        {
            "query": "Korrigiere diesen Text: Das ist ein sehr schlechte Satz",
            "expected": IntentType.GRAMMAR_CORRECTION,
            "description": "Text korrigieren",
        },
        {
            "query": "Prüfe den Text auf Rechtschreibfehler",
            "expected": IntentType.GRAMMAR_CORRECTION,
            "description": "Rechtschreibprüfung",
        },
        # Text-Optimierung
        {
            "query": "Mache diesen Text professioneller",
            "expected": IntentType.TEXT_OPTIMIZATION,
            "description": "Text professionalisieren",
        },
        {
            "query": "Optimiere den Text für eine E-Mail",
            "expected": IntentType.TEXT_OPTIMIZATION,
            "description": "Email-Optimierung",
        },
        # Sentiment-Analyse
        {
            "query": "Analysiere das Sentiment dieses Textes",
            "expected": IntentType.SENTIMENT_ANALYSIS,
            "description": "Sentiment analysieren",
        },
        {
            "query": "Wie ist die Stimmung in diesem Text?",
            "expected": IntentType.SENTIMENT_ANALYSIS,
            "description": "Stimmung bewerten",
        },
    ]

    print("🧪 Testing Enhanced Intent Detection")
    print("=" * 50)

    correct = 0
    total = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        description = test_case["description"]

        detected_intent, confidence = intent_detector.detect_intent(query)

        is_correct = detected_intent == expected
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{status} Test {i:2d}: {description}")
        print(f"    Query: '{query}'")
        print(f"    Expected: {expected.value}")
        print(f"    Detected: {detected_intent.value} (confidence: {confidence:.3f})")

        if not is_correct:
            # Zeige detaillierte Scores für falsche Erkennungen
            debug_info = intent_detector.get_intent_confidence_report(query)
            print(f"    Debug scores: {debug_info}")

        print()

    accuracy = correct / total
    print(f"📊 Results: {correct}/{total} correct ({accuracy:.1%} accuracy)")

    if accuracy < 0.8:
        print("⚠️  Accuracy below 80% - Intent detection needs improvement")
    elif accuracy < 0.9:
        print("⚡ Good accuracy - Minor improvements possible")
    else:
        print("🎉 Excellent accuracy!")

    return accuracy


def test_specific_problematic_cases():
    """Teste die ursprünglich problematischen Fälle"""

    print("\n🎯 Testing Previously Problematic Cases")
    print("=" * 40)

    problematic_cases = [
        {
            "query": "Schreibe eine Geschichte über einen Roboter",
            "expected": "creative_writing",
            "old_detected": "code_generation",
        },
        {
            "query": "Fasse diesen Artikel zusammen",
            "expected": "summarization",
            "old_detected": "creative_writing",
        },
    ]

    for case in problematic_cases:
        query = case["query"]
        expected = case["expected"]
        old_detected = case["old_detected"]

        detected_intent, confidence = intent_detector.detect_intent(query)
        detected_str = detected_intent.value

        is_fixed = detected_str == expected
        status = "✅ FIXED" if is_fixed else "❌ STILL BROKEN"

        print(f"{status}")
        print(f"  Query: '{query}'")
        print(f"  Expected: {expected}")
        print(f"  Previously: {old_detected}")
        print(f"  Now: {detected_str} (confidence: {confidence:.3f})")
        print()


if __name__ == "__main__":
    test_intent_detection()
    test_specific_problematic_cases()
