#!/usr/bin/env python3
"""
Vereinfachte Demo für den Prompt Engineer Agent
Nur Intent Detection und Prompt Optimization ohne pydantic_ai
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.intent_detector_improved import intent_detector, IntentType


@dataclass
class SimplePromptResult:
    """Vereinfachtes Ergebnis der Prompt-Optimierung"""

    original_query: str
    detected_intent: IntentType
    confidence_score: float
    optimized_prompt: str
    recommended_agents: List[str]
    execution_plan: List[str]
    reasoning: str


class SimplePromptEngineer:
    """Vereinfachter Prompt Engineer ohne externe Abhängigkeiten"""

    def __init__(self):
        self.agent_mappings = {
            IntentType.TEXT_OPTIMIZATION: ["optimizer", "lektor"],
            IntentType.GRAMMAR_CORRECTION: ["lektor"],
            IntentType.SENTIMENT_ANALYSIS: ["sentiment"],
            IntentType.CREATIVE_WRITING: ["optimizer"],
            IntentType.CODE_GENERATION: ["query_ref"],
            IntentType.SUMMARIZATION: ["optimizer"],
            IntentType.TRANSLATION: ["optimizer"],
            IntentType.INFORMATION_SEARCH: ["query_ref"],
            IntentType.GENERAL_ASSISTANCE: ["user_interface"],
        }

        self.templates = {
            IntentType.CREATIVE_WRITING: """
Du bist ein kreativer Schriftsteller. Erstelle eine fesselnde Geschichte zu folgendem Thema:

Thema: {topic}

Achte auf:
- Spannenden Aufbau
- Lebendige Charaktere
- Emotionale Tiefe
- Ansprechende Sprache

Beginne mit einem packenden ersten Satz.
""",
            IntentType.CODE_GENERATION: """
Du bist ein erfahrener Softwareentwickler. Erstelle sauberen, dokumentierten Code für:

Anforderung: {task}

Der Code soll:
- Funktional und getestet sein
- Best Practices befolgen
- Gut kommentiert sein
- Fehlerbehandlung enthalten

Füge Beispiele für die Verwendung hinzu.
""",
            IntentType.SUMMARIZATION: """
Du bist ein Experte für Textanalyse. Erstelle eine prägnante Zusammenfassung von:

Text: {content}

Die Zusammenfassung soll:
- Die Kernaussagen erfassen
- Strukturiert und logisch sein
- Etwa 20% der ursprünglichen Länge haben
- Die wichtigsten Details bewahren
""",
            IntentType.SENTIMENT_ANALYSIS: """
Du bist ein Experte für Emotionsanalyse. Analysiere das Sentiment in folgendem Text:

Text: {text}

Bewerte:
- Gesamtstimmung (positiv/neutral/negativ)
- Emotionale Intensität (0-1)
- Hauptemotionen
- Begründung der Bewertung

Gib eine strukturierte Analyse.
""",
            IntentType.TEXT_OPTIMIZATION: """
Du bist ein Kommunikationsexperte. Optimiere folgenden Text:

Originaltext: {original}

Ziel: Professioneller, klarer und wirkungsvoller

Achte auf:
- Angemessenen Ton
- Klare Struktur
- Präzise Formulierung
- Zielgruppengerechte Sprache
""",
            IntentType.GRAMMAR_CORRECTION: """
Du bist ein Sprachexperte. Korrigiere alle Fehler in folgendem Text:

Text: {text}

Korrigiere:
- Rechtschreibung
- Grammatik
- Zeichensetzung
- Stilistische Verbesserungen

Erkläre die wichtigsten Korrekturen.
""",
        }

    async def optimize_prompt(self, user_query: str) -> SimplePromptResult:
        """Optimiere einen Prompt basierend auf der Benutzeranfrage"""

        # 1. Intent Detection
        detected_intent, confidence = intent_detector.detect_intent(user_query)

        # 2. Template Selection
        template = self.templates.get(
            detected_intent,
            "Du bist ein hilfreicher Assistent. Beantworte folgende Anfrage: {query}",
        )

        # 3. Prompt Generation
        optimized_prompt = self._generate_prompt(template, user_query, detected_intent)

        # 4. Agent Recommendations
        recommended_agents = self.agent_mappings.get(detected_intent, [])

        # 5. Execution Plan
        execution_plan = self._create_execution_plan(
            detected_intent, recommended_agents
        )

        # 6. Reasoning
        reasoning = (
            f"Intent '{detected_intent.value}' erkannt mit {confidence:.1%} Konfidenz. "
            f"Prompt optimiert für {detected_intent.value}-Aufgaben."
        )

        return SimplePromptResult(
            original_query=user_query,
            detected_intent=detected_intent,
            confidence_score=confidence,
            optimized_prompt=optimized_prompt,
            recommended_agents=recommended_agents,
            execution_plan=execution_plan,
            reasoning=reasoning,
        )

    def _generate_prompt(self, template: str, query: str, intent: IntentType) -> str:
        """Generiere optimierten Prompt"""

        # Template-Variablen für verschiedene Intents
        variables = {
            "topic": query,
            "task": query,
            "content": query,
            "text": query,
            "original": query,
            "query": query,
        }

        try:
            # Template mit Variablen füllen
            optimized_prompt = template.format(**variables)
        except KeyError:
            # Fallback wenn Template-Variablen fehlen
            optimized_prompt = f"Aufgabe: {query}\n\nBearbeite diese Anfrage professionell und detailliert."

        return optimized_prompt.strip()

    def _create_execution_plan(
        self, intent: IntentType, agents: List[str]
    ) -> List[str]:
        """Erstelle Ausführungsplan"""

        plans = {
            IntentType.CREATIVE_WRITING: [
                "Thema analysieren",
                "Struktur planen",
                "Geschichte schreiben",
            ],
            IntentType.CODE_GENERATION: [
                "Anforderungen verstehen",
                "Code entwickeln",
                "Testen",
            ],
            IntentType.SUMMARIZATION: [
                "Text analysieren",
                "Kernpunkte extrahieren",
                "Zusammenfassung erstellen",
            ],
            IntentType.SENTIMENT_ANALYSIS: [
                "Text segmentieren",
                "Emotionen erkennen",
                "Bewertung erstellen",
            ],
            IntentType.TEXT_OPTIMIZATION: [
                "Text analysieren",
                "Schwächen identifizieren",
                "Verbesserungen anwenden",
            ],
            IntentType.GRAMMAR_CORRECTION: [
                "Fehler finden",
                "Korrekturen anwenden",
                "Qualität prüfen",
            ],
        }

        plan = plans.get(
            intent, ["Anfrage analysieren", "Lösung entwickeln", "Ergebnis liefern"]
        )

        if agents:
            plan.append(f"A2A Routing zu: {', '.join(agents)}")

        return plan


async def demo_simple_prompt_engineer():
    """Demonstriere den vereinfachten Prompt Engineer"""

    print("🧠 SIMPLE PROMPT ENGINEER DEMO")
    print("=" * 50)

    engineer = SimplePromptEngineer()

    test_cases = [
        "Schreibe eine Geschichte über einen mutigen Roboter",
        "Fasse diesen Artikel über KI zusammen",
        "Erstelle eine Python-Funktion für Fibonacci-Zahlen",
        "Korrigiere: Das ist ein sehr schlechte Satz",
        "Analysiere das Sentiment: Ich liebe dieses Produkt!",
        "Mache professioneller: Das Zeug ist echt schlecht",
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)

        result = await engineer.optimize_prompt(query)

        print(f"🎯 Intent: {result.detected_intent.value}")
        print(f"📊 Confidence: {result.confidence_score:.3f}")
        print(
            f"🤖 Agents: {', '.join(result.recommended_agents) if result.recommended_agents else 'None'}"
        )
        print(f"📋 Plan: {' → '.join(result.execution_plan)}")
        print(f"💡 Optimized Prompt (preview):")
        print(f"    {result.optimized_prompt[:150]}...")
        print(f"🧠 Reasoning: {result.reasoning}")


async def compare_prompts():
    """Vergleiche Original vs. Optimierte Prompts"""

    print("\n\n🔀 PROMPT COMPARISON")
    print("=" * 40)

    engineer = SimplePromptEngineer()

    test_query = "Schreibe Code für Sortierung"

    print(f"Original Query: '{test_query}'")
    print("-" * 30)

    result = await engineer.optimize_prompt(test_query)

    print("📝 ORIGINAL:")
    print(f"   {test_query}")

    print("\n✨ OPTIMIZED:")
    print(f"   {result.optimized_prompt}")

    print(f"\n📈 IMPROVEMENTS:")
    print(f"   - Intent detection: {result.detected_intent.value}")
    print(f"   - Strukturierte Anweisungen")
    print(f"   - Spezifische Qualitätskriterien")
    print(f"   - Beispiel-Anforderung")


async def main():
    """Hauptfunktion"""

    print("🚀 PROMPT ENGINEER - SIMPLIFIED DEMO")
    print("=" * 60)
    print("Demonstriert Intent Detection und Prompt Optimization")
    print("ohne externe Abhängigkeiten wie pydantic_ai")

    await demo_simple_prompt_engineer()
    await compare_prompts()

    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("Der Prompt Engineer Agent funktioniert einwandfrei!")


if __name__ == "__main__":
    asyncio.run(main())
