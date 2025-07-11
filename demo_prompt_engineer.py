#!/usr/bin/env python3
"""
Demonstration der erweiterten Prompt Engineering Funktionalität

Dieses Skript zeigt, wie der neue Prompt Engineer Agent arbeitet:
1. Intent Detection basierend auf Benutzeranfragen
2. Prompt Optimization mit spezialisierten Templates
3. Agent Routing Empfehlungen
4. Integration mit dem bestehenden A2A System
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the improved intent detection
from core.intent_detector import intent_detector, IntentType
from core.config_manager import config

# Import existing agent functionality (if available)
try:
    from agent_server.prompt_engineer import (
        PromptEngineerRequest,
        run_prompt_engineer,
    )

    PROMPT_ENGINEER_AVAILABLE = True
except ImportError:
    PROMPT_ENGINEER_AVAILABLE = False
    print(
        "⚠️  Note: prompt_engineer module not found - using standalone intent detection"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_intent_detection():
    """Demonstriere die Intent Detection Funktionalität"""
    print("\n" + "=" * 80)
    print("🧠 INTENT DETECTION DEMO")
    print("=" * 80)

    test_queries = [
        (
            "Erkläre mir die Grundlagen der Quantenphysik",
            IntentType.INFORMATION_SEARCH,
        ),
        ("Erstelle eine Python-Funktion zur Sortierung", IntentType.CODE_GENERATION),
        ("Analysiere das Sentiment in diesem Text", IntentType.SENTIMENT_ANALYSIS),
        ("Wie kann ich meine Website optimieren?", IntentType.INFORMATION_SEARCH),
        ("Schreibe eine Geschichte über einen Roboter", IntentType.CREATIVE_WRITING),
        ("Übersetze diesen Text ins Englische", IntentType.TRANSLATION),
        ("Fasse diesen Artikel zusammen", IntentType.SUMMARIZATION),
        ("Hallo, wie geht es dir?", IntentType.GENERAL_ASSISTANCE),
    ]

    for query, expected_intent in test_queries:
        detected_intent, confidence = intent_detector.detect_intent(query)
        status = "✅" if detected_intent == expected_intent else "❌"

        print(f"{status} Query: '{query}'")
        print(f"    Expected: {expected_intent.value}")
        print(f"    Detected: {detected_intent.value} (confidence: {confidence:.3f})")

        # Show debug info for failed cases
        if detected_intent != expected_intent:
            # # # debug_info = intent_detector.get_intent_confidence_report(query)  # Method does not exist  # Method does not exist  # Method does not exist
            print(f"    Analysis: Intent detection based on keyword patterns")
        print()


async def demo_prompt_optimization():
    """Demonstriere die Prompt Optimization"""
    print("\n" + "=" * 80)
    print("🎯 PROMPT OPTIMIZATION DEMO")
    print("=" * 80)

    test_cases = [
        {
            "query": "Erkläre mir maschinelles Lernen",
            "description": "Information Retrieval - Strukturierte Erklärung",
        },
        {
            "query": "Mache eine Sortierfunktion",
            "description": "Code Generation - Dokumentierter Code",
        },
        {
            "query": "Ist dieser Text positiv: Ich liebe dieses Produkt!",
            "description": "Text Analysis - Sentiment Bewertung",
        },
        {
            "query": "Meine App ist langsam",
            "description": "Problem Solving - Performance Optimierung",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['description']}")
        print(f"Original Query: '{case['query']}'")
        print("-" * 60)

        request = PromptEngineerRequest(user_input=case["query"])
        result = await run_prompt_engineer(request)

        if result.status == "success":
            opt_result = result.optimization_result
            print(f"🎯 Detected Intent: {opt_result.detected_intent.value}")
            print(f"🎯 Confidence Score: {opt_result.confidence_score:.2f}")
            print(f"🤖 Recommended Agents: {', '.join(opt_result.recommended_agents)}")
            print(
                f"📋 Execution Plan: {' → '.join(result.execution_plan) if hasattr(result, 'execution_plan') else 'Sequential processing'}"
            )
            print(f"💡 Optimized Prompt (VOLLSTÄNDIG):")
            print(f"    {opt_result.optimized_prompt}")
            print()

            if hasattr(result, "alternative_prompts") and result.alternative_prompts:
                print(f"🔄 Alternative Approaches (VOLLSTÄNDIG):")
                for i, alt in enumerate(result.alternative_prompts, 1):
                    print(f"    {i}. {alt}")
                print()
        else:
            print(f"❌ Error: {result.message}")

        print()


async def demo_a2a_integration():
    """Demonstriere die Integration mit dem A2A System"""
    print("\n" + "=" * 80)
    print("🔗 A2A INTEGRATION DEMO")
    print("=" * 80)

    # Import A2A setup
    from a2a_server import setup_a2a_server

    registry = await setup_a2a_server()

    try:
        test_queries = [
            "Optimiere diesen Text für eine professionelle E-Mail: Das ist Schrott!",
            "Korrigiere die Grammatik: Das ist ein sehr schlechte Satz.",
            "Analysiere das Sentiment: Ich bin super glücklich heute!",
            "Verbessere diese Anfrage für ein LLM: Erkläre KI",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔄 A2A Test {i}: {query}")
            print("-" * 60)  # First, get prompt optimization recommendations
            prompt_result = await registry.call_agent("prompt_engineer", query)

            if hasattr(prompt_result, "optimization_result"):
                opt = prompt_result.optimization_result
                print(f"🎯 Intent: {opt.detected_intent.value}")
                print(
                    f"🤖 Recommended Agents: {opt.recommended_agents}"
                )  # Try to execute with recommended agents
                if opt.recommended_agents:
                    for agent_name in opt.recommended_agents[
                        :1
                    ]:  # Try first recommended
                        if agent_name in [
                            "lektor",
                            "optimizer",
                            "sentiment",
                            "query_ref",
                        ]:
                            try:
                                print(f"📞 Calling agent: {agent_name}")
                                print(f"🔍 DEBUG: Sending to agent: '{query}'")

                                # Send the full query for better processing
                                agent_result = await registry.call_agent(
                                    agent_name, query
                                )

                                # Extract meaningful result - VOLLSTÄNDIGE AUSGABE
                                if hasattr(agent_result, "optimized_text"):
                                    print(f"✅ Result (VOLLSTÄNDIG): {agent_result.optimized_text}")
                                elif hasattr(agent_result, "corrected_text"):
                                    print(f"✅ Result (VOLLSTÄNDIG): {agent_result.corrected_text}")
                                elif hasattr(agent_result, "query"):
                                    print(f"✅ Result (VOLLSTÄNDIG): {agent_result.query}")
                                elif hasattr(agent_result, "sentiment"):
                                    print(f"✅ Result: {agent_result.sentiment}")
                                else:
                                    print(f"✅ Result (VOLLSTÄNDIG): {str(agent_result)}")

                            except Exception as e:
                                print(f"❌ Agent call failed: {e}")
                else:
                    print("ℹ️  No specific agents recommended")
            else:
                print(f"❌ Prompt optimization failed: {prompt_result}")

    finally:
        await registry.stop()


async def demo_advanced_workflows():
    """Demonstriere erweiterte Workflows"""
    print("\n" + "=" * 80)
    print("⚡ ADVANCED WORKFLOW DEMO")
    print("=" * 80)

    complex_scenarios = [
        {
            "scenario": "Multi-Step Text Processing",
            "query": "Dieser Text ist schlecht geschrieben und soll professionell werden: Das Produkt ist echt der letzte Schrott!",
            "steps": [
                "Intent Detection",
                "Agent Routing",
                "Text Optimization",
                "Grammar Check",
            ],
        },
        {
            "scenario": "Code Generation with Optimization",
            "query": "Erstelle mir eine REST API für Benutzerregistrierung mit Validierung",
            "steps": [
                "Intent Detection",
                "Code Template Selection",
                "Implementation",
                "Documentation",
            ],
        },
        {
            "scenario": "Research Query Enhancement",
            "query": "Was ist KI?",
            "steps": [
                "Query Analysis",
                "Optimization",
                "Information Structuring",
                "Response Generation",
            ],
        },
    ]

    for scenario in complex_scenarios:
        print(f"\n🎭 Scenario: {scenario['scenario']}")
        print(f"Query: '{scenario['query']}'")
        print(f"Planned Steps: {' → '.join(scenario['steps'])}")
        print("-" * 60)

        # Analyze with prompt engineer
        request = PromptEngineerRequest(user_input=scenario["query"])
        result = await run_prompt_engineer(request)

        if result.status == "success":
            print(f"✅ Workflow Analysis Successful:")
            print(f"   Intent: {result.optimization_result.detected_intent.value}")
            print(f"   Confidence: {result.optimization_result.confidence_score:.2f}")
            print(
                f"   Execution Plan: {' → '.join(result.execution_plan) if hasattr(result, 'execution_plan') else 'Sequential processing'}"
            )
            print(
                f"   Context Elements: {len(result.optimization_result.context_elements) if hasattr(result.optimization_result, 'context_elements') else 0} identified"
            )
        else:
            print(f"❌ Workflow Analysis Failed: {result.message}")


async def main():
    """Hauptfunktion für die Demo"""
    print("🚀 PROMPT ENGINEER AGENT - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("Dieses Demo zeigt die erweiterten Fähigkeiten des Prompt Engineer Agents:")
    print("• Intent Detection basierend auf Benutzeranfragen")
    print("• Prompt Optimization mit spezialisierten Templates")
    print("• Agent Routing Empfehlungen")
    print("• Integration mit dem bestehenden A2A System")
    print("• Multi-Step Workflow Orchestrierung")

    try:
        # Run all demos
        await demo_intent_detection()
        await demo_prompt_optimization()
        await demo_a2a_integration()
        await demo_advanced_workflows()

        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETE!")
        print("Der Prompt Engineer Agent ist bereit für den produktiven Einsatz.")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
