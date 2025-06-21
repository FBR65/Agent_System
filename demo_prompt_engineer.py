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

from agent_server.prompt_engineer import (
    PromptEngineerRequest,
    run_prompt_engineer,
    IntentDetector,
    TaskIntention,
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
            TaskIntention.INFORMATION_RETRIEVAL,
        ),
        ("Erstelle eine Python-Funktion zur Sortierung", TaskIntention.CODE_GENERATION),
        ("Analysiere das Sentiment in diesem Text", TaskIntention.TEXT_ANALYSIS),
        ("Wie kann ich meine Website optimieren?", TaskIntention.PROBLEM_SOLVING),
        ("Schreibe eine Geschichte über einen Roboter", TaskIntention.CREATIVE_WRITING),
        ("Übersetze diesen Text ins Englische", TaskIntention.TRANSLATION),
        ("Fasse diesen Artikel zusammen", TaskIntention.SUMMARIZATION),
        ("Hallo, wie geht es dir?", TaskIntention.CONVERSATION),
    ]

    for query, expected_intent in test_queries:
        detected_intent = IntentDetector.detect_intention(query)
        status = "✅" if detected_intent == expected_intent else "❌"

        print(f"{status} Query: '{query}'")
        print(f"    Expected: {expected_intent.value}")
        print(f"    Detected: {detected_intent.value}")
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
            opt_prompt = result.optimized_prompt
            print(f"🎯 Detected Intent: {opt_prompt.detected_intention.value}")
            print(f"🎯 Confidence Score: {opt_prompt.confidence_score:.2f}")
            print(f"🤖 Recommended Agents: {', '.join(opt_prompt.recommended_agents)}")
            print(f"📋 Execution Plan: {' → '.join(result.execution_plan)}")
            print(f"💡 Optimized Prompt (first 200 chars):")
            print(f"    {opt_prompt.optimized_prompt[:200]}...")

            if result.alternative_prompts:
                print(f"🔄 Alternative Approaches:")
                for alt in result.alternative_prompts[:2]:  # Show first 2
                    print(f"    • {alt[:100]}...")
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
            print("-" * 60)

            # First, get prompt optimization recommendations
            prompt_result = await registry.call_agent("prompt_engineer", query)

            if hasattr(prompt_result, "optimized_prompt"):
                opt = prompt_result.optimized_prompt
                print(f"🎯 Intent: {opt.detected_intention.value}")
                print(f"🤖 Recommended Agents: {opt.recommended_agents}")

                # Try to execute with recommended agents
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
                                agent_result = await registry.call_agent(
                                    agent_name, query
                                )

                                # Extract meaningful result
                                if hasattr(agent_result, "optimized_text"):
                                    print(
                                        f"✅ Result: {agent_result.optimized_text[:100]}..."
                                    )
                                elif hasattr(agent_result, "corrected_text"):
                                    print(
                                        f"✅ Result: {agent_result.corrected_text[:100]}..."
                                    )
                                elif hasattr(agent_result, "query"):
                                    print(f"✅ Result: {agent_result.query[:100]}...")
                                elif hasattr(agent_result, "sentiment"):
                                    print(f"✅ Result: {agent_result.sentiment}")
                                else:
                                    print(f"✅ Result: {str(agent_result)[:100]}...")

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
            print(f"   Intent: {result.optimized_prompt.detected_intention.value}")
            print(f"   Confidence: {result.optimized_prompt.confidence_score:.2f}")
            print(f"   Execution Plan: {' → '.join(result.execution_plan)}")
            print(
                f"   Context Elements: {len(result.optimized_prompt.context_elements)} identified"
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
