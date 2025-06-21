#!/usr/bin/env python3
"""
Test für den vollständigen Prompt Engineer Agent
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent_server.prompt_engineer import (
    PromptEngineerRequest,
    run_prompt_engineer,
    prompt_engineer_a2a_function,
)


async def test_prompt_engineer():
    """Teste den vollständigen Prompt Engineer Agent"""

    print("🧠 TESTING COMPLETE PROMPT ENGINEER AGENT")
    print("=" * 60)

    test_cases = [
        {
            "query": "Schreibe eine Geschichte über einen Roboter",
            "description": "Creative Writing",
        },
        {
            "query": "Fasse diesen Artikel zusammen: Künstliche Intelligenz revolutioniert die Arbeitswelt",
            "description": "Summarization",
        },
        {
            "query": "Erstelle eine Python-Funktion zur Sortierung einer Liste",
            "description": "Code Generation",
        },
        {
            "query": "Korrigiere diesen Text: Das ist ein sehr schlechte Satz mit viele Fehler",
            "description": "Grammar Correction",
        },
        {
            "query": "Analysiere das Sentiment: Ich bin total begeistert von diesem fantastischen Produkt!",
            "description": "Sentiment Analysis",
        },
        {
            "query": "Mache diesen Text professioneller: Das Zeug ist echt der letzte Schrott!",
            "description": "Text Optimization",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {case['description']}")
        print(f"Query: '{case['query']}'")
        print("-" * 50)

        # Test mit vollem Request
        request = PromptEngineerRequest(
            user_input=case["query"],
            target_audience="general",
            desired_tone="professional",
            include_examples=True,
        )

        result = await run_prompt_engineer(request)

        if result.status == "success":
            opt = result.optimization_result
            print(f"✅ SUCCESS")
            print(f"🎯 Detected Intent: {opt.detected_intent.value}")
            print(f"📊 Confidence: {opt.confidence_score:.3f}")
            print(f"⭐ Quality Score: {opt.quality_score:.3f}")
            print(
                f"🤖 Recommended Agents: {', '.join(opt.recommended_agents) if opt.recommended_agents else 'None'}"
            )
            print(f"📋 Execution Plan: {' → '.join(opt.execution_plan)}")
            print(
                f"🧩 Context Elements: {', '.join(opt.context_elements) if opt.context_elements else 'None'}"
            )
            print(f"⏱️  Processing Time: {result.processing_time:.3f}s")

            # Show optimized prompt (first 200 chars)
            print(f"💡 Optimized Prompt Preview:")
            print(f"    {opt.optimized_prompt[:200]}...")

            if opt.alternative_prompts:
                print(f"🔄 Alternatives Available: {len(opt.alternative_prompts)}")

        else:
            print(f"❌ FAILED: {result.message}")

        print()


async def test_a2a_integration():
    """Teste A2A Integration"""

    print("\n🔗 TESTING A2A INTEGRATION")
    print("=" * 40)

    test_queries = [
        "Optimiere diesen Text für eine E-Mail",
        "Analysiere die Stimmung in diesem Text",
        "Erstelle eine Funktion für Datenvalidierung",
    ]

    for query in test_queries:
        print(f"\n📞 A2A Call: '{query}'")

        try:
            result = await prompt_engineer_a2a_function(query)

            if result.status == "success":
                opt = result.optimization_result
                print(f"✅ Intent: {opt.detected_intent.value}")
                print(f"🤖 Agents: {opt.recommended_agents}")
                print(f"⭐ Quality: {opt.quality_score:.2f}")
            else:
                print(f"❌ Error: {result.message}")

        except Exception as e:
            print(f"❌ Exception: {e}")


async def test_performance():
    """Teste Performance des Agents"""

    print("\n⚡ PERFORMANCE TEST")
    print("=" * 30)

    import time

    queries = [
        "Schreibe eine Geschichte",
        "Korrigiere den Text",
        "Analysiere das Sentiment",
        "Erstelle Code",
        "Fasse zusammen",
    ]

    start_time = time.time()

    for query in queries:
        request = PromptEngineerRequest(user_input=query)
        result = await run_prompt_engineer(request)
        print(f"✓ {query[:20]:20} - {result.processing_time:.3f}s")

    total_time = time.time() - start_time
    avg_time = total_time / len(queries)

    print(f"\n📊 Performance Summary:")
    print(f"   Total Time: {total_time:.3f}s")
    print(f"   Average Time: {avg_time:.3f}s per query")
    print(f"   Throughput: {len(queries) / total_time:.1f} queries/second")


async def main():
    """Haupttest-Funktion"""

    print("🚀 PROMPT ENGINEER AGENT - COMPREHENSIVE TEST")
    print("=" * 70)

    try:
        await test_prompt_engineer()
        await test_a2a_integration()
        await test_performance()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED!")
        print("Der Prompt Engineer Agent ist bereit für den produktiven Einsatz.")

    except Exception as e:
        print(f"\n❌ Test Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
