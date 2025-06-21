#!/usr/bin/env python3
"""
Advanced Prompt Engineer Agent
Kombiniert Intent Detection, Prompt Optimization und A2A Agent Routing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.intent_detector_improved import intent_detector, IntentType
from core.config_manager import config
from pydantic_ai import Agent
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class PromptOptimizationResult:
    """Ergebnis der Prompt-Optimierung"""

    original_query: str
    detected_intent: IntentType
    confidence_score: float
    optimized_prompt: str
    recommended_agents: List[str]
    execution_plan: List[str]
    context_elements: List[str]
    quality_score: float
    alternative_prompts: List[str] = field(default_factory=list)
    reasoning: str = ""


class PromptEngineerRequest(BaseModel):
    """Request für den Prompt Engineer Agent"""

    user_input: str
    context: Optional[str] = None
    target_audience: Optional[str] = "general"
    desired_tone: Optional[str] = "professional"
    max_length: Optional[int] = 4096
    include_examples: bool = True
    optimize_for_llm: str = "qwen2.5:latest"


class PromptEngineerResponse(BaseModel):
    """Response vom Prompt Engineer Agent"""

    status: str
    message: str
    optimization_result: Optional[PromptOptimizationResult] = None
    processing_time: float
    timestamp: str


class PromptEngineerAgent:
    """Fortschrittlicher Prompt Engineering Agent"""

    def __init__(self):
        self.agent_model = config.get_agent_model("prompt_engineer")
        self.templates = self._load_prompt_templates()
        self.agent_mappings = self._initialize_agent_mappings()

    def _load_prompt_templates(self) -> Dict[IntentType, str]:
        """Lade spezialisierte Prompt-Templates"""
        return {
            IntentType.CREATIVE_WRITING: """
Du bist ein erfahrener Schriftsteller und Geschichtenerzähler. 
Aufgabe: {task}
Kontext: {context}
Zielgruppe: {audience}
Ton: {tone}

Erstelle eine fesselnde und kreative Antwort, die:
- Lebendige Beschreibungen verwendet
- Emotionale Verbindungen schafft
- Einen klaren narrativen Bogen hat
- Zur Zielgruppe passt

{examples}
""",
            IntentType.CODE_GENERATION: """
Du bist ein erfahrener Softwareentwickler und Code-Architekt.
Aufgabe: {task}
Kontext: {context}
Technische Anforderungen: {requirements}

Erstelle sauberen, gut dokumentierten Code, der:
- Best Practices befolgt
- Vollständig funktionsfähig ist
- Kommentare und Dokumentation enthält
- Fehlerbehandlung implementiert
- Testbar und wartbar ist

{examples}
""",
            IntentType.SUMMARIZATION: """
Du bist ein Experte für Textanalyse und Zusammenfassungen.
Aufgabe: {task}
Originaltext: {content}
Gewünschte Länge: {length}
Fokus: {focus}

Erstelle eine präzise Zusammenfassung, die:
- Die wichtigsten Punkte erfasst
- Strukturiert und logisch aufgebaut ist
- Den Originalton bewahrt
- Für die Zielgruppe verständlich ist

{examples}
""",
            IntentType.SENTIMENT_ANALYSIS: """
Du bist ein Experte für Emotionsanalyse und Textverständnis.
Aufgabe: {task}
Zu analysierender Text: {text}
Analysetiefe: {depth}

Führe eine detaillierte Sentiment-Analyse durch, die:
- Emotionen präzise identifiziert
- Stimmungsintensität bewertet
- Kontextuelle Nuancen erkennt
- Begründungen für die Bewertung liefert

{examples}
""",
            IntentType.TEXT_OPTIMIZATION: """
Du bist ein Experte für professionelle Kommunikation und Textoptimierung.
Aufgabe: {task}
Originaltext: {original}
Zielton: {tone}
Kontext: {context}

Optimiere den Text für:
- Klarheit und Verständlichkeit
- Professionellen Ton
- Zielgruppengerechte Sprache
- Wirkungsvolle Formulierung

{examples}
""",
            IntentType.GRAMMAR_CORRECTION: """
Du bist ein Experte für deutsche Grammatik und Rechtschreibung.
Aufgabe: {task}
Zu korrigierender Text: {text}
Sprache: {language}

Korrigiere alle Fehler bezüglich:
- Rechtschreibung
- Grammatik
- Zeichensetzung
- Stilistische Verbesserungen

Erkläre die wichtigsten Korrekturen.

{examples}
""",
            IntentType.TRANSLATION: """
Du bist ein professioneller Übersetzer mit Expertise in verschiedenen Sprachen.
Aufgabe: {task}
Originaltext: {text}
Zielsprache: {target_language}
Kontext: {context}

Erstelle eine präzise Übersetzung, die:
- Bedeutung und Nuancen bewahrt
- Kulturelle Kontexte berücksichtigt
- Natürlich und idiomatisch klingt
- Fachterminologie korrekt übersetzt

{examples}
""",
            IntentType.INFORMATION_SEARCH: """
Du bist ein Experte für Informationsrecherche und Wissensaufbereitung.
Aufgabe: {task}
Suchkontext: {context}
Detailgrad: {detail_level}

Stelle strukturierte, akkurate Informationen bereit, die:
- Vollständig und relevant sind
- Aus vertrauenswürdigen Quellen stammen
- Logisch organisiert sind
- Praktisch anwendbar sind

{examples}
""",
        }

    def _initialize_agent_mappings(self) -> Dict[IntentType, List[str]]:
        """Initialisiere Zuordnung von Intents zu verfügbaren Agenten"""
        return {
            IntentType.TEXT_OPTIMIZATION: ["optimizer", "lektor"],
            IntentType.GRAMMAR_CORRECTION: ["lektor"],
            IntentType.SENTIMENT_ANALYSIS: ["sentiment"],
            IntentType.CREATIVE_WRITING: ["optimizer"],
            IntentType.CODE_GENERATION: ["query_ref"],
            IntentType.SUMMARIZATION: ["optimizer"],
            IntentType.TRANSLATION: ["optimizer"],
            IntentType.INFORMATION_SEARCH: ["query_ref"],
            IntentType.QUESTION_ANSWERING: ["query_ref"],
            IntentType.EMAIL_WRITING: ["optimizer", "lektor"],
            IntentType.FORMAL_WRITING: ["optimizer", "lektor"],
            IntentType.TIME_QUERY: [],  # MCP Tool
            IntentType.WEATHER_QUERY: [],  # MCP Tool
            IntentType.GENERAL_ASSISTANCE: ["user_interface"],
        }

    async def optimize_prompt(
        self, request: PromptEngineerRequest
    ) -> PromptOptimizationResult:
        """Hauptfunktion für Prompt-Optimierung"""
        start_time = datetime.now()

        # 1. Intent Detection
        detected_intent, confidence = intent_detector.detect_intent(request.user_input)
        logger.info(
            f"Detected intent: {detected_intent.value} (confidence: {confidence:.3f})"
        )

        # 2. Context Analysis
        context_elements = self._analyze_context(request.user_input, request.context)

        # 3. Prompt Template Selection
        template = self._select_template(detected_intent)

        # 4. Prompt Generation
        optimized_prompt = self._generate_optimized_prompt(
            template, request, detected_intent, context_elements
        )

        # 5. Agent Routing Recommendations
        recommended_agents = self._recommend_agents(detected_intent, context_elements)

        # 6. Execution Plan
        execution_plan = self._create_execution_plan(
            detected_intent, recommended_agents
        )

        # 7. Quality Assessment
        quality_score = self._assess_prompt_quality(optimized_prompt, detected_intent)

        # 8. Alternative Prompts (optional)
        alternatives = (
            self._generate_alternatives(request, detected_intent)
            if quality_score < 0.8
            else []
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return PromptOptimizationResult(
            original_query=request.user_input,
            detected_intent=detected_intent,
            confidence_score=confidence,
            optimized_prompt=optimized_prompt,
            recommended_agents=recommended_agents,
            execution_plan=execution_plan,
            context_elements=context_elements,
            quality_score=quality_score,
            alternative_prompts=alternatives,
            reasoning=f"Intent detected with {confidence:.1%} confidence. "
            f"Optimized for {request.optimize_for_llm} model.",
        )

    def _analyze_context(
        self, query: str, additional_context: Optional[str] = None
    ) -> List[str]:
        """Analysiere Kontext-Elemente der Anfrage"""
        elements = []

        # Sprache erkennen
        if any(word in query.lower() for word in ["english", "englisch", "translate"]):
            elements.append("multilingual")

        # Formalität erkennen
        if any(
            word in query.lower()
            for word in ["professional", "professionell", "formal", "email"]
        ):
            elements.append("formal_tone")

        # Technischer Kontext
        if any(
            word in query.lower()
            for word in ["code", "python", "api", "function", "algorithm"]
        ):
            elements.append("technical")

        # Kreativität
        if any(
            word in query.lower()
            for word in ["geschichte", "story", "kreativ", "erfinde"]
        ):
            elements.append("creative")

        # Dringlichkeit
        if any(word in query.lower() for word in ["schnell", "urgent", "sofort"]):
            elements.append("urgent")

        if additional_context:
            elements.append("additional_context_provided")

        return elements

    def _select_template(self, intent: IntentType) -> str:
        """Wähle das passende Template für den Intent"""
        return self.templates.get(intent, self.templates[IntentType.INFORMATION_SEARCH])

    def _generate_optimized_prompt(
        self,
        template: str,
        request: PromptEngineerRequest,
        intent: IntentType,
        context_elements: List[str],
    ) -> str:
        """Generiere optimierten Prompt basierend auf Template"""

        # Beispiele basierend auf Intent
        examples = self._get_examples_for_intent(intent)

        # Template-Variablen füllen
        template_vars = {
            "task": request.user_input,
            "context": request.context or "Nicht spezifiziert",
            "audience": request.target_audience,
            "tone": request.desired_tone,
            "examples": examples,
            "requirements": "Standard Best Practices",
            "content": request.user_input,
            "length": f"ca. {request.max_length // 4} Wörter",
            "focus": "Hauptpunkte",
            "text": request.user_input,
            "depth": "detailliert",
            "original": request.user_input,
            "language": "Deutsch",
            "target_language": "Englisch",
            "detail_level": "umfassend",
        }

        # Template mit Variablen füllen
        try:
            optimized_prompt = template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template variable missing: {e}")
            optimized_prompt = f"Aufgabe: {request.user_input}\n\nBitte bearbeite diese Anfrage professionell und detailliert."

        # Kontext-spezifische Anpassungen
        if "formal_tone" in context_elements:
            optimized_prompt += (
                "\n\nWichtig: Verwende einen professionellen, formellen Ton."
            )

        if "urgent" in context_elements:
            optimized_prompt += "\n\nHinweis: Diese Anfrage hat hohe Priorität."

        if "technical" in context_elements:
            optimized_prompt += "\n\nTechnischer Kontext: Berücksichtige technische Genauigkeit und Best Practices."

        return optimized_prompt.strip()

    def _get_examples_for_intent(self, intent: IntentType) -> str:
        """Liefere Intent-spezifische Beispiele"""
        examples = {
            IntentType.CREATIVE_WRITING: """
Beispiel für gute Struktur:
- Spannender Einstieg
- Charakterentwicklung
- Höhepunkt
- Befriedigende Auflösung
""",
            IntentType.CODE_GENERATION: """
Beispiel für guten Code:
```python
def example_function(param: str) -> str:
    \"\"\"Dokumentierte Funktion mit Typ-Hints.\"\"\"
    if not param:
        raise ValueError("Parameter darf nicht leer sein")
    return param.upper()
```
""",
            IntentType.SENTIMENT_ANALYSIS: """
Beispiel-Analyse:
- Sentiment: Positiv (0.8)
- Emotionen: Freude, Zufriedenheit
- Intensität: Hoch
- Begründung: Verwendung positiver Adjektive
""",
        }
        return examples.get(intent, "")

    def _recommend_agents(
        self, intent: IntentType, context_elements: List[str]
    ) -> List[str]:
        """Empfehle passende Agenten für den Intent"""
        base_agents = self.agent_mappings.get(intent, [])

        # Kontext-basierte Anpassungen
        if "formal_tone" in context_elements and "lektor" not in base_agents:
            base_agents.append("lektor")

        if "technical" in context_elements and "query_ref" not in base_agents:
            base_agents.append("query_ref")

        return base_agents

    def _create_execution_plan(
        self, intent: IntentType, recommended_agents: List[str]
    ) -> List[str]:
        """Erstelle Ausführungsplan basierend auf Intent und Agenten"""
        plans = {
            IntentType.TEXT_OPTIMIZATION: [
                "Text analysieren",
                "Ton anpassen",
                "Qualität prüfen",
            ],
            IntentType.CREATIVE_WRITING: [
                "Idee entwickeln",
                "Struktur planen",
                "Text verfassen",
                "Überarbeiten",
            ],
            IntentType.CODE_GENERATION: [
                "Anforderungen analysieren",
                "Architektur planen",
                "Code implementieren",
                "Tests schreiben",
            ],
            IntentType.GRAMMAR_CORRECTION: [
                "Text analysieren",
                "Fehler identifizieren",
                "Korrekturen anwenden",
            ],
            IntentType.SENTIMENT_ANALYSIS: [
                "Text segmentieren",
                "Emotionen identifizieren",
                "Bewertung erstellen",
            ],
            IntentType.SUMMARIZATION: [
                "Hauptpunkte extrahieren",
                "Struktur erstellen",
                "Zusammenfassung formulieren",
            ],
        }

        base_plan = plans.get(
            intent, ["Anfrage analysieren", "Antwort generieren", "Qualität prüfen"]
        )

        # Agent-spezifische Erweiterungen
        if recommended_agents:
            base_plan.append(f"A2A Routing: {' → '.join(recommended_agents)}")

        return base_plan

    def _assess_prompt_quality(self, prompt: str, intent: IntentType) -> float:
        """Bewerte die Qualität des generierten Prompts"""
        score = 0.0

        # Länge bewerten (nicht zu kurz, nicht zu lang)
        length_score = min(len(prompt) / 500, 1.0) * 0.2
        score += length_score

        # Struktur bewerten
        if "Aufgabe:" in prompt or "Du bist" in prompt:
            score += 0.3

        # Intent-spezifische Elemente
        intent_keywords = {
            IntentType.CREATIVE_WRITING: ["kreativ", "Geschichte", "Erzählung"],
            IntentType.CODE_GENERATION: ["Code", "Funktion", "implementier"],
            IntentType.SENTIMENT_ANALYSIS: ["Sentiment", "Emotion", "Analyse"],
            IntentType.TEXT_OPTIMIZATION: ["optimier", "verbessere", "professionell"],
        }

        keywords = intent_keywords.get(intent, [])
        keyword_matches = sum(
            1 for keyword in keywords if keyword.lower() in prompt.lower()
        )
        if keywords:
            score += (keyword_matches / len(keywords)) * 0.3

        # Vollständigkeit (Beispiele, Anweisungen)
        if "Beispiel" in prompt or "beispiel" in prompt:
            score += 0.1

        if len(prompt.split("\n")) > 3:  # Strukturiert
            score += 0.1

        return min(score, 1.0)

    def _generate_alternatives(
        self, request: PromptEngineerRequest, intent: IntentType
    ) -> List[str]:
        """Generiere alternative Prompt-Ansätze"""
        alternatives = []

        # Vereinfachte Version
        simple = f"Einfache Aufgabe: {request.user_input}\nBitte bearbeite das kurz und präzise."
        alternatives.append(simple)

        # Detaillierte Version
        detailed = f"""
Detaillierte Aufgabe: {request.user_input}

Bitte bearbeite diese Anfrage mit besonderer Aufmerksamkeit auf:
1. Vollständigkeit der Antwort
2. Präzision und Genauigkeit
3. Praktische Anwendbarkeit
4. Verständlichkeit für die Zielgruppe

Zielgruppe: {request.target_audience}
Gewünschter Ton: {request.desired_tone}
"""
        alternatives.append(detailed.strip())

        return alternatives


# Pydantic AI Agent Setup
prompt_engineer_ai = Agent(
    model=config.get_agent_model("prompt_engineer"),
    system_prompt="""Du bist ein fortschrittlicher Prompt Engineering Agent.
    Deine Aufgabe ist es, Benutzeranfragen zu analysieren und optimale Prompts zu generieren.""",
)

# Globale Instanz
prompt_engineer = PromptEngineerAgent()


async def run_prompt_engineer(request: PromptEngineerRequest) -> PromptEngineerResponse:
    """Haupteingang für den Prompt Engineer Agent"""
    start_time = datetime.now()

    try:
        # Prompt-Optimierung durchführen
        optimization_result = await prompt_engineer.optimize_prompt(request)

        processing_time = (datetime.now() - start_time).total_seconds()

        return PromptEngineerResponse(
            status="success",
            message=f"Prompt erfolgreich optimiert für Intent: {optimization_result.detected_intent.value}",
            optimization_result=optimization_result,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prompt engineering failed: {e}", exc_info=True)
        processing_time = (datetime.now() - start_time).total_seconds()

        return PromptEngineerResponse(
            status="error",
            message=f"Fehler bei der Prompt-Optimierung: {str(e)}",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
        )


# A2A Integration
async def prompt_engineer_a2a_function(user_input: str) -> PromptEngineerResponse:
    """A2A-kompatible Funktion für den Prompt Engineer Agent"""
    request = PromptEngineerRequest(user_input=user_input)
    return await run_prompt_engineer(request)


if __name__ == "__main__":
    # Test des Agents
    async def test_agent():
        test_queries = [
            "Schreibe eine Geschichte über einen Roboter",
            "Fasse diesen Artikel zusammen",
            "Erstelle eine Python-Funktion zur Sortierung",
            "Korrigiere diesen Text: Das ist ein sehr schlechte Satz",
            "Analysiere das Sentiment: Ich bin super glücklich!",
        ]

        for query in test_queries:
            print(f"\n🔍 Testing: {query}")
            request = PromptEngineerRequest(user_input=query)
            result = await run_prompt_engineer(request)

            if result.status == "success":
                opt = result.optimization_result
                print(
                    f"✅ Intent: {opt.detected_intent.value} ({opt.confidence_score:.2f})"
                )
                print(f"🤖 Agents: {opt.recommended_agents}")
                print(f"📋 Plan: {' → '.join(opt.execution_plan)}")
                print(f"⭐ Quality: {opt.quality_score:.2f}")
            else:
                print(f"❌ Error: {result.message}")

    asyncio.run(test_agent())
