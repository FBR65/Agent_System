import os
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from enum import Enum

load_dotenv(override=True, dotenv_path="../../.env")

# Set up logging
logger = logging.getLogger(__name__)


class TaskIntention(str, Enum):
    """Enum for different task intentions"""

    INFORMATION_RETRIEVAL = "information_retrieval"
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    CREATIVE_WRITING = "creative_writing"
    PROBLEM_SOLVING = "problem_solving"
    DATA_PROCESSING = "data_processing"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONVERSATION = "conversation"


class PromptTemplate(BaseModel):
    """Template for optimized prompts"""

    template_id: str
    intention: TaskIntention
    template: str
    variables: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    best_practices: List[str] = Field(default_factory=list)


class ContextElement(BaseModel):
    """Individual context element"""

    type: str  # 'constraint', 'example', 'domain_knowledge', 'format_spec'
    content: str
    priority: int = Field(ge=1, le=10)  # 1=lowest, 10=highest


class OptimizedPrompt(BaseModel):
    """Optimized prompt structure"""

    original_request: str
    detected_intention: TaskIntention
    optimized_prompt: str
    context_elements: List[ContextElement] = Field(default_factory=list)
    recommended_agents: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class PromptEngineerRequest(BaseModel):
    """Request model for prompt engineering"""

    user_input: str
    target_model: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None
    enable_a2a_routing: bool = True


class PromptEngineerResponse(BaseModel):
    """Response model for prompt engineering"""

    optimized_prompt: OptimizedPrompt
    execution_plan: List[str] = Field(default_factory=list)
    alternative_prompts: List[str] = Field(default_factory=list)
    status: str
    message: str


class PromptEngineerContext(BaseModel):
    """Context for the Prompt Engineer agent"""

    request_id: str = "default"
    session_history: List[Dict[str, Any]] = Field(default_factory=list)


# Prompt templates database
PROMPT_TEMPLATES = {
    TaskIntention.INFORMATION_RETRIEVAL: PromptTemplate(
        template_id="info_retrieval_v1",
        intention=TaskIntention.INFORMATION_RETRIEVAL,
        template="""Sie sind ein Experte für Informationsbeschaffung. Ihre Aufgabe ist es, präzise und umfassende Antworten auf die folgende Anfrage zu geben.

ANFRAGE: {user_query}

ANWEISUNGEN:
1. Strukturieren Sie Ihre Antwort klar und logisch
2. Verwenden Sie Aufzählungen oder Nummerierungen für bessere Lesbarkeit
3. Nennen Sie vertrauenswürdige Quellen, wenn möglich
4. Geben Sie zusätzlichen Kontext, der relevant sein könnte
5. Markieren Sie Unsicherheiten oder Annahmen deutlich

AUSGABEFORMAT:
- Direkte Antwort auf die Hauptfrage
- Zusätzliche relevante Informationen
- Quellen/Referenzen (falls verfügbar)
- Verwandte Themen (optional)

Antworten Sie ausführlich und präzise:""",
        variables=["user_query"],
        examples=[
            "Was sind die wichtigsten Trends in der KI-Entwicklung 2024?",
            "Erkläre mir die Grundlagen der Quantenphysik",
        ],
        best_practices=[
            "Verwende strukturierte Ausgaben",
            "Priorisiere Faktentreue",
            "Gib Quellen an wo möglich",
        ],
    ),
    TaskIntention.CODE_GENERATION: PromptTemplate(
        template_id="code_gen_v1",
        intention=TaskIntention.CODE_GENERATION,
        template="""Sie sind ein erfahrener Software-Entwickler. Erstellen Sie sauberen, effizienten und gut dokumentierten Code für die folgende Anforderung.

ANFORDERUNG: {user_query}

KONTEXT:
- Programmiersprache: {language}
- Zusätzliche Anforderungen: {requirements}

ANWEISUNGEN:
1. Schreiben Sie sauberen, lesbaren Code
2. Fügen Sie aussagekräftige Kommentare hinzu
3. Befolgen Sie Best Practices der gewählten Sprache
4. Implementieren Sie Error Handling wo angebracht
5. Erklären Sie die Logik bei komplexen Algorithmen

AUSGABEFORMAT:
```{language}
# Ihr Code hier
```

ERKLÄRUNG:
- Kurze Beschreibung der Lösung
- Erklärung wichtiger Design-Entscheidungen
- Hinweise zur Verwendung

Beginnen Sie mit der Implementierung:""",
        variables=["user_query", "language", "requirements"],
        examples=[
            "Erstelle eine Python-Funktion zur Sortierung einer Liste",
            "Implementiere einen REST-API Endpoint in FastAPI",
        ],
        best_practices=[
            "Code-Kommentare sind essentiell",
            "Verwende Type Hints",
            "Implementiere Error Handling",
        ],
    ),
    TaskIntention.TEXT_ANALYSIS: PromptTemplate(
        template_id="text_analysis_v1",
        intention=TaskIntention.TEXT_ANALYSIS,
        template="""Sie sind ein Experte für Textanalyse. Analysieren Sie den folgenden Text gründlich und systematisch.

ZU ANALYSIERENDER TEXT:
"{text_content}"

ANALYSE-ANFRAGE: {user_query}

ANALYSEDIMENSIONEN:
1. **Inhaltliche Analyse**
   - Hauptthemen und Kernaussagen
   - Argumentationsstruktur
   - Faktische vs. meinungsbasierte Aussagen

2. **Sprachliche Analyse**
   - Tonalität und Stil
   - Sprachliche Besonderheiten
   - Zielgruppe und Register

3. **Strukturelle Analyse**
   - Textaufbau und Gliederung
   - Kohärenz und Kohäsion
   - Verwendete Textstrategien

AUSGABEFORMAT:
## Zusammenfassung
[Kurze Zusammenfassung der wichtigsten Erkenntnisse]

## Detailanalyse
[Strukturierte Analyse nach den oben genannten Dimensionen]

## Schlussfolgerungen
[Ihre Bewertung und Interpretation]

Beginnen Sie mit der Analyse:""",
        variables=["text_content", "user_query"],
        examples=[
            "Analysiere die Argumentationsstruktur dieses politischen Textes",
            "Bewerte die Überzeugungskraft dieser Werbung",
        ],
    ),
    TaskIntention.PROBLEM_SOLVING: PromptTemplate(
        template_id="problem_solving_v1",
        intention=TaskIntention.PROBLEM_SOLVING,
        template="""Sie sind ein systematischer Problem-Löser. Analysieren Sie das folgende Problem und entwickeln Sie eine strukturierte Lösung.

PROBLEM: {user_query}

KONTEXT: {context}

LÖSUNGSANSATZ:
1. **Problemanalyse**
   - Kernproblem identifizieren
   - Teilprobleme aufschlüsseln
   - Einschränkungen und Rahmenbedingungen

2. **Lösungsoptionen**
   - Verschiedene Ansätze bewerten
   - Vor- und Nachteile abwägen
   - Machbarkeitsprüfung

3. **Empfohlene Lösung**
   - Schritt-für-Schritt Anleitung
   - Benötigte Ressourcen
   - Erfolgsmetriken

4. **Risikobewertung**
   - Potentielle Hindernisse
   - Risikominimierung
   - Backup-Pläne

Entwickeln Sie eine durchdachte Lösung:""",
        variables=["user_query", "context"],
        examples=[
            "Wie kann ich die Performance meiner Webanwendung verbessern?",
            "Entwickle eine Strategie zur Kostenreduzierung",
        ],
    ),
}


def _create_prompt_engineer_agent():
    """Create the prompt engineer agent with proper Ollama configuration."""
    try:
        llm_api_key = os.getenv("API_KEY", "ollama")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("PROMPT_ENGINEER_MODEL", "qwen2.5:latest")

        provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
        model = OpenAIModel(provider=provider, model_name=llm_model_name)

        return Agent(
            model=model,
            result_type=PromptEngineerResponse,
            system_prompt="""Sie sind ein Experte für Prompt Engineering und Agent-Orchestrierung. Ihre Aufgaben:

1. **Intent Detection**: Analysieren Sie Benutzeranfragen und erkennen Sie die zugrundeliegende Absicht
2. **Prompt Optimization**: Erstellen Sie optimierte Prompts basierend auf bewährten Praktiken
3. **Agent Routing**: Empfehlen Sie die besten verfügbaren Agenten für spezifische Aufgaben
4. **Context Management**: Verwalten Sie relevante Kontextinformationen effizient

VERFÜGBARE AGENTEN IN UNSEREM A2A SYSTEM:
- `lektor`: Grammatik- und Rechtschreibkorrektur (Deutsch/Englisch)
- `optimizer`: Textoptimierung für verschiedene Tonalitäten
- `sentiment`: Sentiment- und Emotionsanalyse
- `query_ref`: Anfrage-Optimierung für LLMs
- `user_interface`: Intelligente Anfrage-Weiterleitung

MCP TOOLS VERFÜGBAR:
- `get_current_time`: Aktuelle Zeit/Datum
- `duckduckgo_search`: Web-Suche mit Wetter-Optimierung
- `extract_website_text`: Website-Inhalte extrahieren
- `anonymize_text`: PII-Entfernung
- `convert_to_pdf`: Dateikonvertierung

ANWEISUNGEN:
1. Analysieren Sie die Benutzeranfrage gründlich
2. Identifizieren Sie die beste Lösungsstrategie
3. Erstellen Sie optimierte Prompts mit klarer Struktur
4. Empfehlen Sie geeignete Agenten oder Tools
5. Geben Sie alternative Ansätze wenn sinnvoll

AUSGABE: Strukturierte Antwort mit OptimizedPrompt und Ausführungsplan.""",
        )
    except Exception as e:
        logging.error(f"Failed to initialize prompt engineer agent: {e}")
        raise


prompt_engineer_agent = _create_prompt_engineer_agent()


class IntentDetector:
    """Advanced intent detection for user queries"""

    @staticmethod
    def detect_intention(user_input: str) -> TaskIntention:
        """Detect the primary intention from user input"""
        text_lower = user_input.lower()

        # Code generation keywords
        if any(
            keyword in text_lower
            for keyword in [
                "code",
                "programmier",
                "implement",
                "funktion",
                "class",
                "script",
                "erstelle",
                "schreibe",
                "develop",
                "api",
                "algorithm",
            ]
        ):
            return TaskIntention.CODE_GENERATION

        # Information retrieval keywords
        elif any(
            keyword in text_lower
            for keyword in [
                "was ist",
                "wie funktioniert",
                "erkläre",
                "erklär",
                "explain",
                "what is",
                "how does",
                "define",
                "definition",
                "bedeutung",
            ]
        ):
            return TaskIntention.INFORMATION_RETRIEVAL

        # Text analysis keywords
        elif any(
            keyword in text_lower
            for keyword in [
                "analysiere",
                "analyze",
                "bewerte",
                "evaluate",
                "sentiment",
                "auswerte",
                "untersuche",
                "prüfe",
                "check",
            ]
        ):
            return TaskIntention.TEXT_ANALYSIS

        # Problem solving keywords
        elif any(
            keyword in text_lower
            for keyword in [
                "problem",
                "lösung",
                "solution",
                "wie kann ich",
                "how can i",
                "strategie",
                "strategy",
                "optimiere",
                "optimize",
                "verbesser",
            ]
        ):
            return TaskIntention.PROBLEM_SOLVING

        # Creative writing keywords
        elif any(
            keyword in text_lower
            for keyword in [
                "schreibe",
                "write",
                "erstelle",
                "create",
                "geschichte",
                "story",
                "gedicht",
                "poem",
                "brief",
                "letter",
                "artikel",
                "article",
            ]
        ):
            return TaskIntention.CREATIVE_WRITING

        # Translation keywords
        elif any(
            keyword in text_lower
            for keyword in ["übersetze", "translate", "übersetz", "translation"]
        ):
            return TaskIntention.TRANSLATION

        # Summarization keywords
        elif any(
            keyword in text_lower
            for keyword in [
                "zusammenfass",
                "summarize",
                "fasse zusammen",
                "summary",
                "überblick",
                "overview",
                "kurz",
            ]
        ):
            return TaskIntention.SUMMARIZATION

        else:
            return TaskIntention.CONVERSATION


class PromptOptimizer:
    """Advanced prompt optimization based on detected intent"""

    def __init__(self):
        self.templates = PROMPT_TEMPLATES

    def optimize_prompt(
        self,
        user_input: str,
        intention: TaskIntention,
        context: Optional[Dict[str, Any]] = None,
    ) -> OptimizedPrompt:
        """Optimize a prompt based on intention and context"""

        # Get appropriate template
        template = self.templates.get(intention)

        if template:
            # Fill template with user input and context
            optimized_prompt = self._fill_template(template, user_input, context or {})
            confidence = 0.9
        else:
            # Fallback: create optimized prompt without template
            optimized_prompt = self._create_fallback_prompt(user_input, intention)
            confidence = 0.6

        # Extract context elements
        context_elements = self._extract_context_elements(user_input)

        # Recommend agents based on intention
        recommended_agents = self._recommend_agents(intention, user_input)

        return OptimizedPrompt(
            original_request=user_input,
            detected_intention=intention,
            optimized_prompt=optimized_prompt,
            context_elements=context_elements,
            recommended_agents=recommended_agents,
            confidence_score=confidence,
            reasoning=f"Detected intention: {intention.value}. Applied optimization strategies for this task type.",
        )

    def _fill_template(
        self, template: PromptTemplate, user_input: str, context: Dict[str, Any]
    ) -> str:
        """Fill a template with user input and context"""
        variables = {
            "user_query": user_input,
            "context": context.get("context", ""),
            "language": context.get("language", "Python"),
            "requirements": context.get("requirements", "Standard requirements"),
            "text_content": context.get("text_content", user_input),
        }

        try:
            return template.template.format(**variables)
        except KeyError as e:
            logger.warning(f"Template variable missing: {e}. Using fallback.")
            return self._create_fallback_prompt(user_input, template.intention)

    def _create_fallback_prompt(self, user_input: str, intention: TaskIntention) -> str:
        """Create a fallback optimized prompt"""
        base_prompts = {
            TaskIntention.INFORMATION_RETRIEVAL: f"Beantworten Sie die folgende Frage ausführlich und strukturiert: {user_input}",
            TaskIntention.CODE_GENERATION: f"Erstellen Sie sauberen, gut dokumentierten Code für: {user_input}",
            TaskIntention.TEXT_ANALYSIS: f"Analysieren Sie den folgenden Text systematisch: {user_input}",
            TaskIntention.PROBLEM_SOLVING: f"Entwickeln Sie eine strukturierte Lösung für: {user_input}",
            TaskIntention.CREATIVE_WRITING: f"Schreiben Sie kreativ und ansprechend zu: {user_input}",
            TaskIntention.TRANSLATION: f"Übersetzen Sie präzise und kontextgerecht: {user_input}",
            TaskIntention.SUMMARIZATION: f"Erstellen Sie eine prägnante Zusammenfassung von: {user_input}",
            TaskIntention.CONVERSATION: f"Antworten Sie hilfsbereit und informativ auf: {user_input}",
        }

        return base_prompts.get(
            intention, f"Bearbeiten Sie die folgende Anfrage sorgfältig: {user_input}"
        )

    def _extract_context_elements(self, user_input: str) -> List[ContextElement]:
        """Extract context elements from user input"""
        elements = []
        text_lower = user_input.lower()

        # Format specifications
        if any(
            word in text_lower for word in ["json", "xml", "csv", "markdown", "html"]
        ):
            elements.append(
                ContextElement(
                    type="format_spec",
                    content="Strukturierte Ausgabe erforderlich",
                    priority=8,
                )
            )

        # Constraints
        if any(
            word in text_lower
            for word in ["kurz", "short", "länge", "maximal", "minimal"]
        ):
            elements.append(
                ContextElement(
                    type="constraint", content="Längenbeschränkung beachten", priority=7
                )
            )

        # Examples needed
        if any(
            word in text_lower
            for word in ["beispiel", "example", "zeige", "demonstrate"]
        ):
            elements.append(
                ContextElement(
                    type="example", content="Beispiele erforderlich", priority=6
                )
            )

        return elements

    def _recommend_agents(self, intention: TaskIntention, user_input: str) -> List[str]:
        """Recommend appropriate agents based on intention and input"""
        text_lower = user_input.lower()
        recommended = []

        # Text processing agents
        if any(
            word in text_lower
            for word in ["korrigier", "correct", "grammatik", "rechtschreib"]
        ):
            recommended.append("lektor")

        if any(
            word in text_lower
            for word in ["optimier", "verbesser", "professional", "freundlich"]
        ):
            recommended.append("optimizer")

        if intention == TaskIntention.TEXT_ANALYSIS or "sentiment" in text_lower:
            recommended.append("sentiment")

        # Query optimization
        if intention == TaskIntention.INFORMATION_RETRIEVAL:
            recommended.append("query_ref")

        # User interface for complex routing
        if len(recommended) > 1 or intention == TaskIntention.CONVERSATION:
            recommended.append("user_interface")

        return recommended


async def run_prompt_engineer(request: PromptEngineerRequest) -> PromptEngineerResponse:
    """Run prompt engineering analysis and optimization"""
    try:
        # Step 1: Detect intention
        intention = IntentDetector.detect_intention(request.user_input)
        logger.info(f"Detected intention: {intention}")

        # Step 2: Optimize prompt
        optimizer = PromptOptimizer()
        optimized_prompt = optimizer.optimize_prompt(
            request.user_input, intention, request.additional_context
        )

        # Step 3: Create execution plan
        execution_plan = []

        if optimized_prompt.recommended_agents:
            execution_plan.append(
                f"Route to agents: {', '.join(optimized_prompt.recommended_agents)}"
            )

        execution_plan.extend(
            [
                "Execute optimized prompt",
                "Validate results",
                "Return structured response",
            ]
        )

        # Step 4: Generate alternative prompts
        alternatives = [
            f"Kurze Version: {request.user_input} (Bitte halten Sie die Antwort prägnant.)",
            f"Detaillierte Version: {request.user_input} (Bitte geben Sie eine ausführliche Antwort mit Beispielen.)",
            f"Schritt-für-Schritt: {request.user_input} (Bitte erklären Sie jeden Schritt detailliert.)",
        ]

        return PromptEngineerResponse(
            optimized_prompt=optimized_prompt,
            execution_plan=execution_plan,
            alternative_prompts=alternatives,
            status="success",
            message=f"Prompt successfully optimized for {intention.value} task",
        )

    except Exception as e:
        logger.error(f"Prompt engineering failed: {e}")
        return PromptEngineerResponse(
            optimized_prompt=OptimizedPrompt(
                original_request=request.user_input,
                detected_intention=TaskIntention.CONVERSATION,
                optimized_prompt=request.user_input,
                confidence_score=0.1,
                reasoning=f"Error during optimization: {str(e)}",
            ),
            status="error",
            message=f"Prompt engineering failed: {str(e)}",
        )


# A2A server function for prompt engineering
async def prompt_engineer_a2a_function(
    messages: List[ModelMessage],
) -> PromptEngineerResponse:
    """A2A endpoint for prompt engineering functionality."""
    if not messages:
        return PromptEngineerResponse(
            optimized_prompt=OptimizedPrompt(
                original_request="",
                detected_intention=TaskIntention.CONVERSATION,
                optimized_prompt="",
                confidence_score=0.0,
                reasoning="No messages provided",
            ),
            status="error",
            message="No messages provided",
        )

    # Extract text from the last user message
    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        text = last_message.content
    else:
        text = str(last_message)

    request = PromptEngineerRequest(user_input=text)
    return await run_prompt_engineer(request)


# Example usage
async def main():
    """Main function to demonstrate Prompt Engineer Agent usage."""
    test_cases = [
        "Erkläre mir die Grundlagen der Quantenphysik",
        "Erstelle eine Python-Funktion zur Sortierung einer Liste",
        "Analysiere das Sentiment in diesem Text: Ich bin sehr enttäuscht von diesem Produkt",
        "Wie kann ich die Performance meiner Webanwendung verbessern?",
        "Übersetze diesen Text ins Englische: Guten Tag",
    ]

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test_text}")
        print("=" * 80)

        request = PromptEngineerRequest(user_input=test_text)
        result = await run_prompt_engineer(request)

        print(f"Detected Intention: {result.optimized_prompt.detected_intention}")
        print(f"Confidence: {result.optimized_prompt.confidence_score:.2f}")
        print(f"Recommended Agents: {result.optimized_prompt.recommended_agents}")
        print(f"Optimized Prompt:\n{result.optimized_prompt.optimized_prompt[:300]}...")
        print(f"Execution Plan: {' → '.join(result.execution_plan)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
