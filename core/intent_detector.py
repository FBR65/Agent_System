#!/usr/bin/env python3
"""
COMPLETE Intent Detection with ALL IntentTypes
"""

from typing import Dict, List, Tuple
from enum import Enum


class IntentType(Enum):
    """Enum for ALL possible user intents"""

    ACADEMIC_WRITING = "academic_writing"
    BRAINSTORMING = "brainstorming"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    EMAIL_WRITING = "email_writing"
    FORMAL_WRITING = "formal_writing"
    GENERAL_ASSISTANCE = "general_assistance"
    GRAMMAR_CORRECTION = "grammar_correction"
    INFORMATION_SEARCH = "information_search"
    MARKETING_CONTENT = "marketing_content"
    PRESENTATION = "presentation"
    QUESTION_ANSWERING = "question_answering"
    REPORT_WRITING = "report_writing"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SOCIAL_MEDIA = "social_media"
    SUMMARIZATION = "summarization"
    TECHNICAL_WRITING = "technical_writing"
    TEXT_OPTIMIZATION = "text_optimization"
    TIME_QUERY = "time_query"
    TRANSLATION = "translation"
    WEATHER_QUERY = "weather_query"


class EnhancedIntentDetector:
    """Enhanced intent detector with improved pattern matching and priority checks"""

    def __init__(self):
        """Initialize the enhanced intent detector with improved patterns"""
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize pattern dictionary for intent detection"""
        return {
            IntentType.GRAMMAR_CORRECTION: [
                r"korrigier.*",
                r"fehler.*korrigier.*",
                r"grammatik.*",
                r"rechtschreibung.*",
                r".*ist.*falsch.*",
                r"verbessere.*grammatik.*",
            ],
            IntentType.TEXT_OPTIMIZATION: [
                r"optimier.*",
                r"verbessere.*text.*",
                r"mache.*besser.*",
                r"professioneller.*",
                r"überarbeite.*",
                r"poliere.*auf.*",
            ],
            IntentType.SENTIMENT_ANALYSIS: [
                r"sentiment.*",
                r"gefühl.*analy.*",
                r"stimmung.*",
                r"emotion.*analy.*",
                r"wie.*fühlt.*",
                r"analyse.*gefühl.*",
            ],
            IntentType.CREATIVE_WRITING: [
                r"schreibe.*geschichte.*",
                r"erstelle.*text.*",
                r"verfasse.*",
                r"kreative.*schreiben.*",
                r"dichte.*",
                r"erzähle.*geschichte.*",
            ],
            IntentType.CODE_GENERATION: [
                r"erstelle.*code.*",
                r"programmiere.*",
                r"schreibe.*funktion.*",
                r"python.*funktion.*",
                r"javascript.*",
                r"html.*css.*",
            ],
            IntentType.QUESTION_ANSWERING: [
                r"was.*ist.*",
                r"wie.*funktioniert.*",
                r"erkläre.*",
                r"warum.*",
                r"definition.*",
                r"bedeutung.*",
            ],
        }

    def detect_intent(self, query: str) -> Tuple[IntentType, float]:
        """
        Detect the intent of a user query with priority checks

        Args:
            query: User input string

        Returns:
            Tuple of (IntentType, confidence_score)
        """
        query_lower = query.lower().strip()

        # Priority checks for common misclassifications
        if self._is_grammar_correction_priority(query_lower):
            return IntentType.GRAMMAR_CORRECTION, 0.95

        if self._is_text_optimization_priority(query_lower):
            return IntentType.TEXT_OPTIMIZATION, 0.90

        # Enhanced pattern matching for better detection
        if any(
            word in query_lower
            for word in [
                "erstelle",
                "programmiere",
                "funktion",
                "code",
                "python",
                "javascript",
            ]
        ):
            return IntentType.CODE_GENERATION, 0.8
        elif any(
            word in query_lower
            for word in [
                "erkläre",
                "was ist",
                "wie funktioniert",
                "grundlagen",
                "wie kann ich",
                "optimieren",
                "website",
            ]
        ):
            return IntentType.INFORMATION_SEARCH, 0.7
        elif any(
            word in query_lower
            for word in ["übersetze", "ins englische", "ins deutsche", "translation"]
        ):
            return IntentType.TRANSLATION, 0.8
        elif any(
            word in query_lower
            for word in [
                "fasse zusammen",
                "zusammenfassung",
                "summarize",
                "fasse",
                "artikel zusammen",
            ]
        ):
            return IntentType.SUMMARIZATION, 0.8
        elif "schreib" in query_lower or "geschichte" in query_lower:
            return IntentType.CREATIVE_WRITING, 0.7
        elif "sentiment" in query_lower or "analysier" in query_lower:
            return IntentType.SENTIMENT_ANALYSIS, 0.8
        else:
            return IntentType.GENERAL_ASSISTANCE, 0.1

    def _is_grammar_correction_priority(self, query: str) -> bool:
        """Priority check for grammar correction"""
        grammar_indicators = [
            "korrigiere:",
            "korrigier:",
            "korrigiere ",
            "text korrigieren",
            "satz korrigieren",
            "fehler korrigieren",
            "sehr schlechte satz",
            "schlechte satz",
            "grammatik korrigieren",
            "rechtschreibung korrigieren",
        ]
        return any(indicator in query for indicator in grammar_indicators)

    def _is_text_optimization_priority(self, query: str) -> bool:
        """Priority check for text optimization"""
        optimization_indicators = [
            "optimiere diesen text",
            "optimier den text",
            "text für e-mail",
            "text für email",
            "text für eine e-mail",
            "mache professioneller",
            "verbessere den text",
            "text optimieren",
            "optimiere den text",
            "optimiere text",
            "für e-mail:",
        ]
        return any(indicator in query for indicator in optimization_indicators)


# Create global instance
intent_detector = EnhancedIntentDetector()
