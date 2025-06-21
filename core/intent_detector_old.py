from typing import Dict, List, Tuple
from enum import Enum
import re
from dataclasses import dataclass


class IntentType(Enum):
    """Präzise Intent-Kategorien für bessere Klassifizierung"""

    # Text-Verarbeitung
    GRAMMAR_CORRECTION = "grammar_correction"
    TEXT_OPTIMIZATION = "text_optimization"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"

    # Kreative Aufgaben
    CREATIVE_WRITING = "creative_writing"
    STORYTELLING = "storytelling"
    POETRY = "poetry"

    # Technische Aufgaben
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_EXPLANATION = "code_explanation"

    # Analyse & Information
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DATA_ANALYSIS = "data_analysis"
    INFORMATION_SEARCH = "information_search"
    QUESTION_ANSWERING = "question_answering"

    # Kommunikation
    EMAIL_WRITING = "email_writing"
    FORMAL_WRITING = "formal_writing"
    CASUAL_WRITING = "casual_writing"

    # Spezielle Services
    TIME_QUERY = "time_query"
    WEATHER_QUERY = "weather_query"
    FILE_PROCESSING = "file_processing"

    # Fallback
    GENERAL_ASSISTANCE = "general_assistance"


@dataclass
class IntentPattern:
    """Pattern für Intent-Erkennung mit Gewichtung"""

    keywords: List[str]
    phrases: List[str]
    regex_patterns: List[str]
    weight: float = 1.0
    exclude_keywords: List[str] = None


class EnhancedIntentDetector:
    """Verbesserte Intent Detection mit spezifischen Mustern"""

    def __init__(self):
        self.intent_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[IntentType, IntentPattern]:
        """Initialisiere präzise Intent-Muster"""
        return {
            IntentType.GRAMMAR_CORRECTION: IntentPattern(
                keywords=[
                    "korrigiere",
                    "korrektur",
                    "fehler",
                    "rechtschreibung",
                    "grammatik",
                    "orthografie",
                ],
                phrases=[
                    "korrigiere den text",
                    "rechtschreibfehler",
                    "grammatikfehler",
                    "ist das richtig",
                ],
                regex_patterns=[
                    r"korrigier[e|t]?\s+(?:den\s+)?text",
                    r"fehler\s+(?:in|im)",
                ],
                weight=1.2,
            ),
            IntentType.CREATIVE_WRITING: IntentPattern(
                keywords=[
                    "geschichte",
                    "story",
                    "erzählung",
                    "märchen",
                    "roman",
                    "kurzgeschichte",
                ],
                phrases=[
                    "schreibe eine geschichte",
                    "erzähle eine story",
                    "erfinde eine geschichte",
                ],
                regex_patterns=[
                    r"schreib[e|t]?\s+(?:eine\s+)?geschichte",
                    r"erzähl[e|t]?\s+(?:eine\s+)?geschichte",
                ],
                weight=1.3,
                exclude_keywords=["zusammenfassung", "fasse zusammen", "artikel"],
            ),
            IntentType.SUMMARIZATION: IntentPattern(
                keywords=[
                    "zusammenfassung",
                    "fasse",
                    "zusammen",
                    "resümee",
                    "kurz",
                    "überblick",
                ],
                phrases=[
                    "fasse zusammen",
                    "erstelle zusammenfassung",
                    "kurze zusammenfassung",
                ],
                regex_patterns=[
                    r"fass[e|t]?\s+(?:das\s+|den\s+|diese[ns]?\s+)?(?:zusammen|kurz)",
                    r"zusammenfassung\s+(?:von|über)",
                ],
                weight=1.4,
            ),
            IntentType.TEXT_OPTIMIZATION: IntentPattern(
                keywords=[
                    "optimiere",
                    "verbessere",
                    "professionell",
                    "höflich",
                    "formal",
                    "tonalität",
                ],
                phrases=[
                    "mache professioneller",
                    "verbessere den ton",
                    "optimiere für",
                ],
                regex_patterns=[
                    r"optimier[e|t]?\s+(?:den\s+)?text",
                    r"mach[e|t]?\s+(?:das\s+)?(?:professioneller|höflicher)",
                ],
                weight=1.2,
            ),
            IntentType.SENTIMENT_ANALYSIS: IntentPattern(
                keywords=[
                    "sentiment",
                    "stimmung",
                    "emotion",
                    "gefühl",
                    "analyse",
                    "bewertung",
                ],
                phrases=[
                    "analysiere das sentiment",
                    "wie ist die stimmung",
                    "emotionale analyse",
                ],
                regex_patterns=[
                    r"analys?ier[e|t]?\s+(?:das\s+)?sentiment",
                    r"(?:wie\s+ist\s+die\s+)?stimmung",
                ],
                weight=1.3,
            ),
            IntentType.CODE_GENERATION: IntentPattern(
                keywords=[
                    "code",
                    "programmieren",
                    "python",
                    "javascript",
                    "funktion",
                    "klasse",
                    "algorithmus",
                ],
                phrases=["schreibe code", "programmiere", "erstelle funktion"],
                regex_patterns=[
                    r"schreib[e|t]?\s+(?:einen?\s+)?code",
                    r"programmier[e|t]?\s+(?:mir\s+)?(?:eine[ns]?)?",
                ],
                weight=1.2,
                exclude_keywords=["geschichte", "text", "artikel"],
            ),
            IntentType.WEATHER_QUERY: IntentPattern(
                keywords=["wetter", "temperatur", "regen", "sonne", "bewölkt", "grad"],
                phrases=["wie wird das wetter", "wetter heute", "wetter morgen"],
                regex_patterns=[
                    r"(?:wie\s+(?:wird|ist)\s+das\s+)?wetter",
                    r"temperatur\s+(?:heute|morgen|in)",
                ],
                weight=1.5,
            ),
            IntentType.TIME_QUERY: IntentPattern(
                keywords=["zeit", "uhrzeit", "spät", "datum", "heute", "jetzt"],
                phrases=["wie spät ist es", "welche zeit", "aktuelles datum"],
                regex_patterns=[
                    r"wie\s+spät\s+ist\s+es",
                    r"(?:welche\s+)?(?:zeit|uhrzeit)",
                    r"(?:aktuelles?\s+)?datum",
                ],
                weight=1.6,
            ),
            IntentType.EMAIL_WRITING: IntentPattern(
                keywords=["email", "e-mail", "mail", "anschreiben", "nachricht"],
                phrases=[
                    "schreibe eine email",
                    "erstelle eine nachricht",
                    "email verfassen",
                ],
                regex_patterns=[
                    r"schreib[e|t]?\s+(?:eine\s+)?e?-?mail",
                    r"verfass[e|t]?\s+(?:eine\s+)?nachricht",
                ],
                weight=1.3,
            ),
            IntentType.TRANSLATION: IntentPattern(
                keywords=[
                    "übersetze",
                    "übersetzung",
                    "englisch",
                    "deutsch",
                    "französisch",
                    "translate",
                ],
                phrases=["übersetze ins", "übersetzung von", "auf deutsch"],
                regex_patterns=[
                    r"übersetz[e|t]?\s+(?:das\s+|ins\s+|in\s+|auf\s+)",
                    r"(?:ins?\s+)?(?:deutsche?|englische?)",
                ],
                weight=1.4,
            ),
            IntentType.INFORMATION_SEARCH: IntentPattern(
                keywords=[
                    "suche",
                    "finde",
                    "information",
                    "erkläre",
                    "was",
                    "wie",
                    "warum",
                ],
                phrases=["suche nach", "finde information", "was ist", "erkläre mir"],
                regex_patterns=[
                    r"such[e|t]?\s+(?:nach\s+|mir\s+)",
                    r"(?:was|wie|warum)\s+ist",
                    r"erklär[e|t]?\s+mir",
                ],
                weight=0.8,  # Niedrigere Gewichtung, da sehr allgemein
            ),
        }

    def detect_intent(
        self, query: str, confidence_threshold: float = 0.3
    ) -> Tuple[IntentType, float]:
        """
        Erkenne Intent mit verbesserter Genauigkeit

        Args:
            query: Benutzeranfrage
            confidence_threshold: Mindest-Konfidenz für Intent-Erkennung

        Returns:
            Tuple[IntentType, confidence_score]
        """
        query_lower = query.lower().strip()
        intent_scores = {}

        for intent_type, pattern in self.intent_patterns.items():
            score = self._calculate_intent_score(query_lower, pattern)
            if score > 0:
                intent_scores[intent_type] = score

        if not intent_scores:
            return IntentType.GENERAL_ASSISTANCE, 0.1

        # Sortiere nach Score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_intents[0]

        # Prüfe Konfidenz-Schwellwert
        if best_score < confidence_threshold:
            return IntentType.GENERAL_ASSISTANCE, best_score

        return best_intent, best_score

    def _calculate_intent_score(self, query: str, pattern: IntentPattern) -> float:
        """Berechne Score für Intent basierend auf Mustern"""
        score = 0.0

        # Prüfe Ausschluss-Keywords
        if pattern.exclude_keywords:
            for exclude_keyword in pattern.exclude_keywords:
                if exclude_keyword in query:
                    return 0.0  # Intent ausgeschlossen

        # Keyword-Matching (gewichtet)
        keyword_matches = sum(1 for keyword in pattern.keywords if keyword in query)
        if keyword_matches > 0:
            score += (keyword_matches / len(pattern.keywords)) * 0.4

        # Phrase-Matching (höhere Gewichtung)
        phrase_matches = sum(1 for phrase in pattern.phrases if phrase in query)
        if phrase_matches > 0:
            score += (phrase_matches / len(pattern.phrases)) * 0.6

        # Regex-Matching (höchste Gewichtung)
        regex_matches = sum(
            1
            for regex_pattern in pattern.regex_patterns
            if re.search(regex_pattern, query)
        )
        if regex_matches > 0:
            score += (regex_matches / len(pattern.regex_patterns)) * 0.8

        # Anwenden der Pattern-Gewichtung
        score *= pattern.weight

        return min(score, 1.0)  # Score auf 1.0 begrenzen

    def get_intent_confidence_report(self, query: str) -> Dict[str, float]:
        """Erstelle detaillierten Konfidenz-Report für Debugging"""
        query_lower = query.lower().strip()
        report = {}

        for intent_type, pattern in self.intent_patterns.items():
            score = self._calculate_intent_score(query_lower, pattern)
            if score > 0:
                report[intent_type.value] = round(score, 3)

        return dict(sorted(report.items(), key=lambda x: x[1], reverse=True))


# Globale Instanz
intent_detector = EnhancedIntentDetector()


def detect_user_intent(query: str) -> Tuple[str, float]:
    """Convenience-Funktion für Intent Detection"""
    intent_type, confidence = intent_detector.detect_intent(query)
    return intent_type.value, confidence


def debug_intent_detection(query: str) -> Dict:
    """Debug-Funktion für Intent Detection"""
    intent_type, confidence = intent_detector.detect_intent(query)
    confidence_report = intent_detector.get_intent_confidence_report(query)

    return {
        "query": query,
        "detected_intent": intent_type.value,
        "confidence": confidence,
        "all_scores": confidence_report,
    }
