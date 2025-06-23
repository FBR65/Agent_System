#!/usr/bin/env python3
"""
WORKING LEKTOR - ACTIVE NOW
"""

import asyncio
import logging
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True, dotenv_path="../../.env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LektorResponse(BaseModel):
    """Response model for lektor agent."""

    corrected_text: str
    status: str = "success"
    message: str = "Grammar corrected with working system"


# Context for the lektor agent
class LektorContext(BaseModel):
    """Context passed to the lektor agent."""

    request_id: str = Field(default="default")


def _create_lektor_agent():
    """Create the lektor agent with proper Ollama configuration."""
    try:
        llm_api_key = os.getenv("API_KEY", "ollama")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("LEKTOR_MODEL", "qwen2.5:latest")

        provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
        model = OpenAIModel(provider=provider, model_name=llm_model_name)

        return Agent(
            model=model,
            result_type=LektorResponse,
            retries=3,
            system_prompt="""Du bist ein professioneller deutscher Lektor.

AUFGABE: Korrigiere ALLE Grammatik-, Rechtschreib- und Satzbaufehler im gegebenen Text. 
Gib NUR den korrigierten Text zurück, KEINE Erklärungen oder Kommentare.

DEUTSCHE GRAMMATIK REGELN:
- "ein schlechte Satz" → "ein schlechter Satz" (Adjektiv-Deklination maskulin)
- "mit viele Fehler" → "mit vielen Fehlern" (Dativ Plural)
- "Das Auto sind rot" → "Das Auto ist rot" (Singular Verb)
- "Die Bücher ist" → "Die Bücher sind" (Plural Verb)

BEISPIEL:
Input: "Das ist ein sehr schlechte Satz mit viele Fehler."
Output: "Das ist ein sehr schlechter Satz mit vielen Fehlern."

Korrigiere folgenden Text VOLLSTÄNDIG:""",
        )
    except Exception as e:
        logger.error(f"Failed to initialize lektor agent: {e}")
        raise


# Create the lektor agent
lektor_agent = _create_lektor_agent()


async def lektor_a2a_function(messages: List[ModelMessage]) -> LektorResponse:
    """A2A endpoint for lektor functionality."""
    if not messages:
        return LektorResponse(
            corrected_text="", status="error", message="No text provided for correction"
        )

    # Extract text from the last user message - IMPROVED extraction
    last_message = messages[-1]

    # Handle different message types
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        text = last_message.content
    elif hasattr(last_message, "content"):
        text = str(last_message.content)
    else:
        text = str(last_message)

    # DEBUG: Verbesserte Logging für A2A-Übertragung
    logger.info(f"WORKING LEKTOR: Received full text: '{text}'")
    logger.info(f"WORKING LEKTOR: Text length: {len(text)}")
    logger.info(f"WORKING LEKTOR: Message type: {type(last_message)}")

    # Fallback falls Text leer oder nur Punkt ist
    if not text or text.strip() in ["", "."]:
        logger.error("WORKING LEKTOR: Received empty or invalid text!")
        return LektorResponse(
            corrected_text="Fehler: Kein Text zum Korrigieren erhalten.",
            status="error",
            message="No valid text received for correction",
        )

    try:
        context = LektorContext(request_id="a2a_request")
        result = await lektor_agent.run(text, ctx=context)

        logger.info(f"WORKING LEKTOR: Processing '{text}'")
        logger.info(f"WORKING LEKTOR: Result '{result.data.corrected_text}'")

        return result.data
    except Exception as e:
        logger.error(f"Lektor processing failed: {e}")
        return LektorResponse(
            corrected_text=text,  # Return original text if correction fails
            status="error",
            message=f"Correction failed: {str(e)}",
        )


async def process_lektor_request(text: str) -> LektorResponse:
    """Process lektor request directly."""
    context = LektorContext(request_id="direct_request")
    result = await lektor_agent.run(text, ctx=context)
    return result.data


# Example usage
async def main():
    """Main function to demonstrate lektor agent usage."""
    test_texts = [
        "Das ist ein sehr schlechte Satz mit viele Fehler.",
        "Die Bücher ist auf der Tisch.",
        "Ich gehe zu die Schule morgen.",
        "Er haben ein rote Auto gekauft.",
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")
        result = await process_lektor_request(text)
        print(f"Corrected: {result.corrected_text}")
        print(f"Status: {result.status}")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
