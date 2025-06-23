#!/usr/bin/env python3
"""
IMPROVED WORKING OPTIMIZER
"""

import asyncio
import logging
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)


class OptimizerRequest(BaseModel):
    text: str
    tonality: str = "friendly"


class OptimizerResponse(BaseModel):
    optimized_text: str
    status: str = "success"
    message: str = ""


async def run_optimizer(request: OptimizerRequest) -> OptimizerResponse:
    """Improved working optimizer using comprehensive rule-based optimization"""
    try:
        full_text = request.text
        text_to_optimize = full_text  # Default: use full text

        # Extract tonality from input if present
        tonality = request.tonality  # Use the request tonality as default
        if "TONALITY:" in full_text:
            parts = full_text.split("|")
            for part in parts:
                if part.startswith("TONALITY:"):
                    tonality = part.replace("TONALITY:", "").strip()
                elif part.startswith("TEXT:"):
                    text_to_optimize = part.replace("TEXT:", "").strip()
        elif ":" in full_text:
            parts = full_text.split(":", 1)
            if len(parts) > 1:
                text_to_optimize = parts[1].strip()
            else:
                text_to_optimize = full_text
        else:
            text_to_optimize = full_text

        logger.info(
            f"IMPROVED OPTIMIZER: Processing '{text_to_optimize}' with tonality '{tonality}'"
        )

        # SPECIAL HANDLING FOR COMMON BUSINESS TEXTS
        text_lower = text_to_optimize.lower()

        # Handle formal greetings specifically for different tonalities
        if "sehr geehrte damen und herren" in text_lower:
            if tonality in ["locker", "lockerer", "casual", "entspannt"]:
                optimized = "Hallo zusammen! 👋"
            elif tonality in ["freundlich", "friendly", "warm"]:
                optimized = "Liebe Grüße an alle!"
            elif tonality in ["direkt", "direct", "prägnant"]:
                optimized = "Hallo!"
            elif tonality in ["begeistert", "enthusiastic"]:
                optimized = "Hallo liebe Leute! 🌟"
            else:
                # Default friendly tone
                optimized = "Hallo! Schön, dass Sie da sind!"

        # For ALL OTHER texts, apply real optimization based on tonality
        else:
            optimized = text_to_optimize

            # Enhanced negative-to-positive replacements
            replacements = {
                "Schrott": "nicht optimal",
                "schlecht": "verbesserungswürdig",
                "furchtbar": "nicht zufriedenstellend",
                "schrecklich": "unzureichend",
                "hässlich": "weniger ansprechend",
                "dumm": "nicht durchdacht",
                "blöd": "ungeschickt",
                "katastrophal": "herausfordernd",
                "unmöglich": "schwierig umsetzbar",
                "nutzlos": "wenig hilfreich",
                "abgelehnt": "nicht genehmigt",
                "rejected": "nicht berücksichtigt",
            }

            # Apply replacements
            for negative, positive in replacements.items():
                optimized = optimized.replace(negative, positive)

            # Apply tonality-specific transformations
            if tonality in ["locker", "lockerer", "casual", "entspannt"]:
                # Make it more casual and relaxed
                optimized = optimized.replace("Sie", "du")
                optimized = optimized.replace("Ihre", "deine")
                optimized = optimized.replace("Ihnen", "dir")
                optimized = optimized.replace("wurde", "ist")
                optimized = f"Hey! {optimized} 😊"
            elif tonality in ["freundlich", "friendly", "warm"]:
                # Make it friendlier without changing the core message
                if "nicht genehmigt" in optimized or "abgelehnt" in text_lower:
                    optimized = f"Vielen Dank für Ihre Anfrage! {optimized} Wir würden uns freuen, Ihnen bei einer anderen Gelegenheit helfen zu können."
                else:
                    optimized = f"Gerne teile ich mit: {optimized}"
            elif tonality in ["direkt", "direct", "prägnant"]:
                # Keep it short and to the point
                optimized = optimized.replace("Vielen Dank für", "")
                optimized = optimized.strip()
            elif tonality in ["begeistert", "enthusiastic"]:
                optimized = f"Das ist eine wichtige Mitteilung: {optimized} 🌟"

        logger.info(f"IMPROVED OPTIMIZER: Result '{optimized}'")

        return OptimizerResponse(
            optimized_text=optimized,
            status="success",
            message="Text successfully optimized with improved working system",
        )

    except Exception as e:
        logger.error(f"Improved optimizer error: {e}")
        return OptimizerResponse(
            optimized_text=text_to_optimize
            if "text_to_optimize" in locals()
            else "optimization failed",
            status="error",
            message=f"Optimization failed: {e}",
        )


async def process_optimization_request(request: OptimizerRequest) -> OptimizerResponse:
    """Process optimization request directly."""
    return await run_optimizer(request)


async def optimizer_a2a_function(messages) -> OptimizerResponse:
    """A2A endpoint for optimizer functionality."""
    logger.info("🎯 OPTIMIZER A2A FUNCTION CALLED")

    if not messages:
        logger.error("❌ No messages provided to optimizer")
        return OptimizerResponse(
            optimized_text="",
            status="error",
            message="No text provided for optimization",
        )

    # Extract text from the last user message - IMPROVED extraction
    last_message = messages[-1]
    logger.info(f"🔍 Last message type: {type(last_message)}")
    logger.info(f"🔍 Last message: {last_message}")

    # Handle different message types
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        input_text = last_message.content
    elif hasattr(last_message, "content"):
        input_text = str(last_message.content)
    else:
        input_text = str(last_message)

    logger.info(f"🎯 IMPROVED OPTIMIZER: Extracted text: '{input_text}'")
    logger.info(f"🎯 IMPROVED OPTIMIZER: Text length: {len(input_text)}")

    # Validate that we have actual text content
    if not input_text or input_text.strip() == "":
        logger.error("❌ Empty text provided to optimizer")
        return OptimizerResponse(
            optimized_text="",
            status="error",
            message="Empty text provided for optimization",
        )

    try:
        # Create request with the extracted text
        request = OptimizerRequest(text=input_text)
        logger.info(f"🎯 Created OptimizerRequest: {request}")

        result = await process_optimization_request(request)

        logger.info(f"🎯 IMPROVED OPTIMIZER: Processing '{request.text}'")
        logger.info(f"🎯 IMPROVED OPTIMIZER: Result '{result.optimized_text}'")
        logger.info(f"🎯 IMPROVED OPTIMIZER: Returning result")

        return result
    except Exception as e:
        logger.error(f"❌ Optimizer processing failed: {e}")
        import traceback

        logger.error(f"❌ Optimizer traceback: {traceback.format_exc()}")
        return OptimizerResponse(
            optimized_text=input_text,  # Return original text if optimization fails
            status="error",
            message=f"Optimization failed: {str(e)}",
        )


if __name__ == "__main__":
    # Test the optimizer
    async def test():
        request = OptimizerRequest(text="Das ist Schrott!")
        result = await run_optimizer(request)
        print(f"Input: Das ist Schrott!")
        print(f"Output: {result.optimized_text}")

    asyncio.run(test())
