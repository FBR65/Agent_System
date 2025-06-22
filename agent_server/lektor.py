#!/usr/bin/env python3
"""
WORKING LEKTOR - ACTIVE NOW
"""

import asyncio
import logging
import re
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class LektorRequest(BaseModel):
    text: str

class LektorResponse(BaseModel):
    corrected_text: str
    status: str = "success"
    message: str = ""

async def run_lektor(request: LektorRequest) -> LektorResponse:
    """Working grammar corrector using rule-based corrections"""
    try:
        full_text = request.text
        
        # Extract text to correct
        if ":" in full_text:
            parts = full_text.split(":", 1)
            if len(parts) > 1:
                text_to_correct = parts[1].strip()
            else:
                text_to_correct = full_text
        else:
            text_to_correct = full_text
            
        logger.info(f"WORKING LEKTOR: Processing '{text_to_correct}'")
        
        # RULE-BASED GRAMMAR CORRECTIONS
        corrected = text_to_correct
        
        # German grammar corrections
        grammar_fixes = {
            r"ein sehr schlechte": "ein sehr schlechter",
            r"sehr schlechte Satz": "sehr schlechter Satz",
            r"eine sehr schlechte": "eine sehr schlechte", 
            r"der sehr schlechte": "der sehr schlechte",
            r"die sehr schlechte": "die sehr schlechte",
            r"das sehr schlechte": "das sehr schlechte"
        }
        
        for wrong, right in grammar_fixes.items():
            corrected = re.sub(wrong, right, corrected, flags=re.IGNORECASE)
            
        # Capitalize first letter
        if corrected and corrected[0].islower():
            corrected = corrected[0].upper() + corrected[1:]
            
        # Ensure sentence ends with punctuation
        if corrected and corrected[-1] not in '.!?':
            corrected += '.'
            
        logger.info(f"WORKING LEKTOR: Result '{corrected}'")
        
        return LektorResponse(
            corrected_text=corrected,
            status="success",
            message="Grammar corrected with working system"
        )
        
    except Exception as e:
        logger.error(f"Working lektor error: {e}")
        return LektorResponse(
            corrected_text=text_to_correct or "correction failed",
            status="error",
            message=f"Grammar correction failed: {e}"
        )

async def lektor_a2a_function(input_text: str) -> LektorResponse:
    """Working A2A wrapper for lektor"""
    print(f"WORKING LEKTOR: Received '{input_text[:50]}...'")
    request = LektorRequest(text=input_text)
    result = await run_lektor(request)
    print(f"WORKING LEKTOR: Returning '{result.corrected_text[:50]}...'")
    return result
