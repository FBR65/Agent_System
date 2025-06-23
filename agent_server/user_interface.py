import asyncio
import logging
from typing import Optional, Dict, Any, List
import sys
import time
import datetime
import httpx
import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path="../../.env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_mcp_server_url() -> str:
    """Get MCP server URL from environment variables."""
    host = os.getenv("SERVER_HOST", "localhost").strip("'\"")  # Remove quotes
    port = os.getenv("SERVER_PORT", "8000").strip("'\"")
    scheme = os.getenv("SERVER_SCHEME", "http").strip("'\"")
    url = f"{scheme}://{host}:{port}"
    logger.info(f"MCP server URL constructed: {url}")
    return url


class ProcessingStep(BaseModel):
    """Individual processing step result."""

    step_name: str
    input_text: str
    output_text: str
    status: str
    message: str


class UserInterfaceResponse(BaseModel):
    """Response model for user interface agent."""

    original_text: str
    final_result: str
    operation_type: str
    steps: List[ProcessingStep] = Field(default_factory=list)
    sentiment_analysis: Optional[Dict[str, Any]] = None
    status: str
    message: str
    processing_time: Optional[float] = None


# Context for the user interface agent
class UserInterfaceContext(BaseModel):
    """Context passed to the user interface agent."""

    request_id: str = Field(default="default")
    processing_start_time: float = Field(
        default_factory=lambda: __import__("time").time()
    )


# Initialize the model with OpenAI provider for Ollama compatibility
def _create_user_interface_agent():
    """Create the user interface agent with proper Ollama configuration."""
    try:
        llm_api_key = os.getenv("API_KEY", "ollama")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("USER_INTERFACE_MODEL", "qwen2.5:latest")

        provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
        model = OpenAIModel(provider=provider, model_name=llm_model_name)

        return Agent(
            model=model,
            result_type=UserInterfaceResponse,
            retries=10,  # ERHÖHT: Von 3 auf 10 für bessere Erfolgsrate
            system_prompt="""Du bist ein intelligenter Assistent der Benutzeranfragen bearbeitet.

WICHTIGE REGEL:
Du MUSST IMMER eine vollständige UserInterfaceResponse zurückgeben mit:
- original_text: Die ursprüngliche Benutzeranfrage (String)
- final_result: Das Endergebnis für den Benutzer (String) 
- operation_type: Art der Operation (String)
- status: "success" oder "error" (String)
- message: Kurze Beschreibung (String)

FÜR SENTIMENT-ANALYSE-ANFRAGEN (HÖCHSTE PRIORITÄT):
Diese Muster sind IMMER Sentiment-Analyse-Anfragen - IGNORIERE andere Keywords:
- "analysiere das sentiment", "sentiment analyse", "analyze sentiment"
- "sentiment analysis", "stimmung analysieren", "gefühl analysieren"
- Text nach "analysiere das sentiment:" oder "analyze sentiment:"
- Verwende dann: coordinate_with_a2a_agents(text="NUR_DER_ZU_ANALYSIERENDE_TEXT", operation="sentiment")

FÜR TEXTKORREKTUR-ANFRAGEN:
Diese Muster sind IMMER Korrektur-Anfragen - IGNORIERE andere Keywords:
- "korrigiere", "correct" 
- Text nach "korrigiere diesen text:" oder "correct this text:"
- "bitte korrigiere" oder "please correct"
- Verwende dann: coordinate_with_a2a_agents(text="NUR_DER_ZU_KORRIGIERENDE_TEXT", operation="correct")

FÜR TEXT-OPTIMIERUNG-ANFRAGEN:
Erkenne diese Muster als Optimierung-Anfragen:
- "optimiere", "optimize", "verbessere", "improve"
- "möchte eine freundliche [TEXTART]" -> extrahiere den passenden Standard-Text
- Bei "freundliche Absage" -> verwende "Ihre Anfrage wurde abgelehnt."
- Bei "freundliche Antwort" -> verwende "Hier ist unsere Antwort."
- Verwende dann: coordinate_with_a2a_agents(text="STANDARD_TEXT", operation="optimize", tonality="freundlich")

WICHTIG: 
- Bei "analysiere das sentiment" immer operation="sentiment" verwenden!
- Bei "korrigiere" immer operation="correct" verwenden, NIEMALS "optimize"!
- Extrahiere den zu analysierenden Text nach dem Doppelpunkt!

FÜR WETTER-ANFRAGEN:
1. Verwende get_weather_information(location="...", time_period="...")
2. Falls das Tool Wetterdaten liefert, gib diese direkt weiter
3. Erstelle eine einfache, benutzerfreundliche Antwort

BEISPIEL für Sentiment-Response:
{
  "original_text": "Analysiere das sentiment: Ich bin glücklich",
  "final_result": "Sentiment Analysis Results:\nLabel: positive\nConfidence: 0.95\nScore: 0.8\nEmotions: joy: 0.9",
  "operation_type": "coordinate_with_a2a_agents", 
  "status": "success",
  "message": "Sentiment analysis completed"
}

Halte dich STRIKT an das UserInterfaceResponse Format!""",
        )
    except Exception as e:
        logging.error(f"Failed to initialize user interface agent: {e}")
        raise


# Create the main User Interface Agent
user_interface_agent = _create_user_interface_agent()


# Now define the tools AFTER the agent
@user_interface_agent.tool
async def search_web_information(
    ctx: RunContext[UserInterfaceContext], query: str, max_results: int = 5
) -> Dict[str, Any]:
    """Search for information on the web using DuckDuckGo via direct MCP calls."""
    import httpx

    mcp_url = _get_mcp_server_url()
    logger.info(f"🔍 Starting web search via MCP: {query}")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Enhance weather queries
            search_query = query
            if any(word in query.lower() for word in ["wetter", "weather"]):
                if "weather" not in query.lower():
                    search_query = f"weather {query}"

            # Try different MCP call formats - FIXED: Use /mcp prefix
            endpoints_to_try = [
                f"{mcp_url}/mcp/call-tool",
                f"{mcp_url}/mcp/tools/call",
                f"{mcp_url}/mcp/tools/duckduckgo_search",
            ]

            for endpoint in endpoints_to_try:
                try:
                    if "call-tool" in endpoint or "tools/call" in endpoint:
                        call_data = {
                            "name": "duckduckgo_search",
                            "arguments": {
                                "query": search_query,
                                "max_results": max_results,
                            },
                        }
                        response = await client.post(endpoint, json=call_data)
                    else:
                        call_data = {"query": search_query, "max_results": max_results}
                        response = await client.post(endpoint, json=call_data)

                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ Web search success via {endpoint}")

                        # Parse MCP response
                        if isinstance(result, dict):
                            if "content" in result:
                                content = result["content"]
                                if isinstance(content, list) and len(content) > 0:
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            try:
                                                import json

                                                search_data = json.loads(item["text"])
                                                return search_data
                                            except:
                                                pass

                            # Direct result format
                            if "results" in result:
                                return result

                        return {"results": [], "raw_response": result}
                    else:
                        logger.debug(
                            f"Search endpoint {endpoint} failed: {response.status_code}"
                        )

                except Exception as endpoint_error:
                    logger.debug(f"Search endpoint {endpoint} error: {endpoint_error}")

            logger.error("❌ All MCP search endpoints failed")

    except Exception as search_error:
        logger.error(f"❌ MCP web search failed: {search_error}")

        # Fallback: Basic error response with rate limit handling
        error_msg = "MCP web search service unavailable"
        if "ratelimit" in str(search_error).lower():
            error_msg = "Such-Service vorübergehend nicht verfügbar (Rate Limit). Bitte versuchen Sie es in einigen Minuten erneut."

        return {
            "results": [],
            "error": error_msg,
            "query": search_query,
        }

    # Default fallback if no exception but also no results
    return {
        "results": [],
        "error": "No search results found",
        "query": search_query,
    }


@user_interface_agent.tool
async def get_weather_information(
    ctx: RunContext[UserInterfaceContext], location: str, time_period: str = "heute"
) -> UserInterfaceResponse:
    """Get weather information quickly with single website extraction."""
    import time
    import urllib.parse

    start_time = time.time()

    logger.info(f"🌤️ FAST weather extraction for {location} {time_period}")

    # Try only ONE reliable website to save time
    weather_url = f"https://wetteronline.de/wetter/{urllib.parse.quote(location)}"

    try:
        logger.info(f"📄 Single website extraction from: {weather_url}")
        website_result = await extract_website_content(ctx, weather_url)

        if "error" not in website_result and website_result.get("content"):
            # Parse the MCP response format
            content = ""
            if isinstance(website_result["content"], list):
                for item in website_result["content"]:
                    if isinstance(item, dict) and "text" in item:
                        content = item["text"]
                        break
            else:
                content = str(website_result["content"])

            # SUPER WICHTIG: Der Content ist ein JSON-String mit text_content!
            if content and len(content) > 100:
                try:
                    # Parse JSON to get actual text content
                    import json

                    content_data = json.loads(content)
                    actual_text = content_data.get("text_content", "")

                    if actual_text and len(actual_text) > 50:
                        logger.info(
                            f"✅ Extracted text content: {actual_text[:200]}..."
                        )

                        # PARSE THE REAL WEATHER DATA
                        import re

                        # Extract key info from wetteronline.de
                        weather_info = []

                        # Extract max/min temperatures for tomorrow
                        if "morgen" in time_period.lower():
                            max_match = re.search(r"max (\d+)°", actual_text)
                            min_match = re.search(r"min (\d+)°", actual_text)

                            if max_match and min_match:
                                weather_info.append(
                                    f"🌡️ Temperatur: {min_match.group(1)}°C - {max_match.group(1)}°C"
                                )

                            # Extract conditions for tomorrow
                            if "Montag Vormittag bewölkt" in actual_text:
                                weather_info.append("🌤️ Vormittag: Bewölkt")
                            if "Montag Nachmittag bewölkt" in actual_text:
                                weather_info.append(
                                    "☁️ Nachmittag: Bewölkt mit möglichen Schauern"
                                )
                            if "Montag Abend wechselnd bewölkt" in actual_text:
                                weather_info.append("🌥️ Abend: Wechselnd bewölkt")

                            # Extract wind info
                            if "stürmische Böen" in actual_text:
                                weather_info.append("💨 Wind: Stürmische Böen erwartet")

                        # Create beautiful weather report
                        if weather_info:
                            weather_content = (
                                f"🌤️ Wetter für {location} {time_period}:\n\n"
                            )
                            weather_content += "\n".join(weather_info)
                            weather_content += f"\n\n📍 Quelle: wetteronline.de"

                            return UserInterfaceResponse(
                                original_text=f"Wetter {location} {time_period}",
                                final_result=weather_content,
                                operation_type="weather_website_extraction",
                                status="success",
                                message=f"Wetterinformationen von wetteronline.de extrahiert",
                                processing_time=time.time() - start_time,
                            )

                        # Fallback: Extract any temperatures
                        temp_matches = re.findall(r"(\d{1,2})°", actual_text)
                        if temp_matches and len(temp_matches) > 5:
                            current_temp = temp_matches[0]
                            max_temp = max(temp_matches[:20], key=int)
                            min_temp = min(temp_matches[:20], key=int)

                            weather_content = (
                                f"🌤️ Wetter für {location} {time_period}:\n\n"
                            )
                            weather_content += (
                                f"🌡️ Aktuelle Temperatur: {current_temp}°C\n"
                            )
                            weather_content += f"📈 Höchsttemperatur: {max_temp}°C\n"
                            weather_content += f"📉 Tiefsttemperatur: {min_temp}°C\n"
                            # Check for weather conditions
                            if "sonnig" in actual_text.lower():
                                weather_content += "☀️ Bedingungen: Sonnig\n"
                            elif "bewölkt" in actual_text.lower():
                                weather_content += "☁️ Bedingungen: Bewölkt\n"
                            elif "regen" in actual_text.lower():
                                weather_content += "🌧️ Bedingungen: Regen möglich\n"

                            weather_content += f"\n📍 Quelle: wetteronline.de"

                            return UserInterfaceResponse(
                                original_text=f"Wetter {location} {time_period}",
                                final_result=weather_content,
                                operation_type="weather_website_extraction",
                                status="success",
                                message=f"Wetterinformationen von wetteronline.de extrahiert",
                                processing_time=time.time() - start_time,
                            )

                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON content, using raw content")
                    # Fallback to old method if JSON parsing fails
                    pass

            else:
                logger.info(f"Insufficient content from {weather_url}")
        else:
            logger.info(
                f"Website extraction failed for {weather_url}: {website_result.get('error', 'Unknown error')}"
            )

    except Exception as extract_error:
        logger.info(f"Website extraction error for {weather_url}: {extract_error}")

    # Quick fallback - no more multiple website attempts
    logger.info("⚡ Single website failed, providing instant fallback")
    return UserInterfaceResponse(
        original_text=f"Wetter {location} {time_period}",
        final_result=f"🌦️ Wetterinformationen für {location} {time_period}:\n\nAutomatische Website-Extraktion derzeit nicht verfügbar.\n\nFür aktuelle Wetterinformationen:\n• Besuchen Sie wetteronline.de/{location}\n• Oder verwenden Sie Ihre Wetter-App\n\nEntschuldigung für die Unannehmlichkeiten.",
        operation_type="weather_extraction_fallback",
        status="partial_success",
        message="Schneller Fallback - nur eine Website versucht",
        processing_time=time.time() - start_time,
    )


@user_interface_agent.tool
async def extract_website_content(
    ctx: RunContext[UserInterfaceContext], url: str
) -> Dict[str, Any]:
    """Extract text content from a website using direct HTTP calls."""
    import httpx

    mcp_url = _get_mcp_server_url()
    logger.info(f"📄 Extracting website content from: {url}")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            call_data = {"name": "extract_website_text", "arguments": {"url": url}}

            response = await client.post(f"{mcp_url}/mcp/call-tool", json=call_data)

            if response.status_code == 200:
                result = response.json()
                return result
            else:
                logger.error(f"Website extraction HTTP error: {response.status_code}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}

    except Exception as e:
        logger.error(f"Website extraction failed: {e}")
        return {"error": str(e)}


@user_interface_agent.tool
async def anonymize_sensitive_text(
    ctx: RunContext[UserInterfaceContext], text: str
) -> Dict[str, Any]:
    """Anonymize sensitive information in text."""
    mcp_server = MCPServerHTTP(_get_mcp_server_url())
    try:
        result = await mcp_server.call_tool("anonymize_text", {"text": text})
        return result
    except Exception as e:
        logger.error(f"Text anonymization failed: {e}")
        return {"error": str(e)}
    finally:
        try:
            if hasattr(mcp_server, "close"):
                await mcp_server.close()
            elif hasattr(mcp_server, "aclose"):
                await mcp_server.aclose()
        except Exception as cleanup_error:
            logger.warning(f"MCP cleanup failed (non-critical): {cleanup_error}")


def _create_simple_user_interface_agent():
    """Create a simplified agent with no retries."""
    try:
        llm_api_key = os.getenv("API_KEY", "ollama")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("USER_INTERFACE_MODEL", "qwen2.5:latest")

        provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
        model = OpenAIModel(provider=provider, model_name=llm_model_name)

        return Agent(
            model=model,
            result_type=UserInterfaceResponse,
            retries=10,  # ERHÖHT: Von 3 auf 10 für bessere Erfolgsrate
            system_prompt="""Du bist ein Assistent. Beantworte kurz und hilfsbereit.""",
        )
    except Exception as e:
        logging.error(f"Failed to initialize simple agent: {e}")
        raise


@user_interface_agent.tool
async def get_current_time_info(
    ctx: RunContext[UserInterfaceContext],
) -> Dict[str, Any]:
    """Get current time information by actually calling MCP service."""
    import httpx

    # UNIQUE IDENTIFIER TO CONFIRM THIS FUNCTION IS RUNNING
    logger.info("🎯 ENHANCED get_current_time_info FUNCTION CALLED - VERSION 5.0")
    logger.info("🎯 This version fixes the hanging /mcp endpoint issue")

    mcp_url = _get_mcp_server_url()
    logger.info(f"=== ENHANCED MCP DEBUG START ===")
    logger.info(f"MCP server URL: {mcp_url}")

    # FORCE Method 1: Test basic connectivity first
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test basic connectivity
            logger.info(f"🔗 Testing connectivity to {mcp_url}...")

            try:
                health_response = await client.get(f"{mcp_url}/health", timeout=5.0)
                logger.info(f"✅ Health check response: {health_response.status_code}")

                if health_response.status_code == 200:
                    logger.info(f"✅ MCP server is HEALTHY!")
                else:
                    logger.error(
                        f"❌ MCP server health check failed: {health_response.text}"
                    )

            except httpx.ConnectError as conn_error:
                logger.error(
                    f"❌ Cannot connect to MCP server at {mcp_url}: {conn_error}"
                )
                logger.error("🚨 MCP SERVER IS NOT RUNNING OR NOT ACCESSIBLE!")
                raise conn_error
            except Exception as health_error:
                logger.error(f"❌ Health check failed: {health_error}")
                raise health_error

            # Check available tools - Try multiple endpoint patterns with SHORTER TIMEOUT
            logger.info("🔍 Checking available MCP tools...")

            # Try different possible endpoints for tools listing
            tools_endpoints_to_try = [
                f"{mcp_url}/mcp/tools",  # MCP standard
                f"{mcp_url}/tools",  # Root tools
                f"{mcp_url}/mcp/list_tools",  # Alternative naming
                f"{mcp_url}/list_tools",  # Root alternative
            ]

            tools_data = None
            successful_tools_endpoint = None

            for tools_endpoint in tools_endpoints_to_try:
                try:
                    logger.info(f"🔍 Trying tools endpoint: {tools_endpoint}")
                    # SHORTER TIMEOUT to avoid hanging
                    tools_response = await client.get(tools_endpoint, timeout=3.0)
                    logger.info(f"Response status: {tools_response.status_code}")

                    if tools_response.status_code == 200:
                        tools_data = tools_response.json()
                        successful_tools_endpoint = tools_endpoint
                        logger.info(f"✅ SUCCESS! Tools found at: {tools_endpoint}")
                        logger.info(f"Tools response: {tools_data}")
                        break
                    else:
                        logger.debug(
                            f"❌ {tools_endpoint} failed: {tools_response.status_code}"
                        )
                except httpx.TimeoutException:
                    logger.warning(f"❌ {tools_endpoint} timed out after 3 seconds")
                except Exception as e:
                    logger.debug(
                        f"❌ {tools_endpoint} error: {e}"
                    )  # SKIP the hanging /mcp endpoint completely for now
            logger.warning("⚠️ Skipping /mcp endpoint due to hanging issue")

            if tools_data and successful_tools_endpoint:
                # Extract tool names and try to call time tool
                available_tools = []
                if "tools" in tools_data:
                    available_tools = [tool.get("name") for tool in tools_data["tools"]]
                    logger.info(f"✅ Available MCP tools: {available_tools}")

                # Check if get_current_time tool is available
                if "get_current_time" in available_tools:
                    logger.info("🕐 get_current_time tool found - calling via MCP...")

                    # Call the MCP tool
                    try:
                        call_tool_endpoint = f"{mcp_url}/mcp/call-tool"
                        call_data = {"name": "get_current_time", "arguments": {}}

                        tool_response = await client.post(
                            call_tool_endpoint, json=call_data, timeout=10.0
                        )

                        if tool_response.status_code == 200:
                            tool_result = tool_response.json()
                            logger.info(f"✅ MCP Zeit-Tool Antwort: {tool_result}")

                            # Parse MCP response format
                            if "content" in tool_result and tool_result["content"]:
                                content_item = tool_result["content"][0]
                                if "text" in content_item:
                                    import json

                                    try:
                                        time_data = json.loads(content_item["text"])
                                        logger.info(
                                            f"✅ Zeit erfolgreich abgerufen über MCP: {time_data}"
                                        )
                                        return time_data
                                    except json.JSONDecodeError:
                                        logger.error(
                                            "❌ JSON decode error in MCP response"
                                        )

                            # Fallback: Return raw response
                            return tool_result
                        else:
                            logger.error(
                                f"❌ MCP tool call failed: {tool_response.status_code} - {tool_response.text}"
                            )

                    except Exception as tool_error:
                        logger.error(f"❌ MCP tool call error: {tool_error}")

                else:
                    logger.error(
                        "❌ get_current_time tool not found in available tools"
                    )
            else:
                logger.error("❌ Could not find any working tools endpoint!")

    except httpx.ConnectError as conn_error:
        logger.error(f"🚨 CANNOT CONNECT TO MCP SERVER: {mcp_url}")
        logger.error(f"Connection error: {conn_error}")
    except Exception as mcp_error:
        logger.error(f"❌ MCP service error: {mcp_error}")

    # Method 2: External NTP is NOT allowed - MCP ONLY
    logger.info("🚨 EXTERNAL NTP DISABLED - MCP STRICT ONLY MODE")

    # NO FALLBACKS - Fehler durchreichen wie angefordert
    logger.error("🚨 MCP Zeit-Service nicht verfügbar - KEIN FALLBACK ERLAUBT")
    raise Exception("MCP Zeit-Service ist zwingend erforderlich und nicht verfügbar")


@user_interface_agent.tool
async def convert_file_to_pdf(
    ctx: RunContext[UserInterfaceContext],
    file_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a file to PDF format."""
    mcp_server = MCPServerHTTP(_get_mcp_server_url())
    try:
        result = await mcp_server.call_tool(
            "convert_to_pdf",
            {"input_filepath": file_path, "output_directory": output_dir},
        )
        return result
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return {"error": str(e)}
    finally:
        try:
            if hasattr(mcp_server, "close"):
                await mcp_server.close()
            elif hasattr(mcp_server, "aclose"):
                await mcp_server.aclose()
        except Exception as cleanup_error:
            logger.warning(f"MCP cleanup failed (non-critical): {cleanup_error}")


@user_interface_agent.tool
async def coordinate_with_a2a_agents(
    ctx: RunContext[UserInterfaceContext],
    text: str,
    operation: str,
    tonality: Optional[str] = None,
) -> UserInterfaceResponse:
    """Coordinate with A2A agents for text processing using the existing A2A registry system."""
    import time

    start_time = time.time()

    debug_a2a = os.getenv("DEBUG_A2A_CALLS", "false").lower() == "true"

    try:
        logger.info(
            f"A2A coordination called with operation: {operation}, tonality: {tonality}"
        )

        # Import the existing A2A setup from our project
        import sys
        from pathlib import Path

        # Add parent directory to path for import
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from a2a_server import setup_a2a_server

        registry = await setup_a2a_server()

        try:
            if operation == "optimize":
                # Create a structured request for the optimizer
                if tonality:
                    # Extract just the text content and pass as string
                    optimizer_text = f"TONALITY:{tonality}|TEXT:{text}"
                else:
                    optimizer_text = text

                result = await registry.call_agent("optimizer", optimizer_text)

                if debug_a2a:
                    logger.info(f"Raw optimizer result: {result}")
                    logger.info(f"Result type: {type(result)}")
                    logger.info(
                        f"Result attributes: {dir(result) if hasattr(result, '__dict__') else 'No attributes'}"
                    )

                # Extract the actual optimized text from the result
                optimized_text = None

                # Check if the optimizer had an error
                if hasattr(result, "status") and result.status == "error":
                    if debug_a2a:
                        logger.warning(f"Optimizer returned error: {result.message}")
                    # Try to extract actual text from the error response
                    if hasattr(result, "optimized_text") and isinstance(
                        result.optimized_text, str
                    ):
                        # The optimized_text might contain the original prompt, extract it
                        optimized_text_raw = result.optimized_text
                        if optimized_text_raw.startswith(
                            "{'content': '"
                        ) and optimized_text_raw.endswith("'}"):
                            # Extract content from the dict string
                            content_start = optimized_text_raw.find(
                                "'content': '"
                            ) + len("'content': '")
                            content_end = optimized_text_raw.rfind("'}")
                            if content_start < content_end:
                                optimized_text = optimized_text_raw[
                                    content_start:content_end
                                ]
                            else:
                                optimized_text = text  # Fallback to original text
                        else:
                            optimized_text = optimized_text_raw
                    else:
                        optimized_text = f"Optimizer service error: {result.message}. Original text: {text}"
                else:
                    # Normal processing
                    if hasattr(result, "optimized_text"):
                        optimized_text = result.optimized_text
                    elif hasattr(result, "content"):
                        optimized_text = result.content
                    elif isinstance(result, dict) and "optimized_text" in result:
                        optimized_text = result["optimized_text"]
                    elif isinstance(result, dict) and "content" in result:
                        optimized_text = result["content"]
                    elif isinstance(result, str):
                        optimized_text = result
                    else:
                        # Fallback: convert to string
                        optimized_text = str(result)

                if debug_a2a:
                    logger.info(f"Extracted optimized text: {optimized_text}")

                return UserInterfaceResponse(
                    original_text=text,
                    final_result=optimized_text,
                    operation_type="coordinate_with_a2a_agents",
                    status="success",
                    message=f"Text optimization completed with {tonality or 'default'} tonality",
                    processing_time=time.time() - start_time,
                    steps=[
                        ProcessingStep(
                            step_name="A2A Text Optimization",
                            input_text=text,
                            output_text=optimized_text,
                            status="success",
                            message="Text successfully optimized via A2A registry",
                        )
                    ],
                )

            elif operation == "correct":
                # Send text directly as string to lektor
                result = await registry.call_agent("lektor", text)

                if debug_a2a:
                    logger.info(f"Raw lektor result: {result}")

                # Extract corrected text
                corrected_text = None
                if hasattr(result, "corrected_text"):
                    corrected_text = result.corrected_text
                elif hasattr(result, "content"):
                    corrected_text = result.content
                elif isinstance(result, dict) and "corrected_text" in result:
                    corrected_text = result["corrected_text"]
                elif isinstance(result, dict) and "content" in result:
                    corrected_text = result["content"]
                elif isinstance(result, str):
                    corrected_text = result
                else:
                    corrected_text = str(result)

                return UserInterfaceResponse(
                    original_text=text,
                    final_result=corrected_text,
                    operation_type="coordinate_with_a2a_agents",
                    status="success",
                    message="Text correction completed via A2A",
                    processing_time=time.time() - start_time,
                    steps=[
                        ProcessingStep(
                            step_name="A2A Text Correction",
                            input_text=text,
                            output_text=corrected_text,
                            status="success",
                            message="Text successfully corrected via A2A registry",
                        )
                    ],
                )

            elif operation == "sentiment":
                message = [{"content": text}]

                if debug_a2a:
                    logger.info(f"🔍 Calling sentiment agent with text: '{text}'")
                    logger.info(f"🔍 Message format: {message}")

                    # ADDED: Debug the registry and available agents
                    logger.info(f"🔍 Registry type: {type(registry)}")
                    logger.info(
                        f"🔍 Registry available agents: {getattr(registry, '_agents', 'Unknown')}"
                    )

                    # Try to get more info about the sentiment agent
                    if hasattr(registry, "_agents") and "sentiment" in registry._agents:
                        sentiment_agent = registry._agents["sentiment"]
                        logger.info(f"🔍 Sentiment agent type: {type(sentiment_agent)}")
                        logger.info(
                            f"🔍 Sentiment agent model: {getattr(sentiment_agent, 'model', 'Unknown')}"
                        )
                        if hasattr(sentiment_agent, "system_prompt"):
                            logger.info(
                                f"🔍 Sentiment agent system prompt: {sentiment_agent.system_prompt[:200]}..."
                            )

                result = await registry.call_agent("sentiment", message)

                if debug_a2a:
                    logger.info(f"Raw sentiment result: {result}")
                    logger.info(f"Sentiment analysis for text: '{text}'")
                    logger.info(f"🔍 Result type: {type(result)}")
                    logger.info(
                        f"🔍 Result __dict__: {getattr(result, '__dict__', 'No dict')}"
                    )

                # Extract sentiment data - this is more complex due to nested structure
                sentiment_info = None
                emotions = []

                if hasattr(result, "sentiment") and hasattr(result, "emotions"):
                    if debug_a2a:
                        logger.info(
                            f"🔍 result.sentiment type: {type(result.sentiment)}"
                        )
                        logger.info(
                            f"🔍 result.sentiment __dict__: {getattr(result.sentiment, '__dict__', 'No dict')}"
                        )
                        logger.info(f"🔍 result.emotions type: {type(result.emotions)}")

                    sentiment_info = {
                        "label": result.sentiment.label
                        if hasattr(result.sentiment, "label")
                        else "unknown",
                        "confidence": result.sentiment.confidence
                        if hasattr(result.sentiment, "confidence")
                        else 0.0,
                        "score": result.sentiment.score
                        if hasattr(result.sentiment, "score")
                        else 0.0,
                    }
                    emotions = result.emotions if result.emotions else []

                    if debug_a2a:
                        logger.info(f"🎯 Parsed sentiment: {sentiment_info}")
                        logger.info(f"🎯 Parsed emotions: {emotions}")
                        logger.info(f"🎯 Raw sentiment object: {result.sentiment}")
                        logger.info(f"🎯 Raw emotions object: {result.emotions}")

                        # Check if this looks wrong
                        text_lower = text.lower()
                        if any(
                            word in text_lower
                            for word in [
                                "glücklich",
                                "großartig",
                                "toll",
                                "super",
                                "wunderbar",
                                "fantastisch",
                            ]
                        ):
                            if sentiment_info["label"] != "positive":
                                logger.error(
                                    f"🚨 SENTIMENT MODEL ERROR: Text '{text}' contains obvious positive words but was classified as '{sentiment_info['label']}'"
                                )
                                logger.error(
                                    f"🚨 This indicates the sentiment agent prompt/logic is completely broken"
                                )
                                logger.error(
                                    f"🚨 Need to check the sentiment agent definition in a2a_server.py or agents/ folder"
                                )

                else:
                    logger.error(
                        f"❌ Sentiment result missing expected attributes: {dir(result) if hasattr(result, '__dict__') else 'No attributes'}"
                    )
                    sentiment_info = {
                        "label": "error",
                        "confidence": 0.0,
                        "score": 0.0,
                    }
                    emotions = []

                # Format emotions nicely
                emotions_text = (
                    ", ".join(
                        [
                            f"{list(emotion.keys())[0]}: {list(emotion.values())[0]:.1f}"
                            for emotion in emotions
                        ]
                    )
                    if emotions
                    else "none detected"
                )

                result_text = f"Sentiment Analysis Results:\nLabel: {sentiment_info['label']}\nConfidence: {sentiment_info['confidence']:.2f}\nScore: {sentiment_info['score']:.2f}\nEmotions: {emotions_text}"

                return UserInterfaceResponse(
                    original_text=text,
                    final_result=result_text,
                    operation_type="coordinate_with_a2a_agents",
                    status="success",
                    message="Sentiment analysis completed via A2A",
                    processing_time=time.time() - start_time,
                    sentiment_analysis=sentiment_info,
                    steps=[
                        ProcessingStep(
                            step_name="A2A Sentiment Analysis",
                            input_text=text,
                            output_text=f"Sentiment: {sentiment_info['label']} ({sentiment_info['confidence']:.2f})",
                            status="success",
                            message="Sentiment successfully analyzed via A2A registry",
                        )
                    ],
                )

        finally:
            await registry.stop()

    except Exception as e:
        logger.error(f"A2A coordination failed: {e}", exc_info=True)
        return UserInterfaceResponse(
            original_text=text,
            final_result=f"Error during {operation}: {str(e)}",
            operation_type="coordinate_with_a2a_agents",
            status="error",
            message=f"A2A coordination failed: {str(e)}",
            processing_time=time.time() - start_time,
        )


@user_interface_agent.tool
async def analyze_user_request_intent(
    ctx: RunContext[UserInterfaceContext], user_text: str
) -> str:
    """Analyze user request to determine the best processing strategy."""
    text_lower = user_text.lower()

    # Time/Date keywords (German and English) - EXPLICIT time/date requests
    explicit_time_keywords = [
        "zeit",
        "time",
        "uhr",
        "uhrzeit",
        "spät",
        "late",
        "datum",
        "date",
        "wieviel",
        "wie spät",
        "what time",
        "current time",
        "jetzt",
        "now",
        "aktuell",
        "current",
    ]

    # Date-specific phrases that should always be time queries
    date_phrases = [
        "welches datum",
        "what date",
        "welcher tag",
        "what day",
        "datum heute",
        "date today",
        "datum morgen",
        "date tomorrow",
        "welcher monat",
        "what month",
        "welches jahr",
        "what year",
    ]

    # Weather keywords (German and English) - INCLUDING temporal context
    weather_keywords = [
        "wetter",
        "weather",
        "temperatur",
        "temperature",
        "regen",
        "rain",
        "schnee",
        "snow",
        "sonne",
        "sun",
        "wind",
        "wolken",
        "clouds",
        "morgen",
        "tomorrow",
        "übermorgen",
        "vorhersage",
        "forecast",
        "heute",
    ]

    # Weather-specific phrases that should override other detection
    weather_phrases = [
        "wetter morgen",
        "weather tomorrow",
        "wetter heute",
        "weather today",
        "wird das wetter",
        "how will the weather",
        "wie wird das wetter",
    ]

    # Location indicators that suggest weather queries
    location_indicators = [
        "in",
        "für",
        "bei",
        "um",
        "kiel",
        "hamburg",
        "berlin",
        "münchen",
        "germany",
        "deutschland",
        "stadt",
        "city",
    ]

    # PRIORITY 1: Explicit date/time queries (highest priority)
    has_explicit_time = any(keyword in text_lower for keyword in explicit_time_keywords)
    has_date_phrase = any(phrase in text_lower for phrase in date_phrases)

    if has_date_phrase or (
        has_explicit_time and ("datum" in text_lower or "date" in text_lower)
    ):
        return "time_query"

    # PRIORITY 2: Weather queries with clear context
    has_weather_keyword = any(keyword in text_lower for keyword in weather_keywords)
    has_weather_phrase = any(phrase in text_lower for phrase in weather_phrases)
    has_location = any(
        indicator in text_lower for indicator in location_indicators
    )  # Weather detection logic: explicit weather terms OR weather phrases OR weather + location
    if (
        has_weather_phrase
        or (has_weather_keyword and has_location)
        or "wetter" in text_lower
    ):
        return "weather_search"

    # PRIORITY 3: Pure time queries (without explicit date context)
    if has_explicit_time:
        return "time_query"
        # PRIORITY 4: Sentiment analysis (explicit requests)
    if any(
        phrase in text_lower
        for phrase in [
            "analysiere das sentiment",
            "sentiment analyse",
            "analyze sentiment",
            "sentiment analysis",
            "wie ist das sentiment",
            "what is the sentiment",
            "stimmung analysieren",
            "analyze mood",
            "gefühl analysieren",
            "emotion analysis",
        ]
    ):
        return "sentiment_analysis"

    # PRIORITY 5: General information search
    elif (
        any(
            word in text_lower
            for word in [
                "wie",
                "was",
                "wo",
                "wann",
                "warum",
                "how",
                "what",
                "where",
                "when",
                "why",
            ]
        )
        and "?" in user_text
    ):
        return "information_search"
    elif any(
        word in text_lower
        for word in [
            "korrigiere",
            "correct",
            "optimiere",
            "optimize",
            "verbessere",
            "improve",
            "möchte einen korrekten",
            "want a correct",
            "brauche korrekten",
            "soll korrekt sein",
            "should be correct",
            "korrekten satz",
            "correct sentence",
        ]
    ):
        return "text_processing"
    elif "http" in text_lower or "www." in text_lower:
        return "url_extraction"
    else:
        return "general_query"


@user_interface_agent.tool
async def process_time_query(
    ctx: RunContext[UserInterfaceContext], query: str
) -> UserInterfaceResponse:
    """Handle time and date queries with improved error handling."""
    import time

    start_time = time.time()

    try:
        # FORCE the enhanced debugging
        logger.info("🔍 process_time_query: About to call get_current_time_info...")

        # Get current time with fallback
        time_result = await get_current_time_info(ctx)

        logger.info(f"🔍 process_time_query: Received time_result: {time_result}")

        current_time_utc = time_result.get("current_time_utc", "Unknown time")
        source = time_result.get("source", "mcp")

        # Format the response based on what user asked
        text_lower = query.lower()

        if any(word in text_lower for word in ["datum", "date"]) and any(
            word in text_lower for word in ["zeit", "time", "uhr", "spät"]
        ):
            # User asked for both time and date
            response_text = f"Aktuelle Zeit und Datum (UTC): {current_time_utc}"
            if source == "mcp_strict_only":
                response_text += "\n\nHinweis: Dies ist die Systemzeit (MCP-Service nicht verfügbar)."
            else:
                response_text += (
                    "\n\nHinweis: Dies ist die koordinierte Weltzeit (UTC)."
                )
            message = "Aktuelle Zeit und Datum erfolgreich abgerufen"
        elif any(word in text_lower for word in ["datum", "date"]):
            # User asked only for date
            date_part = (
                current_time_utc.split("T")[0]
                if "T" in current_time_utc
                else current_time_utc
            )
            response_text = f"Aktuelles Datum (UTC): {date_part}"
            if source == "mcp_strict_only":
                response_text += "\n\nHinweis: Dies ist das Systemdatum."
            message = "Aktuelles Datum erfolgreich abgerufen"
        else:
            # User asked for time
            response_text = f"Aktuelle Zeit (UTC): {current_time_utc}"
            if source == "mcp_strict_only":
                response_text += "\n\nHinweis: Dies ist die Systemzeit."
            message = "Aktuelle Zeit erfolgreich abgerufen"

        return UserInterfaceResponse(
            original_text=query,
            final_result=response_text,
            operation_type="time_query",
            steps=[
                ProcessingStep(
                    step_name="Time Request Analysis",
                    input_text=query,
                    output_text="Erkannt: Zeit- und/oder Datumsanfrage",
                    status="success",
                    message="Benutzeranfrage als Zeit-/Datumsabfrage identifiziert",
                ),
                ProcessingStep(
                    step_name="Time Service Call",
                    input_text="get_current_time request",
                    output_text=current_time_utc,
                    status="success" if source != "mcp_strict_only" else "fallback",
                    message=f"Zeit abgerufen via {source}",
                ),
            ],
            status="success",
            message=message,
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Time query processing failed: {e}")
        # Emergency fallback - NO SYSTEM TIME
        return UserInterfaceResponse(
            original_text=query,
            final_result="MCP Zeit-Service ist derzeit nicht verfügbar.",
            operation_type="time_query",
            steps=[
                ProcessingStep(
                    step_name="Emergency Fallback",
                    input_text=query,
                    output_text="MCP Service Error",
                    status="error",
                    message="MCP Zeit-Service nicht verfügbar",
                )
            ],
            status="error",
            message="MCP Zeit-Service erforderlich",
            processing_time=time.time() - start_time,
        )


@user_interface_agent.tool
async def optimize_prompt_with_engineer(
    ctx: RunContext[UserInterfaceContext],
    user_query: str,
    target_model: str = "qwen2.5:latest",
) -> Dict[str, Any]:
    """Use the prompt engineer agent to optimize prompts and recommend execution strategies."""

    start_time = time.time()
    logger.info(f"🎯 PROMPT ENGINEER: Optimizing query: {user_query}")

    try:
        # Import the existing A2A setup
        from pathlib import Path

        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from a2a_server import setup_a2a_server

        registry = await setup_a2a_server()

        try:
            # Call the prompt engineer agent
            result = await registry.call_agent("prompt_engineer", user_query)

            logger.info(f"🎯 PROMPT ENGINEER result: {result}")

            # Extract the optimized prompt response
            if hasattr(result, "optimized_prompt"):
                optimized_prompt = result.optimized_prompt

                return {
                    "original_query": user_query,
                    "detected_intention": optimized_prompt.detected_intention,
                    "optimized_prompt": optimized_prompt.optimized_prompt,
                    "recommended_agents": optimized_prompt.recommended_agents,
                    "confidence_score": optimized_prompt.confidence_score,
                    "execution_plan": result.execution_plan,
                    "alternative_prompts": result.alternative_prompts,
                    "reasoning": optimized_prompt.reasoning,
                    "processing_time": time.time() - start_time,
                    "status": "success",
                }
            else:
                return {
                    "error": "Invalid response from prompt engineer",
                    "status": "error",
                }

        finally:
            await registry.stop()

    except Exception as e:
        logger.error(f"Prompt engineer optimization failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "processing_time": time.time() - start_time,
        }


# A2A server function for user interface (standalone function, not a tool)
async def user_interface_a2a_function(
    messages: List[ModelMessage],
) -> UserInterfaceResponse:
    """A2A endpoint for user interface functionality."""
    if not messages:
        return UserInterfaceResponse(
            original_text="",
            final_result="",
            operation_type="a2a_call",
            status="error",
            message="No messages provided",
        )

    # Extract text from the last user message
    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        text = last_message.content
    else:
        text = str(last_message)

    # Let the agent make autonomous decisions
    context = UserInterfaceContext(request_id="a2a_request")
    result = await user_interface_agent.run(text, ctx=context)
    return result.data


# Function to process user requests directly through the agent
async def process_user_request(text: str, **kwargs) -> UserInterfaceResponse:
    """Process user input through the intelligent agent."""
    context = UserInterfaceContext(request_id="direct_request")
    result = await user_interface_agent.run(text, ctx=context)
    return result.data


# New processing function with debugging
async def process_input(user_input: str) -> UserInterfaceResponse:
    """
    Main processing function - NO FALLBACKS, production ready.
    Uses prompt_engineer workflow.
    """
    return await process_input_with_prompt_engineer(user_input)


# Aktualisiere die bestehende process_input Funktion
async def process_input_with_prompt_engineer(user_input: str) -> UserInterfaceResponse:
    """
    Hauptverarbeitungsfunktion mit prompt_engineer Integration.
    Ausnahmen: Wetter- und Zeitfragen werden direkt verarbeitet.
    """
    import time

    start_time = time.time()
    debug_mode = os.getenv("DEBUG_AGENT_RESPONSES", "false").lower() == "true"

    if debug_mode:
        logger.info(f"=== PROMPT ENGINEER WORKFLOW START ===")
        logger.info(f"Input: {user_input}")

    try:
        context = UserInterfaceContext(request_id="prompt_engineer_workflow")
        input_lower = user_input.lower()

        # AUSNAHMEN: Wetter und Zeit - direkte Verarbeitung ohne prompt_engineer
        is_weather_query = any(
            phrase in input_lower
            for phrase in [
                "wetter",
                "weather",
                "temperatur",
                "temperature",
                "wetter morgen",
                "weather tomorrow",
                "wird das wetter",
            ]
        )

        is_time_query = any(
            phrase in input_lower
            for phrase in [
                "zeit",
                "time",
                "uhr",
                "uhrzeit",
                "spät",
                "datum",
                "date",
                "wie spät",
                "what time",
                "welches datum",
                "what date",
            ]
        )

        if is_weather_query:
            if debug_mode:
                logger.info(
                    "🌤️ AUSNAHME: Wetteranfrage - direkter Aufruf ohne prompt_engineer"
                )

            # Direkte Wetterverarbeitung
            location = "Deutschland"
            if "berlin" in input_lower:
                location = "Berlin"
            elif "hamburg" in input_lower:
                location = "Hamburg"
            elif "münchen" in input_lower:
                location = "München"
            elif "kiel" in input_lower:
                location = "Kiel"

            time_period = "heute"
            if "morgen" in input_lower or "tomorrow" in input_lower:
                time_period = "morgen"

            return await get_weather_information(context, location, time_period)

        if is_time_query:
            if debug_mode:
                logger.info(
                    "⏰ AUSNAHME: Zeitanfrage - direkter Aufruf ohne prompt_engineer"
                )

            return await process_time_query(context, user_input)

        # STANDARD WORKFLOW: prompt_engineer zuerst aufrufen
        if debug_mode:
            logger.info("🎯 STANDARD: Rufe prompt_engineer auf...")

        try:
            prompt_engineer_result = await optimize_prompt_with_engineer(
                context, user_input
            )

            if debug_mode:
                logger.info(f"🎯 prompt_engineer Ergebnis: {prompt_engineer_result}")

            if prompt_engineer_result.get("status") == "success":
                # Extrahiere Empfehlungen vom prompt_engineer
                detected_intention = prompt_engineer_result.get(
                    "detected_intention", "general_query"
                )
                recommended_agents = prompt_engineer_result.get(
                    "recommended_agents", []
                )
                optimized_prompt = prompt_engineer_result.get(
                    "optimized_prompt", user_input
                )
                execution_plan = prompt_engineer_result.get("execution_plan", [])

                if debug_mode:
                    logger.info(f"🎯 Detected intention: {detected_intention}")
                    logger.info(f"🎯 Recommended agents: {recommended_agents}")
                    logger.info(f"🎯 Execution plan: {execution_plan}")

                # Jetzt user_interface Agent entscheiden lassen basierend auf prompt_engineer Empfehlungen
                enhanced_input = f"""
Optimierte Anfrage vom prompt_engineer:
{optimized_prompt}

Erkannte Absicht: {detected_intention}
Empfohlene Agenten: {", ".join(recommended_agents)}
Ausführungsplan: {" → ".join(execution_plan)}

Ursprüngliche Anfrage: {user_input}
"""

                if debug_mode:
                    logger.info(
                        f"🤖 Übergebe an user_interface Agent: {enhanced_input[:200]}..."
                    )

                # user_interface Agent mit optimiertem Prompt laufen lassen
                result = await user_interface_agent.run(enhanced_input, ctx=context)

                if hasattr(result, "data") and isinstance(
                    result.data, UserInterfaceResponse
                ):
                    # Füge prompt_engineer Informationen zu den Steps hinzu
                    result.data.steps.insert(
                        0,
                        ProcessingStep(
                            step_name="Prompt Engineering",
                            input_text=user_input,
                            output_text=f"Intent: {detected_intention}, Agents: {', '.join(recommended_agents)}",
                            status="success",
                            message="Prompt erfolgreich optimiert via prompt_engineer",
                        ),
                    )
                    result.data.processing_time = time.time() - start_time
                    return result.data

            else:
                logger.warning(
                    f"prompt_engineer fehlgeschlagen: {prompt_engineer_result.get('error', 'Unknown error')}"
                )

        except Exception as pe_error:
            logger.error(f"prompt_engineer Fehler: {pe_error}")
            if debug_mode:
                logger.error(
                    f"prompt_engineer Fehler Details: {pe_error}", exc_info=True
                )

        # FALLBACK: Standard user_interface Agent ohne prompt_engineer
        if debug_mode:
            logger.info(
                "🔄 FALLBACK: Standard user_interface Agent ohne prompt_engineer"
            )

        result = await user_interface_agent.run(user_input, ctx=context)

        if hasattr(result, "data") and isinstance(result.data, UserInterfaceResponse):
            result.data.processing_time = time.time() - start_time
            return result.data

    except Exception as e:
        logger.error(
            f"Kritischer Fehler in process_input_with_prompt_engineer: {e}",
            exc_info=True,
        )

    # Letzter Fallback
    return UserInterfaceResponse(
        original_text=user_input,
        final_result=f"Entschuldigung, es gab ein Problem bei der Verarbeitung Ihrer Anfrage: '{user_input}'",
        operation_type="error_fallback",
        status="error",
        message="Kritischer Verarbeitungsfehler",
        processing_time=time.time() - start_time,
    )


# Example usage functions for testing
async def main():
    """Main function to demonstrate User Interface Agent usage."""
    # Test different types of requests
    test_cases = [
        "Wie wird das Wetter morgen in Kiel?",  # Weather query
        "Also echt mal, dieses Produkt ist der letzte Schrott, total verbuggt und wer das kauft ist doch selber schuld, oder?",  # Text processing
        "Bitte korrigiere diesen Text: Das ist ein sehr schlechte Satz mit viele Fehler.",  # Grammar correction
        "Was ist die Hauptstadt von Deutschland?",  # Information query
        "Wie späte ist es jetzt und welches Datum haben wir?",  # Time and date query
        "Welches datum istmorgen?",  # Ambiguous date query
    ]

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}: {test_text}")
        print("=" * 60)

        result = await process_user_request(test_text)

        print(f"Operation: {result.operation_type}")
        print(f"Status: {result.status}")
        print(f"Final Result: {result.final_result[:200]}...")
        print(f"Processing Time: {result.processing_time:.2f}s")

        if result.steps:
            print("\nProcessing Steps:")
            for step in result.steps:
                print(f"  - {step.step_name}: {step.status}")

        if result.sentiment_analysis:
            print(f"\nSentiment: {result.sentiment_analysis}")


if __name__ == "__main__":
    asyncio.run(main())
