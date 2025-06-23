#!/usr/bin/env python3
"""
A2A Server - CURRENT WORKING VERSION
"""

import asyncio
import logging
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)


class FixedA2ARegistry:
    """Emergency A2A Registry - WORKING VERSION"""

    def __init__(self):
        self.agents: Dict[str, Callable] = {}

    def register_agent(self, name: str, agent_func: Callable):
        """Register an agent function"""
        self.agents[name] = agent_func
        logger.info(f"Agent {name} registered successfully")

    async def call_agent(self, agent_name: str, input_data) -> Any:
        """Call a specific agent with input data."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent_func = self.agents[agent_name]

        # DEBUG: Verbesserte Logging für A2A-Übertragung
        logger.info(f"A2A Registry: Calling {agent_name}")
        logger.info(f"A2A Registry: Input data type: {type(input_data)}")
        logger.info(f"A2A Registry: Input data: '{input_data}'")

        # Convert input data to message format - FIXED: Use correct message type
        try:
            # Try to import the correct UserMessage class
            try:
                from pydantic_ai.messages import UserMessage
            except ImportError:
                try:
                    # Alternative: Use a direct message creation
                    class UserMessage:
                        def __init__(self, content: str):
                            self.content = content
                            self.role = "user"
                except:
                    # Final fallback: Simple message class
                    class UserMessage:
                        def __init__(self, content: str):
                            self.content = content
                            self.role = "user"

            if isinstance(input_data, str):
                messages = [UserMessage(content=input_data)]
            elif isinstance(input_data, list):
                messages = []
                for item in input_data:
                    if isinstance(item, dict) and "content" in item:
                        messages.append(UserMessage(content=item["content"]))
                    else:
                        messages.append(UserMessage(content=str(item)))
            else:
                messages = [UserMessage(content=str(input_data))]

            # DEBUG: Überprüfe die erstellten Messages
            logger.info(f"A2A Registry: Created {len(messages)} proper UserMessage(s)")
            for i, msg in enumerate(messages):
                logger.info(
                    f"A2A Registry: Message {i}: '{msg.content}' (type: {type(msg)})"
                )

            return await agent_func(messages)

        except Exception as e:
            logger.error(f"❌ Message creation failed: {e}")

            # Fallback: Create simple message objects
            class SimpleMessage:
                def __init__(self, content):
                    self.content = content
                    self.role = "user"

            if isinstance(input_data, str):
                messages = [SimpleMessage(input_data)]
            elif isinstance(input_data, list):
                messages = []
                for item in input_data:
                    if isinstance(item, dict) and "content" in item:
                        messages.append(SimpleMessage(item["content"]))
                    else:
                        messages.append(SimpleMessage(str(item)))
            else:
                messages = [SimpleMessage(str(input_data))]

            logger.info(f"A2A Registry: Created {len(messages)} simple message(s)")
            return await agent_func(messages)

    async def stop(self):
        """Stop the registry"""
        logger.info("A2A Registry stopped")


# Global registry instance
registry = FixedA2ARegistry()


async def setup_a2a_server():
    """Setup A2A server - CURRENT VERSION"""

    logger.info("🔧 A2A Server Setup starting...")

    # Register available agents using their A2A wrapper functions
    try:
        logger.info("🔍 Attempting to import optimizer...")
        from agent_server.optimizer import optimizer_a2a_function

        logger.info("✅ Optimizer imported successfully")
        registry.register_agent("optimizer", optimizer_a2a_function)
        logger.info("✅ Agent optimizer registered successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import optimizer: {e}")
        print(f"Warning: optimizer not available: {e}")
        # Try to give more details about the import error
        import traceback

        logger.error(f"Import traceback: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"❌ Failed to register optimizer: {e}")
        print(f"Warning: optimizer registration failed: {e}")
        import traceback

        logger.error(f"Registration traceback: {traceback.format_exc()}")

    try:
        logger.info("🔍 Attempting to import lektor...")
        from agent_server.lektor import lektor_a2a_function

        logger.info("✅ Lektor imported successfully")
        registry.register_agent("lektor", lektor_a2a_function)
        logger.info("✅ Agent lektor registered successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import lektor: {e}")
        print(f"Warning: lektor not available: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to register lektor: {e}")
        print(f"Warning: lektor registration failed: {e}")

    try:
        from agent_server.sentiment import sentiment_a2a_function

        registry.register_agent("sentiment", sentiment_a2a_function)
    except Exception as e:
        print(f"Warning: sentiment not available: {e}")

    try:
        logger.info("🔍 Attempting to import query_ref...")
        from agent_server.query_ref import query_ref_a2a_function

        logger.info("✅ Query_ref imported successfully")
        registry.register_agent("query_ref", query_ref_a2a_function)
        logger.info("✅ Agent query_ref registered successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import query_ref: {e}")
        print(f"Warning: query_ref not available: {e}")
        import traceback

        logger.error(f"Import traceback: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"❌ Failed to register query_ref: {e}")
        print(f"Warning: query_ref registration failed: {e}")
        import traceback

        logger.error(f"Registration traceback: {traceback.format_exc()}")

    try:
        from agent_server.prompt_engineer import run_prompt_engineer

        async def prompt_wrapper(messages):
            from agent_server.prompt_engineer import PromptEngineerRequest

            # Extract text from messages - IMPROVED extraction
            if isinstance(messages, list) and len(messages) > 0:
                last_message = messages[-1]

                # Handle different message types
                if hasattr(last_message, "content") and isinstance(
                    last_message.content, str
                ):
                    text = last_message.content
                elif hasattr(last_message, "content"):
                    text = str(last_message.content)
                else:
                    text = str(last_message)
            elif isinstance(messages, str):
                text = messages
            else:
                text = str(messages)

            logger.info(f"🎯 PROMPT ENGINEER: Extracted text: '{text}'")

            request = PromptEngineerRequest(user_input=text)
            return await run_prompt_engineer(request)

        registry.register_agent("prompt_engineer", prompt_wrapper)
    except Exception as e:
        print(f"Warning: prompt_engineer not available: {e}")

    # Log final registry state
    logger.info(
        f"🎯 A2A Registry Setup Complete. Registered agents: {list(registry.agents.keys())}"
    )

    return registry


if __name__ == "__main__":
    print("A2A Server ready")
