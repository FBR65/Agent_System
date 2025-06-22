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

    async def call_agent(self, agent_name: str, input_data: str) -> Any:
        """Call agent - EMERGENCY WORKING VERSION"""
        print(f"A2A Registry: Calling {agent_name}")

        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent_func = self.agents[agent_name]

        try:
            # EMERGENCY: Just call the A2A wrapper functions directly
            return await agent_func(input_data)

        except Exception as e:
            logger.error(f"Error calling agent {agent_name}: {e}")
            raise

    async def stop(self):
        """Stop the registry"""
        logger.info("A2A Registry stopped")


# Global registry instance
registry = FixedA2ARegistry()


async def setup_a2a_server():
    """Setup A2A server - CURRENT VERSION"""

    # Register available agents using their A2A wrapper functions
    try:
        from agent_server.optimizer import optimizer_a2a_function

        registry.register_agent("optimizer", optimizer_a2a_function)
    except Exception as e:
        print(f"Warning: optimizer not available: {e}")

    try:
        from agent_server.lektor import lektor_a2a_function

        registry.register_agent("lektor", lektor_a2a_function)
    except Exception as e:
        print(f"Warning: lektor not available: {e}")

    try:
        from agent_server.sentiment import sentiment_a2a_function

        registry.register_agent("sentiment", sentiment_a2a_function)
    except Exception as e:
        print(f"Warning: sentiment not available: {e}")

    try:
        from agent_server.query_ref import query_ref_a2a_function

        registry.register_agent("query_ref", query_ref_a2a_function)
    except Exception as e:
        print(f"Warning: query_ref not available: {e}")

    try:
        from agent_server.prompt_engineer import run_prompt_engineer

        async def prompt_wrapper(text):
            from agent_server.prompt_engineer import PromptEngineerRequest

            request = PromptEngineerRequest(user_input=text)
            return await run_prompt_engineer(request)

        registry.register_agent("prompt_engineer", prompt_wrapper)
    except Exception as e:
        print(f"Warning: prompt_engineer not available: {e}")

    return registry


if __name__ == "__main__":
    print("A2A Server ready")
