import os
from pathlib import Path
from typing import Dict, Any
import json
from dotenv import load_dotenv, dotenv_values


class ConfigManager:
    """
    Centralized configuration manager for modular Agent System.
    Loads main .env and specialized configuration files (A2AP, MCP, Prompt Engineer).
    """

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.config: Dict[str, Any] = {}
        self._load_configurations()

    def _load_configurations(self):
        """Load main configuration and modular configs"""
        # Load main .env file
        main_env_path = self.base_path / ".env"
        if main_env_path.exists():
            load_dotenv(main_env_path)
            self.config.update(dotenv_values(main_env_path))

        # Load A2AP configuration
        a2ap_config_file = os.getenv("A2AP_CONFIG_FILE", ".env.a2ap")
        a2ap_path = self.base_path / a2ap_config_file
        if a2ap_path.exists():
            a2ap_config = dotenv_values(a2ap_path)
            self.config.update(a2ap_config)
            self._set_env_vars(a2ap_config)

        # Load MCP configuration
        mcp_config_file = os.getenv("MCP_CONFIG_FILE", ".env.mcp")
        mcp_path = self.base_path / mcp_config_file
        if mcp_path.exists():
            mcp_config = dotenv_values(mcp_path)
            self.config.update(mcp_config)
            self._set_env_vars(mcp_config)

        # Load Prompt Engineer configuration
        pe_config_file = os.getenv(
            "PROMPT_ENGINEER_CONFIG_FILE", ".env.prompt_engineer"
        )
        pe_path = self.base_path / pe_config_file
        if pe_path.exists():
            pe_config = dotenv_values(pe_path)
            self.config.update(pe_config)
            self._set_env_vars(pe_config)

    def _set_env_vars(self, config_dict: Dict[str, str]):
        """Set environment variables from config dictionary"""
        for key, value in config_dict.items():
            if key and value:
                os.environ[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to environment variable"""
        return self.config.get(key, os.getenv(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        value = self.get(key, str(default))
        return str(value).lower() in ("true", "1", "yes", "on")

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value"""
        try:
            return float(self.get(key, default))
        except (ValueError, TypeError):
            return default

    def get_json(self, key: str, default: Dict = None) -> Dict:
        """Get JSON configuration value"""
        if default is None:
            default = {}
        try:
            value = self.get(key)
            if isinstance(value, str):
                return json.loads(value)
            return value if value else default
        except (json.JSONDecodeError, TypeError):
            return default

    def get_a2ap_config(self) -> Dict[str, Any]:
        """Get all A2AP related configuration"""
        return {k: v for k, v in self.config.items() if k.startswith("A2AP_")}

    def get_mcp_config(self) -> Dict[str, Any]:
        """Get all MCP related configuration"""
        return {k: v for k, v in self.config.items() if k.startswith("MCP_")}

    def get_prompt_engineer_config(self) -> Dict[str, Any]:
        """Get all Prompt Engineer related configuration"""
        return {k: v for k, v in self.config.items() if k.startswith("PE_")}

    def get_agent_model(self, agent_type: str) -> str:
        """Get model for specific agent type"""
        model_key = f"{agent_type.upper()}_MODEL"
        return self.get(model_key, self.get("OPTIMIZER_MODEL", "qwen2.5:latest"))

    def is_a2ap_enabled(self) -> bool:
        """Check if A2AP is enabled"""
        return self.get_bool("A2AP_ENABLED", True)

    def is_mcp_enabled(self) -> bool:
        """Check if MCP is enabled"""
        return self.get_bool("MCP_ENABLED", True)

    def is_prompt_engineer_enabled(self) -> bool:
        """Check if Prompt Engineer is enabled"""
        return self.get_bool("PE_ENABLED", True)

    def reload(self):
        """Reload all configurations"""
        self.config.clear()
        self._load_configurations()


# Global configuration instance
config = ConfigManager()