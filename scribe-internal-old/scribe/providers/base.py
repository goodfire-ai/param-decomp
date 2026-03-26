"""Base provider class for AI CLI integrations."""

from abc import ABC, abstractmethod


class AICLIProvider(ABC):
    """Abstract base class for AI CLI providers."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'claude', 'gemini')."""
        pass

    @abstractmethod
    def get_provider_display_name(self) -> str:
        """Return the provider display name (e.g., 'Claude Code CLI', 'Gemini CLI')."""
        pass

    # @abstractmethod
    # def get_command_base(self) -> List[str]:
    #     """Return the base command for the CLI."""
    #     pass

    # @abstractmethod
    # def get_settings_dir(self) -> str:
    #     """Return the settings directory name (e.g., '.claude', '.gemini')."""
    #     pass

    # @abstractmethod
    # def get_settings_filename(self) -> str:
    #     """Return the settings file name (e.g., 'settings.json', 'config.json')."""
    #     pass

    # @abstractmethod
    # def get_settings_content(self) -> Dict[str, Any]:
    #     """Return the settings content for this provider."""
    #     pass

    # @abstractmethod
    # def build_copilot_command(self, args: List[str]) -> List[str]:
    #     """Build command for copilot mode."""
    #     pass

    # @abstractmethod
    # def build_agent_command(self, prompt: str, args: List[str]) -> List[str]:
    #     """Build command for agent mode with a planning prompt."""
    #     pass

    # @abstractmethod
    # def build_chat_command(self, session_id: str, prompt: Optional[str] = None) -> List[str]:
    #     """Build command for chat/resume mode."""
    #     pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this AI CLI is available on the system."""
        pass

    # def get_blueprint_suffix(self) -> str:
    #     """Return suffix for provider-specific blueprint directories."""
    #     return f"_{self.get_provider_name()}"

    # @abstractmethod
    # def get_instructions_template_name(self) -> str:
    #     """Return the name of the instructions template file."""
    #     pass

    # @abstractmethod
    # def get_instructions_file_name(self) -> str:
    #     """Return the name of the instructions file to create."""
    #     pass

    # @abstractmethod
    # def get_settings_template_name(self) -> str:
    #     """Return the name of the settings template file."""
    #     pass

    # @abstractmethod
    # def supports_mcp_config_flag(self) -> bool:
    #     """Return whether this provider supports --mcp-config flag."""
    #     pass

    # @abstractmethod
    # def setup_settings(self, mcp_config: Optional[Dict[str, Any]] = None) -> None:
    #     """Create or update AI CLI settings for this provider.

    #     Each provider must implement this method to handle their specific
    #     settings creation and update logic.
    #     """
    #     pass
