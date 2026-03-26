"""
Scribe Convo MCP Server
"""

from fastmcp import FastMCP
from pathlib import Path
from dotenv import load_dotenv
from scribe.extensions import _chat_log_utils as _chat
from typing import Dict, Any

# Initialize MCP server
mcp = FastMCP("scribe-convo")


@mcp.tool
async def get_most_recent_convo(project_path: str = None) -> Dict[str, Any]:
    """
    Get the most recent conversation between the user and an AI agent related to the
    project with `project_path`. Returns the most recent conversation across all projects
    if `project_path` is not provided.
    
    Returns:
        Dict containing:
        - transcript: The conversation transcript
        - session_id: The session ID of the conversation (if available)
        - last_line_index: The index of the final line that was read
    """
    conversation_filepath = _chat._most_recent_claude_convo_path(project_path)

    if not conversation_filepath:
        return {"transcript": "No conversation found.", "session_id": None, "last_line_index": -1}

    # Use incremental reader to get both transcript and last line (start from beginning with -1)
    conversation_transcript, last_line_index = _chat._process_claude_convo_incremental(conversation_filepath, -1)
    session_id = _chat._get_session_id_from_convo(conversation_filepath)

    if not conversation_transcript:
        return {"transcript": "Could not process the conversation transcript.", "session_id": session_id, "last_line_index": last_line_index}

    return {"transcript": conversation_transcript, "session_id": session_id, "last_line_index": last_line_index}


@mcp.tool
async def get_convo_by_session_id(session_id: str, start_line: int = -1) -> Dict[str, Any]:
    """
    Get the conversation between the user and Claude Code for the given session_id.
    
    Args:
        session_id: The session ID to retrieve the conversation for
        start_line: Line number already read - will return messages AFTER this line.
                   Use -1 to read from beginning (default).
    
    Returns:
        Dict containing:
        - transcript: The conversation transcript (only new messages if start_line >= 0)
        - last_line_index: The index of the final line that was read
        - has_new_messages: Boolean indicating if there are new messages
    """
    conversation_filepath = _chat._session_id_to_claude_convo_path(session_id)

    if not conversation_filepath:
        return {
            "transcript": "No conversation found for the specified session_id.",
            "last_line_index": -1,
            "has_new_messages": False
        }

    conversation_transcript, last_line_index = _chat._process_claude_convo_incremental(
        conversation_filepath, start_line
    )

    # Check if there are new messages (last_line_index > start_line means we read new lines)
    has_new_messages = bool(conversation_transcript) and last_line_index > start_line

    if not has_new_messages and start_line >= 0:
        return {
            "transcript": "No new messages since last check.",
            "last_line_index": last_line_index,
            "has_new_messages": False
        }

    if not conversation_transcript and start_line == -1:
        return {
            "transcript": "Could not process the conversation transcript.",
            "last_line_index": last_line_index,
            "has_new_messages": False
        }

    return {
        "transcript": conversation_transcript,
        "last_line_index": last_line_index,
        "has_new_messages": has_new_messages
    }


# Main entry point for STDIO transport
if __name__ == "__main__":
    # Load environment variables from ~/.scribe/.env
    load_dotenv(Path.home() / ".scribe" / ".env")

    mcp.run(transport="stdio")
