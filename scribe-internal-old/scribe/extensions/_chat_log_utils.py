from typing import Optional  # noqa: F401 (future use, keep consistent with other utils)
import os
import json
from pathlib import Path
# Note: keep imports minimal and only what is used


def _active_filepath_to_claude_convo_path(focused_filepath: str) -> str:
    """
    Given the path of a file (e.g. a scribe notebook) that the user has open, fetches the
    path to the most recent Claude Code conversation that mentions that file.
    """
    claude_dir = Path.home() / ".claude" / "projects"

    if not claude_dir.exists():
        return None

    # Find all JSONL files across all project directories
    jsonl_files = []
    for project_dir in claude_dir.iterdir():
        if project_dir.is_dir():
            for jsonl_file in project_dir.glob("*.jsonl"):
                jsonl_files.append(jsonl_file)

    # Sort by modification time (most recent first)
    jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Search each file for the focused_filepath
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Check in message content if it exists
                        if "message" in data and "content" in data["message"]:
                            if focused_filepath in data["message"]["content"]:
                                return str(jsonl_file)
                        # Also check in tool calls and results if they exist
                        if "tool_calls" in data:
                            tool_str = json.dumps(data["tool_calls"])
                            if focused_filepath in tool_str:
                                return str(jsonl_file)
                        if "tool_results" in data:
                            result_str = json.dumps(data["tool_results"])
                            if focused_filepath in result_str:
                                return str(jsonl_file)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    return None


def _session_id_to_claude_convo_path(session_id: str) -> str:
    """
    Given a session_id, returns the full path to the most recent Claude Code conversation
    file whose filename contains that session_id.
    """
    claude_dir = Path.home() / ".claude" / "projects"

    if not claude_dir.exists():
        return None

    matching_files = []
    try:
        for project_dir in claude_dir.iterdir():
            if not project_dir.is_dir():
                continue
            # Look for JSONL files that include the session_id in the filename
            for jsonl_file in project_dir.glob(f"*{session_id}*.jsonl"):
                matching_files.append(jsonl_file)
    except Exception:
        # If any error occurs while scanning, fail gracefully
        return None

    if not matching_files:
        return None

    # Return the most recently modified matching file
    matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(matching_files[0])


def _most_recent_claude_convo_path_for_project(project_path: str) -> str:
    """
    Gets the most recently updated Claude Code conversation for a given project.
    """
    project_dir = Path(project_path)
    if not project_dir.exists():
        print(f"Project directory {project_path} does not exist")
        return None

    # Find all JSONL files across all project directories
    jsonl_files = []
    for jsonl_file in project_dir.glob("*.jsonl"):
        jsonl_files.append(jsonl_file)

    if not jsonl_files:
        print(f"No JSONL files found in project directory {project_path}")
        return None

    # Sort by modification time (most recent first) and return the first one
    jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(jsonl_files[0])


def _most_recent_claude_convo_path(project_path: str = None) -> str:
    """
    Gets the most recently updated Claude Code conversation.
    """
    claude_dir = Path.home() / ".claude" / "projects"

    if not claude_dir.exists():
        print("No Claude directory found")
        return None

    if project_path is not None:
        # Convert filesystem project path to Claude project directory name
        # Claude uses the full path with slashes replaced by dashes
        # e.g., /mnt/polished-lake/home/mark/projects/scribe -> -mnt-polished-lake-home-mark-projects-scribe
        project_name = project_path.replace("/", "-").lstrip("-")
        claude_project_dir = claude_dir / project_name

        if claude_project_dir.exists():
            return _most_recent_claude_convo_path_for_project(str(claude_project_dir))
        else:
            # If exact match doesn't work, search for conversations mentioning this project
            # across all Claude project directories
            matching_files = []
            for project_dir in claude_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                for jsonl_file in project_dir.glob("*.jsonl"):
                    try:
                        with open(jsonl_file, "r") as f:
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    if project_path in json.dumps(data):
                                        matching_files.append(jsonl_file)
                                        break
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        continue

            if matching_files:
                matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                return str(matching_files[0])

            return None

    # Find all JSONL files across all project directories
    jsonl_files = []
    for project_dir in claude_dir.iterdir():
        if not project_dir.is_dir():
            continue
        recent_file_path = _most_recent_claude_convo_path_for_project(str(project_dir))
        if recent_file_path:  # Only add if not None
            jsonl_files.append(Path(recent_file_path))

    if not jsonl_files:
        return None

    # Sort by modification time (most recent first) and return the first one
    jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(jsonl_files[0])


def _process_claude_convo(convo_path: str) -> str:
    """
    Given the path to a Claude Code conversation, processes it to extract the relevant
    messages/content as a string that can be used as a prompt for another model.
    """
    import json

    if not convo_path or not os.path.exists(convo_path):
        return ""

    def escape_tags(text):
        """Escape any existing USER/ASSISTANT tags in the content"""
        text = text.replace("<USER>", "&lt;USER&gt;")
        text = text.replace("</USER>", "&lt;/USER&gt;")
        text = text.replace("<ASSISTANT>", "&lt;ASSISTANT&gt;")
        text = text.replace("</ASSISTANT>", "&lt;/ASSISTANT&gt;")
        return text

    # Collect all messages first
    messages = []

    try:
        with open(convo_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)

                    # Skip metadata messages
                    if data.get("isMeta", False):
                        continue

                    # Process user messages
                    if data.get("type") == "user" and "message" in data:
                        msg = data["message"]
                        if "content" in msg:
                            content = msg["content"]
                            user_text = []

                            # Handle string content
                            if isinstance(content, str):
                                user_text.append(escape_tags(content))
                            # Handle list content (structured messages)
                            elif isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict):
                                        if item.get("type") == "text":
                                            text = item.get("text", "")
                                            if text:
                                                user_text.append(escape_tags(text))
                                        # Handle tool results (like from mcp__scribe__execute_code)
                                        elif item.get("type") == "tool_result":
                                            tool_content = item.get("content", [])
                                            result_parts = []
                                            
                                            if isinstance(tool_content, list):
                                                for content_item in tool_content:
                                                    # Handle text results
                                                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                                                        result_text = content_item.get("text", "")
                                                        # Try to parse JSON result for better formatting
                                                        try:
                                                            result_data = json.loads(result_text)
                                                            if isinstance(result_data, list):
                                                                for exec_result in result_data:
                                                                    if "outputs" in exec_result:
                                                                        # Check if this is an edit_cell result
                                                                        if "cell_index" in exec_result or "actual_notebook_index" in exec_result:
                                                                            result_parts.append("[Cell Edit Result]")
                                                                            if "cell_index" in exec_result:
                                                                                result_parts.append(f"  Cell index: {exec_result['cell_index']}")
                                                                            if "actual_notebook_index" in exec_result:
                                                                                result_parts.append(f"  Notebook index: {exec_result['actual_notebook_index']}")
                                                                            if "execution_count" in exec_result:
                                                                                result_parts.append(f"  Execution count: {exec_result['execution_count']}")
                                                                        else:
                                                                            result_parts.append("[Execution Result]")
                                                                        
                                                                        outputs = exec_result.get("outputs", [])
                                                                        if outputs:
                                                                            for output in outputs:
                                                                                if output.get("type") == "text":
                                                                                    result_parts.append(f"  Output: {escape_tags(output.get('content', ''))}")
                                                                                elif output.get("type") == "image":
                                                                                    result_parts.append("  Output: [Image]")
                                                                                elif output.get("type") == "error":
                                                                                    error_name = output.get("name", "Error")
                                                                                    error_msg = output.get("message", "")
                                                                                    result_parts.append(f"  Error: {escape_tags(error_name)}: {escape_tags(error_msg)}")
                                                                                    # Show key part of traceback if available
                                                                                    traceback = output.get("traceback", [])
                                                                                    if traceback and isinstance(traceback, list) and len(traceback) > 0:
                                                                                        # Show the last line which usually has the actual error
                                                                                        result_parts.append(f"  Traceback: {escape_tags(traceback[-1])}")
                                                                        else:
                                                                            result_parts.append("  Output: (no output)")
                                                            else:
                                                                result_parts.append(f"[Tool Result]: {escape_tags(result_text)}")
                                                        except:
                                                            # If not JSON or can't parse, just show as is
                                                            if result_text:
                                                                result_parts.append(f"[Tool Result]: {escape_tags(result_text)}")
                                                    # Handle image results
                                                    elif isinstance(content_item, dict) and content_item.get("type") == "image":
                                                        result_parts.append("[Image output]")
                                            
                                            # Add results to the previous assistant message if they exist
                                            if result_parts and messages and messages[-1][0] == "assistant":
                                                # Append to the last assistant message
                                                messages[-1] = ("assistant", messages[-1][1] + "\n" + "\n".join(result_parts))
                                            elif result_parts:
                                                # Otherwise add as user message
                                                user_text.extend(result_parts)

                            if user_text:
                                messages.append(("user", "".join(user_text)))

                    # Process assistant messages
                    elif data.get("type") == "assistant" and "message" in data:
                        msg = data["message"]
                        if "content" in msg:
                            content = msg["content"]
                            assistant_parts = []

                            # Handle list content (structured messages)
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict):
                                        # Handle text responses
                                        if item.get("type") == "text":
                                            text = item.get("text", "")
                                            if text:
                                                assistant_parts.append(
                                                    escape_tags(text)
                                                )
                                        # Handle tool use
                                        elif item.get("type") == "tool_use":
                                            tool_name = item.get("name", "Unknown")
                                            tool_input = item.get("input", {})

                                            # Special handling for Bash commands
                                            if tool_name == "Bash":
                                                command = tool_input.get("command", "")
                                                assistant_parts.append(
                                                    f"[Tool: {tool_name}] `{escape_tags(command)}`"
                                                )
                                            # Special handling for TodoWrite to show the todos
                                            elif tool_name == "TodoWrite":
                                                todos = tool_input.get("todos", [])
                                                if todos:
                                                    todo_lines = [
                                                        f"[Tool: {tool_name}]"
                                                    ]
                                                    for todo in todos:
                                                        status = todo.get(
                                                            "status", "unknown"
                                                        )
                                                        content = escape_tags(
                                                            todo.get("content", "")
                                                        )
                                                        todo_lines.append(
                                                            f"  - [{status}] {content}"
                                                        )
                                                    assistant_parts.append(
                                                        "\n".join(todo_lines)
                                                    )
                                                else:
                                                    assistant_parts.append(
                                                        f"[Tool: {tool_name}]"
                                                    )
                                            # Special handling for Edit tool - show full content
                                            elif tool_name == "Edit":
                                                file_path = tool_input.get("file_path", "")
                                                old_string = tool_input.get("old_string", "")
                                                new_string = tool_input.get("new_string", "")
                                                replace_all = tool_input.get("replace_all", False)
                                                
                                                edit_lines = [f"[Tool: Edit]"]
                                                edit_lines.append(f"  File: {escape_tags(file_path)}")
                                                if replace_all:
                                                    edit_lines.append("  Replace all: true")
                                                edit_lines.append("  --- OLD ---")
                                                edit_lines.append(escape_tags(old_string))
                                                edit_lines.append("  --- NEW ---")
                                                edit_lines.append(escape_tags(new_string))
                                                edit_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(edit_lines))
                                            # Special handling for MultiEdit tool - show all edits
                                            elif tool_name == "MultiEdit":
                                                file_path = tool_input.get("file_path", "")
                                                edits = tool_input.get("edits", [])
                                                
                                                edit_lines = [f"[Tool: MultiEdit]"]
                                                edit_lines.append(f"  File: {escape_tags(file_path)}")
                                                edit_lines.append(f"  Number of edits: {len(edits)}")
                                                
                                                for i, edit in enumerate(edits, 1):
                                                    old_string = edit.get("old_string", "")
                                                    new_string = edit.get("new_string", "")
                                                    replace_all = edit.get("replace_all", False)
                                                    
                                                    edit_lines.append(f"  Edit {i}:")
                                                    if replace_all:
                                                        edit_lines.append("    Replace all: true")
                                                    edit_lines.append("    --- OLD ---")
                                                    edit_lines.append(escape_tags(old_string))
                                                    edit_lines.append("    --- NEW ---")
                                                    edit_lines.append(escape_tags(new_string))
                                                    edit_lines.append("    --- END ---")
                                                
                                                assistant_parts.append("\n".join(edit_lines))
                                            # Special handling for Write tool - show full content
                                            elif tool_name == "Write":
                                                file_path = tool_input.get("file_path", "")
                                                content = tool_input.get("content", "")
                                                
                                                write_lines = [f"[Tool: Write]"]
                                                write_lines.append(f"  File: {escape_tags(file_path)}")
                                                write_lines.append("  --- CONTENT ---")
                                                write_lines.append(escape_tags(content))
                                                write_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(write_lines))
                                            # Special handling for Scribe execute_code MCP command
                                            elif tool_name == "mcp__scribe__execute_code":
                                                code = tool_input.get("code", "")
                                                
                                                exec_lines = [f"[Tool: mcp__scribe__execute_code]"]
                                                exec_lines.append("  --- CODE ---")
                                                exec_lines.append(escape_tags(code))
                                                exec_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(exec_lines))
                                            # Special handling for Scribe edit_cell MCP command
                                            elif tool_name == "mcp__scribe__edit_cell":
                                                code = tool_input.get("code", "")
                                                cell_index = tool_input.get("cell_index", "")
                                                
                                                edit_lines = [f"[Tool: mcp__scribe__edit_cell]"]
                                                edit_lines.append(f"  Cell index: {escape_tags(str(cell_index))}")
                                                edit_lines.append("  --- CODE ---")
                                                edit_lines.append(escape_tags(code))
                                                edit_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(edit_lines))
                                            else:
                                                # For other tools, try to show meaningful input if available
                                                if tool_input:
                                                    # Try to extract meaningful fields
                                                    if "file_path" in tool_input:
                                                        assistant_parts.append(
                                                            f"[Tool: {tool_name}] file_path => {escape_tags(str(tool_input['file_path']))}"
                                                        )
                                                    elif "pattern" in tool_input:
                                                        assistant_parts.append(
                                                            f"[Tool: {tool_name}] pattern => {escape_tags(str(tool_input['pattern']))}"
                                                        )
                                                    elif "query" in tool_input:
                                                        assistant_parts.append(
                                                            f"[Tool: {tool_name}] query => {escape_tags(str(tool_input['query']))}"
                                                        )
                                                    else:
                                                        assistant_parts.append(
                                                            f"[Tool: {tool_name}]"
                                                        )
                                                else:
                                                    assistant_parts.append(
                                                        f"[Tool: {tool_name}]"
                                                    )
                            # Handle string content
                            elif isinstance(content, str):
                                assistant_parts.append(escape_tags(content))

                            if assistant_parts:
                                messages.append(
                                    ("assistant", "\n".join(assistant_parts))
                                )

                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue

    except Exception as e:
        return f"Error processing conversation: {str(e)}"

    # Now consolidate consecutive messages from the same role
    transcript = []
    current_role = None
    current_content = []

    for role, content in messages:
        if role != current_role:
            # Save previous block if exists
            if current_role and current_content:
                if current_role == "user":
                    transcript.append(
                        f"<USER>\n{chr(10).join(current_content)}\n</USER>"
                    )
                else:
                    transcript.append(
                        f"<ASSISTANT>\n{chr(10).join(current_content)}\n</ASSISTANT>"
                    )
            # Start new block
            current_role = role
            current_content = [content]
        else:
            # Continue current block
            current_content.append(content)

    # Don't forget the last block
    if current_role and current_content:
        if current_role == "user":
            transcript.append(f"<USER>\n{chr(10).join(current_content)}\n</USER>")
        else:
            transcript.append(
                f"<ASSISTANT>\n{chr(10).join(current_content)}\n</ASSISTANT>"
            )

    return "\n\n".join(transcript)


def _get_session_id_from_convo(convo_path: str) -> Optional[str]:
    """
    Extract the session ID from a Claude conversation file.
    Returns the first sessionId found in the JSONL file.
    """
    if not convo_path or not os.path.exists(convo_path):
        return None
    
    try:
        with open(convo_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "sessionId" in data:
                        return data["sessionId"]
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    return None


def _process_claude_convo_incremental(convo_path: str, start_line: int = -1) -> tuple[str, int]:
    """
    Process a Claude conversation file starting from a specific line number.
    
    Args:
        convo_path: Path to the conversation file
        start_line: Line number to start reading AFTER (0-indexed).
                   To read from beginning, use -1.
    
    Returns:
        tuple: (transcript, last_line_number)
            - transcript: Processed conversation text (empty string if no new messages)
            - last_line_number: The index of the last line read (returns start_line if no new lines)
    """
    if not convo_path or not os.path.exists(convo_path):
        return "", -1
    
    def escape_tags(text):
        """Escape any existing USER/ASSISTANT tags in the content"""
        text = text.replace("<USER>", "&lt;USER&gt;")
        text = text.replace("</USER>", "&lt;/USER&gt;")
        text = text.replace("<ASSISTANT>", "&lt;ASSISTANT&gt;")
        text = text.replace("</ASSISTANT>", "&lt;/ASSISTANT&gt;")
        return text
    
    # Collect all messages starting after start_line
    messages = []
    last_line_read = start_line  # Initialize to start_line to maintain continuity
    
    try:
        with open(convo_path, "r") as f:
            for line_num, line in enumerate(f):
                # Skip lines up to and including start_line
                # (we want to read AFTER start_line)
                if line_num <= start_line:
                    continue
                
                last_line_read = line_num
                
                try:
                    data = json.loads(line)
                    
                    # Skip metadata messages
                    if data.get("isMeta", False):
                        continue
                    
                    # Process user messages
                    if data.get("type") == "user" and "message" in data:
                        msg = data["message"]
                        if "content" in msg:
                            content = msg["content"]
                            user_text = []
                            
                            # Handle string content
                            if isinstance(content, str):
                                user_text.append(escape_tags(content))
                            # Handle list content (structured messages)
                            elif isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict):
                                        if item.get("type") == "text":
                                            text = item.get("text", "")
                                            if text:
                                                user_text.append(escape_tags(text))
                                        # Handle tool results (like from mcp__scribe__execute_code)
                                        elif item.get("type") == "tool_result":
                                            tool_content = item.get("content", [])
                                            result_parts = []
                                            
                                            if isinstance(tool_content, list):
                                                for content_item in tool_content:
                                                    # Handle text results
                                                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                                                        result_text = content_item.get("text", "")
                                                        # Try to parse JSON result for better formatting
                                                        try:
                                                            result_data = json.loads(result_text)
                                                            if isinstance(result_data, list):
                                                                for exec_result in result_data:
                                                                    if "outputs" in exec_result:
                                                                        # Check if this is an edit_cell result
                                                                        if "cell_index" in exec_result or "actual_notebook_index" in exec_result:
                                                                            result_parts.append("[Cell Edit Result]")
                                                                            if "cell_index" in exec_result:
                                                                                result_parts.append(f"  Cell index: {exec_result['cell_index']}")
                                                                            if "actual_notebook_index" in exec_result:
                                                                                result_parts.append(f"  Notebook index: {exec_result['actual_notebook_index']}")
                                                                            if "execution_count" in exec_result:
                                                                                result_parts.append(f"  Execution count: {exec_result['execution_count']}")
                                                                        else:
                                                                            result_parts.append("[Execution Result]")
                                                                        
                                                                        outputs = exec_result.get("outputs", [])
                                                                        if outputs:
                                                                            for output in outputs:
                                                                                if output.get("type") == "text":
                                                                                    result_parts.append(f"  Output: {escape_tags(output.get('content', ''))}")
                                                                                elif output.get("type") == "image":
                                                                                    result_parts.append("  Output: [Image]")
                                                                                elif output.get("type") == "error":
                                                                                    error_name = output.get("name", "Error")
                                                                                    error_msg = output.get("message", "")
                                                                                    result_parts.append(f"  Error: {escape_tags(error_name)}: {escape_tags(error_msg)}")
                                                                                    # Show key part of traceback if available
                                                                                    traceback = output.get("traceback", [])
                                                                                    if traceback and isinstance(traceback, list) and len(traceback) > 0:
                                                                                        # Show the last line which usually has the actual error
                                                                                        result_parts.append(f"  Traceback: {escape_tags(traceback[-1])}")
                                                                        else:
                                                                            result_parts.append("  Output: (no output)")
                                                            else:
                                                                result_parts.append(f"[Tool Result]: {escape_tags(result_text)}")
                                                        except:
                                                            # If not JSON or can't parse, just show as is
                                                            if result_text:
                                                                result_parts.append(f"[Tool Result]: {escape_tags(result_text)}")
                                                    # Handle image results
                                                    elif isinstance(content_item, dict) and content_item.get("type") == "image":
                                                        result_parts.append("[Image output]")
                                            
                                            # Add results to the previous assistant message if they exist
                                            if result_parts and messages and messages[-1][0] == "assistant":
                                                # Append to the last assistant message
                                                messages[-1] = ("assistant", messages[-1][1] + "\n" + "\n".join(result_parts))
                                            elif result_parts:
                                                # Otherwise add as user message
                                                user_text.extend(result_parts)
                            
                            if user_text:
                                messages.append(("user", "".join(user_text)))
                    
                    # Process assistant messages
                    elif data.get("type") == "assistant" and "message" in data:
                        msg = data["message"]
                        if "content" in msg:
                            content = msg["content"]
                            assistant_parts = []
                            
                            # Handle list content (structured messages)
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict):
                                        # Handle text responses
                                        if item.get("type") == "text":
                                            text = item.get("text", "")
                                            if text:
                                                assistant_parts.append(escape_tags(text))
                                        # Handle tool use
                                        elif item.get("type") == "tool_use":
                                            tool_name = item.get("name", "Unknown")
                                            tool_input = item.get("input", {})
                                            
                                            # Special handling for Bash commands
                                            if tool_name == "Bash":
                                                command = tool_input.get("command", "")
                                                assistant_parts.append(
                                                    f"[Tool: {tool_name}] `{escape_tags(command)}`"
                                                )
                                            # Special handling for TodoWrite to show the todos
                                            elif tool_name == "TodoWrite":
                                                todos = tool_input.get("todos", [])
                                                if todos:
                                                    todo_lines = [f"[Tool: {tool_name}]"]
                                                    for todo in todos:
                                                        status = todo.get("status", "unknown")
                                                        content = escape_tags(todo.get("content", ""))
                                                        todo_lines.append(f"  - [{status}] {content}")
                                                    assistant_parts.append("\n".join(todo_lines))
                                                else:
                                                    assistant_parts.append(f"[Tool: {tool_name}]")
                                            # Special handling for Edit tool - show full content
                                            elif tool_name == "Edit":
                                                file_path = tool_input.get("file_path", "")
                                                old_string = tool_input.get("old_string", "")
                                                new_string = tool_input.get("new_string", "")
                                                replace_all = tool_input.get("replace_all", False)
                                                
                                                edit_lines = [f"[Tool: Edit]"]
                                                edit_lines.append(f"  File: {escape_tags(file_path)}")
                                                if replace_all:
                                                    edit_lines.append("  Replace all: true")
                                                edit_lines.append("  --- OLD ---")
                                                edit_lines.append(escape_tags(old_string))
                                                edit_lines.append("  --- NEW ---")
                                                edit_lines.append(escape_tags(new_string))
                                                edit_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(edit_lines))
                                            # Special handling for MultiEdit tool - show all edits
                                            elif tool_name == "MultiEdit":
                                                file_path = tool_input.get("file_path", "")
                                                edits = tool_input.get("edits", [])
                                                
                                                edit_lines = [f"[Tool: MultiEdit]"]
                                                edit_lines.append(f"  File: {escape_tags(file_path)}")
                                                edit_lines.append(f"  Number of edits: {len(edits)}")
                                                
                                                for i, edit in enumerate(edits, 1):
                                                    old_string = edit.get("old_string", "")
                                                    new_string = edit.get("new_string", "")
                                                    replace_all = edit.get("replace_all", False)
                                                    
                                                    edit_lines.append(f"  Edit {i}:")
                                                    if replace_all:
                                                        edit_lines.append("    Replace all: true")
                                                    edit_lines.append("    --- OLD ---")
                                                    edit_lines.append(escape_tags(old_string))
                                                    edit_lines.append("    --- NEW ---")
                                                    edit_lines.append(escape_tags(new_string))
                                                    edit_lines.append("    --- END ---")
                                                
                                                assistant_parts.append("\n".join(edit_lines))
                                            # Special handling for Write tool - show full content
                                            elif tool_name == "Write":
                                                file_path = tool_input.get("file_path", "")
                                                content = tool_input.get("content", "")
                                                
                                                write_lines = [f"[Tool: Write]"]
                                                write_lines.append(f"  File: {escape_tags(file_path)}")
                                                write_lines.append("  --- CONTENT ---")
                                                write_lines.append(escape_tags(content))
                                                write_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(write_lines))
                                            # Special handling for Scribe execute_code MCP command
                                            elif tool_name == "mcp__scribe__execute_code":
                                                code = tool_input.get("code", "")
                                                
                                                exec_lines = [f"[Tool: mcp__scribe__execute_code]"]
                                                exec_lines.append("  --- CODE ---")
                                                exec_lines.append(escape_tags(code))
                                                exec_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(exec_lines))
                                            # Special handling for Scribe edit_cell MCP command
                                            elif tool_name == "mcp__scribe__edit_cell":
                                                code = tool_input.get("code", "")
                                                cell_index = tool_input.get("cell_index", "")
                                                
                                                edit_lines = [f"[Tool: mcp__scribe__edit_cell]"]
                                                edit_lines.append(f"  Cell index: {escape_tags(str(cell_index))}")
                                                edit_lines.append("  --- CODE ---")
                                                edit_lines.append(escape_tags(code))
                                                edit_lines.append("  --- END ---")
                                                assistant_parts.append("\n".join(edit_lines))
                                            else:
                                                # For other tools, try to show meaningful input if available
                                                if tool_input:
                                                    # Try to extract meaningful fields
                                                    if "file_path" in tool_input:
                                                        assistant_parts.append(
                                                            f"[Tool: {tool_name}] file_path => {escape_tags(str(tool_input['file_path']))}"
                                                        )
                                                    elif "pattern" in tool_input:
                                                        assistant_parts.append(
                                                            f"[Tool: {tool_name}] pattern => {escape_tags(str(tool_input['pattern']))}"
                                                        )
                                                    elif "query" in tool_input:
                                                        assistant_parts.append(
                                                            f"[Tool: {tool_name}] query => {escape_tags(str(tool_input['query']))}"
                                                        )
                                                    else:
                                                        assistant_parts.append(f"[Tool: {tool_name}]")
                                                else:
                                                    assistant_parts.append(f"[Tool: {tool_name}]")
                            # Handle string content
                            elif isinstance(content, str):
                                assistant_parts.append(escape_tags(content))
                            
                            if assistant_parts:
                                messages.append(("assistant", "\n".join(assistant_parts)))
                
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    
    except Exception as e:
        return f"Error processing conversation: {str(e)}", -1
    
    # If no new lines were read, return empty transcript with start_line preserved
    if last_line_read == start_line:
        return "", start_line
    
    # Now consolidate consecutive messages from the same role
    transcript = []
    current_role = None
    current_content = []
    
    for role, content in messages:
        if role != current_role:
            # Save previous block if exists
            if current_role and current_content:
                if current_role == "user":
                    transcript.append(f"<USER>\n{chr(10).join(current_content)}\n</USER>")
                else:
                    transcript.append(f"<ASSISTANT>\n{chr(10).join(current_content)}\n</ASSISTANT>")
            # Start new block
            current_role = role
            current_content = [content]
        else:
            # Continue current block
            current_content.append(content)
    
    # Don't forget the last block
    if current_role and current_content:
        if current_role == "user":
            transcript.append(f"<USER>\n{chr(10).join(current_content)}\n</USER>")
        else:
            transcript.append(f"<ASSISTANT>\n{chr(10).join(current_content)}\n</ASSISTANT>")
    
    return "\n\n".join(transcript), last_line_read
