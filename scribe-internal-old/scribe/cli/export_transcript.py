"""Export conversation transcript to clipboard."""

import os
import subprocess
import sys
from pathlib import Path
import platform
import click
import base64

# Import the chat log utilities from extensions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scribe.extensions import _chat_log_utils as _chat


def is_ssh_session():
    """Check if we're in an SSH session."""
    return "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ


def copy_via_osc52(text):
    """Copy text to clipboard using OSC 52 escape sequence for SSH sessions."""
    try:
        # Base64 encode the text
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")

        # Check if we're in tmux or screen
        is_tmux = "TMUX" in os.environ
        is_screen = "TERM" in os.environ and "screen" in os.environ.get("TERM", "")

        # Build the OSC 52 escape sequence
        if is_screen and not is_tmux:
            # GNU Screen needs DCS wrapper
            osc52_sequence = f"\033P\033]52;c;{encoded}\a\033\\"
        elif is_tmux:
            # tmux needs its own escape sequence
            osc52_sequence = f"\033Ptmux;\033\033]52;c;{encoded}\a\033\\"
        else:
            # Standard OSC 52 sequence
            osc52_sequence = f"\033]52;c;{encoded}\a"

        # Try different methods to write the OSC 52 sequence
        written = False
        
        # Method 1: Try /dev/tty first (works in normal SSH sessions)
        try:
            with open('/dev/tty', 'w') as tty:
                tty.write(osc52_sequence)
                tty.flush()
                written = True
        except (OSError, IOError):
            pass
        
        # Method 2: If in Claude Code, find parent terminal PTY
        if not written and os.environ.get("CLAUDECODE") == "1":
            try:
                # Trace parent processes to find the terminal PTY
                pid = os.getpid()
                for _ in range(5):  # Trace up to 5 levels
                    try:
                        # Get parent PID from /proc/[pid]/stat
                        with open(f'/proc/{pid}/stat', 'r') as f:
                            # Format: pid (comm) state ppid ...
                            stat_line = f.read()
                            # Find closing paren of command name
                            close_paren = stat_line.rfind(')')
                            fields = stat_line[close_paren + 2:].split()
                            ppid = int(fields[1])  # ppid is 4th field overall, 2nd after command
                        
                        # Check if parent has a TTY in /proc/[ppid]/fd/
                        for fd_num in ['0', '1', '2']:  # Check stdin, stdout, stderr
                            fd_path = f'/proc/{ppid}/fd/{fd_num}'
                            try:
                                link = os.readlink(fd_path)
                                if link.startswith('/dev/pts/') and link[9:].isdigit():
                                    # Found a PTY! Try to write to it
                                    with open(link, 'w') as pty:
                                        pty.write(osc52_sequence)
                                        pty.flush()
                                        written = True
                                        break
                            except:
                                pass
                        
                        if written:
                            break
                            
                        pid = ppid
                        if ppid <= 1:
                            break
                    except:
                        break
            except:
                pass

        return True
    except Exception as e:
        click.echo(f"Error with OSC 52 clipboard: {e}", err=True)
        return False


def copy_to_clipboard(text):
    """Copy text to system clipboard, cross-platform."""

    # For SSH sessions, use OSC 52 escape sequences
    if is_ssh_session():
        return copy_via_osc52(text)

    # For local sessions, use system clipboard tools
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE, text=True)
            process.communicate(text)
            return process.returncode == 0
        elif system == "Linux":
            # Try different clipboard utilities in order of preference
            clipboard_cmds = [
                ["xclip", "-selection", "clipboard"],
                ["wl-copy"],
                ["xsel", "-b"],
            ]

            for cmd in clipboard_cmds:
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        text=True,
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                    )
                    stdout, stderr = process.communicate(text)
                    if process.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue

            # If all native tools fail on Linux, try OSC 52 as fallback
            return copy_via_osc52(text)
        elif system == "Windows":
            process = subprocess.Popen(
                ["clip"], stdin=subprocess.PIPE, text=True, shell=True
            )
            process.communicate(text)
            return process.returncode == 0
        else:
            return False
    except Exception as e:
        click.echo(f"Error copying to clipboard: {e}", err=True)
        return False


def export_transcript():
    """Export the most recent conversation transcript for the current project to clipboard."""

    # Get current working directory
    current_dir = os.getcwd()
    click.echo(f"Looking for conversations in project: {current_dir}")

    # Find the most recent conversation for this project
    conversation_filepath = _chat._most_recent_claude_convo_path(current_dir)

    if not conversation_filepath:
        click.echo("No conversation found for the current project.", err=True)
        return False

    click.echo(f"Found conversation: {Path(conversation_filepath).name}")

    # Process the conversation to get the transcript
    conversation_transcript = _chat._process_claude_convo(conversation_filepath)

    if not conversation_transcript:
        click.echo("Could not process the conversation transcript.", err=True)
        return False

    # Copy to clipboard
    if copy_to_clipboard(conversation_transcript):
        # Count messages for feedback
        user_count = conversation_transcript.count("<USER>")
        assistant_count = conversation_transcript.count("<ASSISTANT>")
        click.echo(
            f"✅ Copied transcript to clipboard ({user_count} user messages, {assistant_count} assistant messages)"
        )
        return True
    else:
        click.echo(
            "Failed to copy to clipboard. The transcript has been printed below:",
            err=True,
        )
        click.echo("-" * 80)
        click.echo(conversation_transcript)
        click.echo("-" * 80)
        return False


if __name__ == "__main__":
    # For testing directly
    export_transcript()
