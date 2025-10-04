from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pathlib import Path

from .converse import Converse
from .config import settings


def main():
    console = Console()
    console.print(Panel(Text(f"Chat with {settings.agent.agent_name}", justify="center")))

    history = FileHistory(Path.home() / ".ollama-rag-history")
    session = PromptSession(history=history)

    converse = Converse()

    while True:
        try:
            user_input = session.prompt(f"{settings.agent.user_name}> ")
            if user_input.lower() in ["exit", "quit"]:
                break

            with console.status("[bold green]Thinking..."):
                agent_response = converse.ask(user_input)
            
            console.print(Panel(agent_response, title=settings.agent.agent_name, border_style="green"))

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}")

if __name__ == "__main__":
    main()
