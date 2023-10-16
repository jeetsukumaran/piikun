
from rich.console import Console
from rich.theme import Theme

piikun_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
})

console = Console(theme=piikun_theme)

