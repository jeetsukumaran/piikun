

import sys
import logging
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler

class LogHandler(RichHandler):
    """
    Parameters
    ----------
    level : Union[int, str], optional
        Log level, by default logging.NOTSET.
    console : :class:`~rich.console.Console`, optional
        Optional console instance to write logs. Default is to use a global console instance writing to stdout.
    show_time : bool, optional
        Show a column for the time, by default True.
    omit_repeated_times : bool, optional
        Omit repetition of the same time, by default True.
    show_level : bool, optional
        Show a column for the level, by default True.
    show_path : bool, optional
        Show the path to the original log call, by default True.
    enable_link_path : bool, optional
        Enable terminal link of path column to file, by default True.
    highlighter : Highlighter, optional
        Highlighter to style log messages, or None to use ReprHighlighter, by default None.
    markup : bool, optional
        Enable console markup in log messages, by default False.
    rich_tracebacks : bool, optional
        Enable rich tracebacks with syntax highlighting and formatting, by default False.
    tracebacks_width : Optional[int], optional
        Number of characters used to render tracebacks, or None for full width, by default None.
    tracebacks_extra_lines : int, optional
        Additional lines of code to render tracebacks, or None for full width, by default None.
    tracebacks_theme : str, optional
        Override pygments theme used in traceback.
    tracebacks_word_wrap : bool, optional
        Enable word wrapping of long tracebacks lines, by default True.
    tracebacks_show_locals : bool, optional
        Enable display of locals in tracebacks, by default False.
    locals_max_length : int, optional
        Maximum length of containers before abbreviating, or None for no abbreviation, by default 10.
    locals_max_string : int, optional
        Maximum length of string before truncating, or None to disable, by default 80.
    log_time_format : Union[str, TimeFormatterCallable], optional
        If ``log_time`` is enabled, either string for strftime or callable that formats the time, by default "[%x %X] ".
    """
    # def __init__(
    #     self,
    #     *args,
    #     **kwargs
    # ):
    #     # kwargs["extras"] = "markup"
    #     kwargs["rich_tracebacks"] = True
    #     kwargs["show_time"] = True
    #     kwargs["show_path"] = False
    #     kwargs["show_level"] = True
    #     kwargs["tracebacks_word_wrap"] = False
    #     super().__init__(*args, **kwargs)


def get_logger(
    logging_level=logging.INFO,
    # Show a column for the time
    show_time=True,
    # Omit repetition of the same time
    omit_repeated_times=False,
    # Show a column for the level
    show_level=True,
    # Show the path to the original log call
    show_path=False,
    ## Enable terminal link of path column to file
    # enable_link_path=True,
    ## Highlighter to style log messages, or None to use ReprHighlighter, by default None.
    # highlighter=None
    # Enable console markup in log messages
    markup=True,
    # Enable rich tracebacks with syntax highlighting and formatting
    rich_tracebacks=True,
    ## Number of characters used to render tracebacks, or None for full width, by default None.
    # tracebacks_width=None,
    ## Additional lines of code to render tracebacks, or None for full width, by default None.
    # tracebacks_extra_lines=None,
    ## Override pygments theme used in traceback.
    # tracebacks_theme=None,
    # Enable word wrapping of long tracebacks lines
    tracebacks_word_wrap=True,
    # Enable display of locals in tracebacks
    tracebacks_show_locals=True,
    # Maximum length of containers before abbreviating, or None for no abbreviation, by default 10.
    locals_max_length=10,
    # Maximum length of string before truncating, or None to disable, by default 80.
    locals_max_string=80,
    ## If ``log_time`` is enabled, either string for strftime or callable that formats the time, by default "[%x %X] ".
    # log_time_format="[%Y-%m-%d %H:%M:%S]"
):
    # console_fmt = "%(asctime)s.%(msecs)03d|%(levelname)s|%(pathname)s:%(lineno)d|%(message)s"
    # date_fmt = "[%Y-%m-%d %H:%M:%S]"
    # console_fmt = "[dim cyan] [%(asctime)s] %(message)s"
    # date_fmt = "%Y-%m-%d %H:%M:%S"
    console_fmt = "%(message)s"
    date_fmt = "[%Y-%m-%d %H:%M:%S]"
    logger = logging.getLogger("root")
    handler = RichHandler(
        level=logging_level,
        show_time=show_time,
        omit_repeated_times=omit_repeated_times,
        show_level=show_level,
        show_path=show_path,
        markup=markup,
    )
    handler.setFormatter(
        logging.Formatter(
            fmt=console_fmt,
            datefmt=date_fmt,
        )
    )
    logger.setLevel(logging_level)
    logger.addHandler(handler)
    return logger


def terminate_error(
    message,
    exit_code,
    ):
    if message:
        logger.error(f"Terminating run: {message}")
    sys.exit(exit_code)


package_console_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
})
console = Console(theme=package_console_theme)
logger = get_logger()
