

import sys
import logging
import datetime
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


class RuntimeClient:
    @staticmethod
    def ensure_random_seed(random_seed=None):
        if random_seed is None:
            rtmp = random.Random()
            random_seed = rtmp.getrandbits(32)
        return random_seed

    @staticmethod
    def get_rng(random_seed=None):
        random_seed = RuntimeClient.ensure_random_seed(random_seed)
        rng = random.Random(random_seed)
        rng._random_seed = random_seed
        return rng

    @property
    def logger(self):
        if (
            not hasattr(self, "_logger")
            or self._logger is None
        ):
            self._logger = logger
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def __init__(
        self,
        random_seed=None,
        logger=None,
        output_directory=None,
        output_title=None,
        output_configuration=None,
    ):
        self.logger = logger
            # from yakherd import Logger
        self.logger.info(
            f"Initializing system runtime context at: {datetime.datetime.now()}"
        )
        self.opened_output_handles = {}

    @property
    def output_title(self):
        if (
            not hasattr(self, "_output_title")
            or self._output_title is None
        ):
            self._output_title = ""
        return self._output_title
    @output_title.setter
    def output_title(self, value):
        if value:
            self._output_title = value.strip()
        else:
            self._output_title = ""
    @output_title.deleter
    def output_title(self):
        del self._output_title


    @property
    def output_directory(self):
        if (
            not hasattr(self, "_output_directory")
            or self._output_directory is None
        ):
            self._output_directory = pathlib.Path(args.output_directory)
        return self._output_directory
    @output_directory.setter
    def output_directory(self, value):
        self._output_directory = value
    @output_directory.deleter
    def output_directory(self):
        del self._output_directory

    def compose_output_name(self, subtitle=None, ext=None,):
        s = []
        try:
            subtitle = subtitle.strip()
        except AttributeError:
            subtitle = ""
        if subtitle:
            s.append("-")
            s.append(subtitle)
        if ext:
            ext = ext.strip()
            if not ext.startswith(".") and not s[-1].endswith("."):
                s.append(".")
            s.append(ext)
        output_name = "".join(s)
        return output_name

    def compose_output_path(self, subtitle=None, ext=None,):
        output_name = self.compose_output_name(
            subtitle=subtitle,
            ext=ext,
        )
        output_path = pathlib.Pathlib(args.output_directory) / output_name
        return output_path

    def open_output(
        self,
        subtitle=None,
        ext=None,
        mode="w",
        is_internally_disambiguate=True,
    ):
        output_path = self.compose_output_path(subtitle=subtitle, ext=ext)
        if is_internally_disambiguate:
            disambigution_idx = 1
            while output_path in self.opened_output_handles:
                disambigution_idx += 1
                output_path = self.compose_output_path(subtitle=f"{subtitle}-{disambigution_idx}", ext=ext)
        output_handle = open(output_path, mode)
        self.opened_output_handles[output_path] = output_handle

