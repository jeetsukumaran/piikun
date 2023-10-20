

import sys
import logging
import datetime
import pathlib
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler

def compose_output_title_from_source(source_path):
    return pathlib.Path(source_path).stem.split("__partitions")[0]

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
    console=None,
):
    # console_fmt = "%(asctime)s.%(msecs)03d|%(levelname)s|%(pathname)s:%(lineno)d|%(message)s"
    # date_fmt = "[%Y-%m-%d %H:%M:%S]"
    # console_fmt = "[dim cyan] [%(asctime)s] %(message)s"
    # date_fmt = "%Y-%m-%d %H:%M:%S"
    console_fmt = "%(message)s"
    date_fmt = "[%Y-%m-%d %H:%M:%S]"
    logger = logging.getLogger("root")
    if not console:
        console = Console(stderr=True)
    handler = RichHandler(
        level=logging_level,
        show_time=show_time,
        omit_repeated_times=omit_repeated_times,
        show_level=show_level,
        show_path=show_path,
        markup=markup,
        console=console,
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


class RuntimeClient:

    _stderr_console = Console(
        stderr=True,
    )
    _logger = None

    # @staticmethod
    # def ensure_random_seed(random_seed=None):
    #     if random_seed is None:
    #         rtmp = random.Random()
    #         random_seed = rtmp.getrandbits(32)
    #     return random_seed

    # @staticmethod
    # def get_rng(random_seed=None):
    #     random_seed = RuntimeClient.ensure_random_seed(random_seed)
    #     rng = random.Random(random_seed)
    #     rng._random_seed = random_seed
    #     return rng

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
            f"Runtime context at {datetime.datetime.now()}"
        )
        self.opened_output_handles = {}
        self.output_title = output_title
        self.output_directory = output_directory

    @property
    def console(self):
        if (
            not hasattr(self, "_console")
            or self._console is None
        ):
            self._console = RuntimeClient._stderr_console
        return self._console
    @console.setter
    def console(self, value):
        self._console = value

    @property
    def logger(self):
        if RuntimeClient._logger is None:
            RuntimeClient._logger = get_logger(console=self.console)
        return RuntimeClient._logger
    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def output_title(self):
        if (
            not hasattr(self, "_output_title")
            or (not self._output_title)
        ):
            self._output_title = None
        return self._output_title
    @output_title.setter
    def output_title(self, value):
        if value:
            self._output_title = value.strip()
        else:
            self._output_title = None


    @property
    def output_directory(self):
        if (
            not hasattr(self, "_output_directory")
            or self._output_directory is None
        ):
            self._output_directory = pathlib.Path.cwd()
        return self._output_directory
    @output_directory.setter
    def output_directory(self, value):
        self._output_directory = value
    @output_directory.deleter
    def output_directory(self):
        del self._output_directory

    def compose_output_name(self, subtitle=None, ext=None,):
        s = []
        if self.output_title:
            s.append(self.output_title)
        if subtitle:
            s.append(subtitle.strip())
        output_name = "__".join(s)
        if ext:
            ext = ext.strip()
            if not ext.startswith(".") and not output_name.endswith("."):
                output_name += "."
            output_name += ext
        return output_name

    def compose_output_path(self, subtitle=None, ext=None,):
        output_name = self.compose_output_name(
            subtitle=subtitle,
            ext=ext,
        )
        output_path = pathlib.Path(self.output_directory) / output_name
        return output_path

    def open_output(
        self,
        subtitle=None,
        ext=None,
        mode="w",
        is_internally_disambiguate=True,
    ):
        if self.output_title == "-":
            return sys.stdout
        output_path = self.compose_output_path(subtitle=subtitle, ext=ext)
        if is_internally_disambiguate:
            disambigution_idx = 1
            while output_path in self.opened_output_handles:
                disambigution_idx += 1
                output_path = self.compose_output_path(subtitle=f"{subtitle}-{disambigution_idx}", ext=ext)
        output_handle = open(output_path, mode)
        self.opened_output_handles[output_path] = output_handle
        return output_handle

