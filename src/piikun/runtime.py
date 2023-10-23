

import time
import sys
import logging
import datetime
import pathlib
from rich.rule import Rule
from rich import progress
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler

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
    logger._runtime_handler = handler
    return logger


class RuntimeContext:

    _stderr_console = Console(
        stderr=True,
    )

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
            f"Runtime context {id(self)} initialized at {datetime.datetime.now()}"
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
            self._console = RuntimeContext._stderr_console
        return self._console
    @console.setter
    def console(self, value):
        self._console = value

    @property
    def logger(self):
        if (
            not hasattr(self, "_logger")
            or (not self._logger)
        ):
            self._logger = get_logger(console=self.console)
        return self._logger
    @logger.setter
    def logger(self, value):
        self._logger = value

    def terminate_error(
        self,
        message,
        exit_code,
        ):
        if message:
            self.logger.error(f"Terminating run: {message}")
        sys.exit(exit_code)

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

    def open_output_datastore(
        self,
        *args,
        **kwargs
    ):
        output_handle = self.open_output(**kwargs)
        data_store = DataStore(
            file_handle=output_handle,
            config_d=kwargs.get("datastore_config_d", None),
        )
        return data_store

    def open_output_json_list_file(
        self,
        *args,
        **kwargs
    ):
        output_handle = self.open_output(**kwargs)
        data_store = JsonListFile(
            file_handle=output_handle,
        )
        return data_store

    def sleep(self, *args, **kwargs):
        # to avoid importing time everywhere when debugging
        time.sleep(*args, **kwargs)

    def compose_output_title(
        self,
        output_title=None,
        source_paths=None,
        is_merge_output=None,
        title_from_source_stem_fn=None,
        title_from_source_path_fn=None,
        default_output_title=None,
    ):
        if output_title:
            output_title = output_title.strip()
        elif not source_paths:
            output_title = default_output_title
        else:
            if not title_from_source_stem_fn:
                title_from_source_stem_fn = lambda x: x
            if not title_from_source_path_fn:
                title_from_source_path_fn = lambda path: title_from_source_stem_fn(pathlib.Path(path).stem)
            if len(source_paths) == 1:
                output_title = title_from_source_path_fn(source_paths[0])
            elif is_merge_output and len(source_paths) > 1:
                 output_title = (
                    title_from_source_path_fn(source_paths[0]) + "+others"
                )
        self.output_title = output_title
        return self.output_title

class JsonListFile:
    def __init__(
        self,
        file_handle,
    ):
        self.filename = filename
        self.file_handle = None

    def __enter__(self):
        # self.file_handle = open(self.filename, 'w')
        self.file_handle.write('[')
        self.first_item = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file_handle.write(']')
        # self.file_handle.close()

    def write(self, item):
        if not self.first_item:
            self.file_handle.write(',')
        else:
            self.first_item = False
        json.dump(item, self.file_handle)

class DataStore:
    def __init__(self, file_handle, config_d):
        self.file_handle = file_handle
        self.is_preamble_written = False
        self.configure(config_d)

    def configure(self, config_d=None):
        if not config_d:
            config_d = {}
        self.delimiter = config_d.get("delimiter", "\t")
        self.is_write_preamble = config_d.get("is_write_header", True)

    @property
    def path(self):
        if not hasattr(self, "_path") or self._path is None:
            self._path = pathlib.Path(self.file_handle.name)
        return self._path

    def write_preamble(self, preamble):
        self.file_handle.write(preamble)
        self.file_handle.write("\n")
        self.is_preamble_written = True

    def write_v(self, data_v):
        data_str = self.delimiter.join(f"{v}" for v in data_v)
        self.file_handle.write(data_str)
        self.file_handle.write("\n")

    def write_d(self, data_d):
        if not self.is_preamble_written and self.is_write_preamble:
            header = self.delimiter.join(f"{k}" for k in data_d.keys())
            self.write_preamble(preamble=header)
        self.write_v(data_d.values())

    def close(self):
        self.file_handle.close()

