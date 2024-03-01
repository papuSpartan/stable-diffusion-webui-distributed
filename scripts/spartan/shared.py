import logging
from inspect import getsourcefile
from logging import Handler
from logging.handlers import RotatingFileHandler
from os.path import abspath
from pathlib import Path
from typing import Union
from modules.shared import cmd_opts
from pydantic import BaseModel, Field
from rich.text import Text
from rich.logging import RichHandler

extension_path = Path(abspath(getsourcefile(lambda: 0))).parent.parent.parent

# https://rich.readthedocs.io/en/stable/logging.html
LOG_LEVEL = 'DEBUG' if cmd_opts.distributed_debug else 'INFO'
logger = logging.getLogger("distributed")

class MyRichHandler(RichHandler):
    def get_level_text(self, record):
        rich_output = super().get_level_text(record)
        prefix = Text.from_markup("[bold][link=https://github.com/papuSpartan/stable-diffusion-webui-distributed][reverse]DISTRIBUTED[/reverse][/link][/bold] | ")
        return prefix+rich_output

rich_handler = MyRichHandler(
    rich_tracebacks=True,
    markup=True,
    show_time=False,
    keywords=["distributed", "Distributed", "worker", "Worker", "world", "World"]
)
logger.propagate = False  # prevent log duplication by webui since it now uses the logging module
logger.setLevel(LOG_LEVEL)
log_path = extension_path.joinpath('distributed.log')
file_handler = RotatingFileHandler(filename=log_path, maxBytes=10_000_000, backupCount=1)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.formatter = formatter

logger.addHandler(rich_handler)
logger.addHandler(file_handler)


gui_formatter = logging.Formatter('%(levelname)s - %(message)s')
class GuiHandler(Handler):
    messages = []

    def emit(self, record):
        formatted_msg = gui_formatter.format(record)
        self.messages.append(formatted_msg)
        if len(self.messages) >= 16:
            self.messages.remove(self.messages[0])

    def dump(self):
        messages = str()
        for msg in reversed(self.messages):
            messages += f"{msg}\n"
        return messages


gui_handler = GuiHandler()
logger.addHandler(gui_handler)
# end logging

warmup_samples = 2  # number of samples to do before recording a valid benchmark sample


class BenchmarkPayload(BaseModel):
    validate_assignment = True
    prompt: str = Field(default="A herd of cows grazing at the bottom of a sunny valley")
    negative_prompt: str = Field(default="")
    steps: int = Field(default=20)
    width: int = Field(default=512)
    height: int = Field(default=512)
    batch_size: int = Field(default=1)


benchmark_payload: Union[BenchmarkPayload, None] = None
