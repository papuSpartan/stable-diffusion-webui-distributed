import logging
from logging import Handler
from inspect import getsourcefile
from typing import Union
from rich.logging import RichHandler
from modules.shared import cmd_opts
from pydantic import BaseModel, Field
from os.path import abspath

from pathlib import Path
extension_path = Path(abspath(getsourcefile(lambda: 0))).parent.parent.parent

# https://rich.readthedocs.io/en/stable/logging.html
log_level = 'DEBUG' if cmd_opts.distributed_debug else 'INFO'
logger = logging.getLogger("distributed")
rich_handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_time=False,
    keywords=["distributed", "Distributed", "worker", "Worker", "world", "World"]
)
logger.propagate = False  # prevent log duplication by webui since it now uses the logging module
logger.setLevel(log_level)
log_path = extension_path.joinpath('distributed.log')
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.formatter = formatter

logger.addHandler(rich_handler)
logger.addHandler(file_handler)


class Gui_Handler(Handler):
    messages = []

    def emit(self, record):
        formatted_msg = formatter.format(record)
        self.messages.append(formatted_msg)
        if len(self.messages) >= 16:
            self.messages.remove(self.messages[0])

    def dump(self):
        messages = str()
        for msg in self.messages:
            messages += f"{msg}\n"
        return messages

gui_handler = Gui_Handler()
logger.addHandler(gui_handler)
# end logging

warmup_samples = 2  # number of samples to do before recording a valid benchmark sample


class Benchmark_Payload(BaseModel):
    validate_assignment = True
    prompt: str = Field(default="A herd of cows grazing at the bottom of a sunny valley")
    negative_prompt: str = Field(default="")
    steps: int = Field(default=20)
    width: int = Field(default=512)
    height: int = Field(default=512)
    batch_size: int = Field(default=1)
benchmark_payload: Union[Benchmark_Payload, None] = None
