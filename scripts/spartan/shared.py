import logging
from typing import Union
from rich.logging import RichHandler
from modules.shared import cmd_opts
from pydantic import BaseModel, Field

# https://rich.readthedocs.io/en/stable/logging.html
log_level = 'DEBUG' if cmd_opts.distributed_debug else 'INFO'
logger = logging.getLogger("distributed")
handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_time=False,
    keywords=["distributed", "Distributed", "worker", "Worker", "world", "World"]
)
logger.addHandler(handler)
logger.setLevel(log_level)
logger.propagate = False  # prevent log duplication by webui since it now uses the logging module

warmup_samples = 2  # number of samples to do before recording a valid benchmark sample


class Benchmark_Payload(BaseModel):
    prompt: str = Field(default="A herd of cows grazing at the bottom of a sunny valley")
    negative_prompt: str
    steps: int = Field(default=20)
    width: int = Field(default=512)
    height: int = Field(default=512)
    batch_size: int = Field(default=1)
benchmark_payload: Union[Benchmark_Payload, None] = None
