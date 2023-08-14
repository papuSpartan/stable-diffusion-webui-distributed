import logging
from rich.logging import RichHandler
from modules.shared import cmd_opts

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
benchmark_payload: dict = {
    "prompt": "A herd of cows grazing at the bottom of a sunny valley",
    "negative_prompt": "",
    "steps": 20,
    "width": 512,
    "height": 512,
    "batch_size": 1
}
