import logging
from rich.logging import RichHandler
from modules.shared import cmd_opts

log_level = 'DEBUG' if cmd_opts.distributed_debug else 'INFO'
logging.basicConfig(level=log_level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

benchmark_payload: dict = {
    "prompt": "A herd of cows grazing at the bottom of a sunny valley",
    "negative_prompt": "",
    "steps": 20,
    "width": 512,
    "height": 512,
    "batch_size": 1
}
