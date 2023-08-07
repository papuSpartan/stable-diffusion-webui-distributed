from pydantic import BaseModel, Field
from typing import List, Dict
from .shared import benchmark_payload


class Worker(BaseModel):
    avg_ipm: float | None
    master: bool = False
    address: str | None
    port: int = 7860
    last_mpe: float | None
    tls: bool


class Config(BaseModel):
    workers: List[Dict[str, Worker]]
    benchmark_payload: Dict = Field(default=benchmark_payload)