from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from .shared import benchmark_payload


class Worker(BaseModel):
    avg_ipm: float | None = Field(title='Average Speed', description='the speed of a device measured in ipm(images per minute)', ge=0)
    master: bool = Field(description="whether or not an instance is the master(local) node", default=False)
    address: Optional[str] = Field(default='localhost')
    port: Optional[int] = Field(default=7860, ge=0, le=65535)
    last_mpe: Optional[float] = Field(title='Last Mean Percent Error', description='The MPE of eta predictions in the last session')
    tls: Optional[bool] = Field(title='Transport Layer Security', description='Whether or not to make requests to a worker securely', default=False)
    # auth


class Config(BaseModel):
    workers: List[Dict[str, Worker]]
    benchmark_payload: Dict = Field(default=benchmark_payload, description='the payload used when benchmarking a node')