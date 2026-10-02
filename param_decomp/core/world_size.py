"""Supported physical worlds: one node or multiple complete eight-device nodes."""

from pydantic import Field

from param_decomp.core.base_config import BaseConfig

GPUS_PER_NODE = 8


class SingleNode(BaseConfig):
    n_gpus: int = Field(ge=1, le=GPUS_PER_NODE, strict=True)

    @property
    def device_count(self) -> int:
        return self.n_gpus

    @property
    def gpus_per_node(self) -> int:
        return self.n_gpus


class MultiNode(BaseConfig):
    n_nodes: int = Field(ge=2, strict=True)

    @property
    def device_count(self) -> int:
        return self.n_nodes * GPUS_PER_NODE

    @property
    def gpus_per_node(self) -> int:
        return GPUS_PER_NODE


type WorldSize = SingleNode | MultiNode


def world_size_from_device_count(device_count: int) -> WorldSize:
    if 1 <= device_count <= GPUS_PER_NODE:
        return SingleNode(n_gpus=device_count)
    if device_count > GPUS_PER_NODE and device_count % GPUS_PER_NODE == 0:
        return MultiNode(n_nodes=device_count // GPUS_PER_NODE)
    raise ValueError(
        f"World size must be 1..{GPUS_PER_NODE} devices on one node or a multiple "
        f"of {GPUS_PER_NODE} across full nodes; got {device_count} devices"
    )
