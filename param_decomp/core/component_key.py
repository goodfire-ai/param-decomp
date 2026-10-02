"""`ComponentKey`, the one identity of a decomposition component: a site and an index into
its components."""

import re
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, NonNegativeInt, Strict
from pydantic.dataclasses import dataclass

_SITE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_site_id(value: str) -> str:
    if not _SITE_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid site ID: {value!r}")
    return value


SiteId = Annotated[str, AfterValidator(validate_site_id)]
"""One URL-, path-, and component-key-safe decomposed site identifier."""


@dataclass(frozen=True, order=True, config=ConfigDict(extra="forbid"))
class ComponentKey:
    """Validated on construction and inside records; ordered by site, then index. Persisted
    as `"<site>:<index>"` (`encoded` / `decode`) or as its fields."""

    site: Annotated[SiteId, Strict()]
    component_idx: Annotated[NonNegativeInt, Strict()]

    def encoded(self) -> str:
        return f"{self.site}:{self.component_idx}"

    @classmethod
    def decode(cls, value: str) -> "ComponentKey":
        """Parse the one persisted component-key encoding at an artifact boundary."""
        site, separator, index = value.rpartition(":")
        if not separator or not site or not index.isascii() or not index.isdecimal():
            raise ValueError(f"invalid encoded component key: {value!r}")
        key = cls(site=site, component_idx=int(index))
        if key.encoded() != value:
            raise ValueError(f"non-canonical encoded component key: {value!r}")
        return key
