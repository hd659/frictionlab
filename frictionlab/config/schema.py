"""Typed, dependency-free experiment configs with all-errors-at-once validation.

A config file is one JSON (or YAML, if PyYAML is installed) mapping:

    {"experiment": "coase", "master_seed": 28, "replicates": 2, "small": true,
     "params": {"values": [150, 140], "costs": [70, 80]}}

Top-level keys are checked against `ExperimentConfig`; `params` against the
experiment's own `{name: Param}` schema (see `experiments.registry`). Every
problem is collected with its dotted key path (`params.levels[2]`) and raised
together as one `ConfigError`, so a user fixes a file in one pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

Errors = List[Tuple[str, str]]
TYPES = {"int": int, "float": (int, float), "bool": bool, "str": str}


class ConfigError(ValueError):
    """One or more config problems, each as (dotted path, message)."""

    def __init__(self, errors: Errors) -> None:
        self.errors = list(errors)
        lines = "\n".join(f"  - {p}: {m}" for p, m in self.errors)
        super().__init__(f"{len(self.errors)} config error(s):\n{lines}")

    @property
    def paths(self) -> List[str]:
        return [p for p, _ in self.errors]


@dataclass(frozen=True)
class Param:
    """Schema for one value: scalar or list of `type`, with optional bounds.

    `friction=True` marks an integer-tick friction: an int that must be >= 0.
    """
    type: str = "int"
    seq: bool = False
    length: Optional[int] = None
    min: Optional[float] = None
    max: Optional[float] = None
    nullable: bool = False
    friction: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"unknown Param type {self.type!r}; use one of {sorted(TYPES)}")
        if self.friction and self.type != "int":
            raise ValueError("friction params are integer ticks (type='int')")

    def _scalar(self, path: str, v: Any, errs: Errors) -> Any:
        ok = isinstance(v, TYPES[self.type]) and (self.type == "bool" or not isinstance(v, bool))
        if not ok:
            errs.append((path, f"expected {self.type}, got {type(v).__name__} {v!r}"))
            return v
        if self.type == "float":
            v = float(v)
        if self.friction and v < 0:
            errs.append((path, f"negative friction {v}; integer-tick frictions must be >= 0"))
        elif self.min is not None and v < self.min:
            errs.append((path, f"{v!r} is below the minimum {self.min!r}"))
        elif self.max is not None and v > self.max:
            errs.append((path, f"{v!r} is above the maximum {self.max!r}"))
        return v

    def check(self, path: str, v: Any, errs: Errors) -> Any:
        """Normalized value (floats coerced, tuples -> lists); appends problems to `errs`."""
        if v is None:
            if not self.nullable:
                errs.append((path, "must not be null"))
            return None
        if not self.seq:
            return self._scalar(path, v, errs)
        if not isinstance(v, (list, tuple)):
            errs.append((path, f"expected a list of {self.type}, got {type(v).__name__} {v!r}"))
            return v
        if not v:
            errs.append((path, "must be a non-empty list"))
        elif self.length is not None and len(v) != self.length:
            errs.append((path, f"expected {self.length} item(s), got {len(v)}"))
        return [self._scalar(f"{path}[{i}]", x, errs) for i, x in enumerate(v)]


Schema = Mapping[str, Param]


def check_mapping(data: Any, schema: Schema, path: str, errs: Errors,
                  required: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Validate `data` against `schema`: unknown keys, missing required, per-value."""
    if not isinstance(data, Mapping):
        errs.append((path or "<root>", f"expected a mapping, got {type(data).__name__}"))
        return {}
    pre = f"{path}." if path else ""
    out: Dict[str, Any] = {}
    for k in data:
        if k not in schema:
            errs.append((f"{pre}{k}", f"unknown key; allowed: {', '.join(schema)}"))
    for k in required:
        if k not in data:
            errs.append((f"{pre}{k}", "required key is missing"))
    for k, p in schema.items():                       # schema order -> deterministic output
        if k in data:
            out[k] = p.check(f"{pre}{k}", data[k], errs)
    return out


def validate_params(params: Any, schema: Schema, path: str = "params") -> Dict[str, Any]:
    errs: Errors = []
    out = check_mapping(params, schema, path, errs)
    if errs:
        raise ConfigError(errs)
    return out


TOP: Dict[str, Param] = {
    "experiment": Param("str", doc="registered experiment name"),
    "master_seed": Param("int", min=0, nullable=True, doc="null -> experiment default"),
    "n_workers": Param("int", min=1),
    "replicates": Param("int", min=1, nullable=True, doc="null -> experiment default"),
    "small": Param("bool", doc="start from the experiment's small (fast) defaults"),
    "output": Param("str", nullable=True, doc="output directory for reports"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    params: Dict[str, Any] = field(default_factory=dict)
    master_seed: Optional[int] = None
    n_workers: int = 1
    replicates: Optional[int] = None
    small: bool = False
    output: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any, param_schema: Optional[Schema] = None) -> "ExperimentConfig":
        """Validate top level (and `params` if a schema is given); raise all errors at once."""
        errs: Errors = []
        top = check_mapping({k: v for k, v in data.items() if k != "params"}
                            if isinstance(data, Mapping) else data, TOP, "", errs, ("experiment",))
        raw = data.get("params", {}) if isinstance(data, Mapping) else {}
        if param_schema is not None:
            params = check_mapping(raw, param_schema, "params", errs)
        elif isinstance(raw, Mapping):
            params = dict(raw)
        else:
            errs.append(("params", f"expected a mapping, got {type(raw).__name__}"))
            params = {}
        if errs:
            raise ConfigError(errs)
        return cls(params=params, **top)

    def to_dict(self) -> Dict[str, Any]:
        return {"experiment": self.experiment, "master_seed": self.master_seed,
                "n_workers": self.n_workers, "replicates": self.replicates, "small": self.small,
                "output": self.output, "params": json.loads(json.dumps(self.params))}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def loads_config(text: str, fmt: str = "json") -> Dict[str, Any]:
    """Parse config text; `fmt` is 'json' or 'yaml' (YAML needs PyYAML)."""
    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError([("<file>", f"invalid JSON: {e}")]) from None
    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ConfigError([("<file>", "YAML configs need PyYAML (pip install pyyaml); "
                                          "or use a .json config")]) from None
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError([("<file>", f"invalid YAML: {e}")]) from None
    raise ConfigError([("<file>", f"unsupported config format {fmt!r}; use json or yaml")])


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a .json / .yaml / .yml config file into a raw mapping (unvalidated)."""
    p = Path(path)
    fmt = {".json": "json", ".yaml": "yaml", ".yml": "yaml"}.get(p.suffix.lower())
    if fmt is None:
        raise ConfigError([("<file>", f"unsupported config extension {p.suffix!r}; "
                                      "use .json, .yaml or .yml")])
    return loads_config(p.read_text(encoding="utf-8"), fmt)
