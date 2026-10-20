# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load the curated T-Head autotune pool.

The file format matches the Iluvatar GDN2 ``tune_configs.yaml``: a JSON mapping
from kernel name to ``{"META": {...}, "num_warps": ..., "num_stages": ...}``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import triton

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "tune_configs.yaml"


@lru_cache(maxsize=1)
def _load_tuned_config() -> dict[str, list[dict[str, Any]]]:
    """Read the backend's JSON-compatible tune_configs.yaml."""
    if not _CONFIG_PATH.is_file():
        return {}
    data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Autotune config must be a mapping: {_CONFIG_PATH}")
    return data


def autotune_configs(
    kernel_name: str,
    defaults: list[triton.Config],
) -> list[triton.Config]:
    """Return the measured yaml pool, or the full search list when asked.

    ``FLAG_ATTN_THEAD_GDR_TUNED_CONFIGS`` defaults to ``1`` and must be set
    before this module is imported. ``0`` keeps ``defaults`` so a new search
    can replace the yaml entries.
    """
    if os.environ.get("FLAG_ATTN_THEAD_GDR_TUNED_CONFIGS", "1") == "0":
        return defaults

    entries = _load_tuned_config().get(kernel_name)
    if not entries:
        return defaults

    configs: list[triton.Config] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{kernel_name}[{index}] must be a mapping")
        meta = entry.get("META", {})
        if not isinstance(meta, dict):
            raise ValueError(f"{kernel_name}[{index}].META must be a mapping")
        kwargs = {
            name: entry[name]
            for name in ("num_warps", "num_stages", "num_ctas", "maxnreg")
            if name in entry
        }
        configs.append(triton.Config(meta, **kwargs))
    return configs
