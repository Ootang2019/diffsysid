#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _pop_option(argv: list[str], flag: str):
    remaining: list[str] = []
    value = None
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == flag:
            if i + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value.")
            value = argv[i + 1]
            i += 2
            continue
        if item.startswith(flag + "="):
            value = item.split("=", 1)[1]
            i += 1
            continue
        remaining.append(item)
        i += 1
    return value, remaining


def _config_to_argv(config: dict, *, positional_keys: tuple[str, ...] = ()) -> list[str]:
    argv: list[str] = []
    for key in positional_keys:
        value = config.get(key)
        if value is not None:
            argv.append(str(value))
    for key, value in config.items():
        if key in positional_keys or key.startswith("_") or value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, list):
            argv.append(flag)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([flag, str(value)])
    return argv


def resolve_config_argv(
    argv: list[str] | None,
    *,
    script_file: str,
    default_config_name: str,
    config_section: str | None = None,
    positional_keys: tuple[str, ...] = (),
) -> list[str]:
    argv_in = list(sys.argv[1:] if argv is None else argv)
    config_path_arg, argv_in = _pop_option(argv_in, "--config")
    config_section_arg, argv_in = _pop_option(argv_in, "--config-section")
    section = config_section_arg or config_section

    config_path: Path | None = None
    if config_path_arg is not None:
        config_path = Path(config_path_arg)
    elif len(argv_in) == 0:
        default_path = Path(script_file).with_name(default_config_name)
        if default_path.exists():
            config_path = default_path

    if config_path is None:
        return argv_in

    config_data = json.loads(config_path.read_text())
    if section is not None:
        if section not in config_data:
            raise KeyError(f"Config section '{section}' not found in {config_path}")
        config_data = config_data[section]
    if not isinstance(config_data, dict):
        raise TypeError(f"Config payload in {config_path} must be a JSON object.")
    return _config_to_argv(config_data, positional_keys=positional_keys) + argv_in
