from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import __version__


def repository_root() -> Path:
    configured = os.environ.get("N3KOMATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    cwd = Path.cwd().resolve()
    candidates = [*Path(__file__).resolve().parents, cwd, *cwd.parents]
    for candidate in candidates:
        if ((candidate / "Makefile").is_file() and
                (candidate / "engine").is_dir() and (candidate / "models").is_dir()):
            return candidate
    return cwd


def load_profiles(root: Path) -> dict[str, dict[str, Any]]:
    app_root = Path(__file__).resolve().parents[2]
    config_dirs = [
        app_root / "config" / "profiles",
        Path(sys.prefix) / "share" / "n3komata" / "config" / "profiles",
    ]
    for config_dir in config_dirs:
        paths = [(channel, config_dir / f"{channel}.toml")
                 for channel in ("release", "experimental")]
        existing = [(channel, path) for channel, path in paths if path.is_file()]
        if existing:
            profiles: dict[str, dict[str, Any]] = {}
            for channel, path in existing:
                try:
                    data = tomllib.loads(path.read_text(encoding="utf-8"))
                except OSError as exc:
                    raise RuntimeError(f"Cannot read profile file: {path} ({exc})") from exc
                entries = data.get("profiles")
                if not isinstance(entries, dict):
                    raise RuntimeError(f"No profiles found in {path}")
                for profile_id, profile in entries.items():
                    if profile_id in profiles:
                        raise RuntimeError(f"Duplicate profile '{profile_id}' in {path}")
                    profiles[profile_id] = {**profile, "profile_channel": channel}
            if profiles:
                return profiles

    legacy_paths = [app_root / "config" / "profiles.toml",
                    Path(sys.prefix) / "share" / "n3komata" / "config" / "profiles.toml"]
    for path in legacy_paths:
        if path.is_file():
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise RuntimeError(f"Cannot read profile file: {path} ({exc})") from exc
            profiles = data.get("profiles")
            if isinstance(profiles, dict) and profiles:
                return {key: {**value, "profile_channel": "release"}
                        for key, value in profiles.items()}
    raise RuntimeError("No release or experimental profiles found")


def discover_local_profiles(root: Path, profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Add generic launch profiles for GGUFs without an explicit profile."""
    models_dir = root / "models"
    if not models_dir.is_dir():
        return profiles

    configured_models = {
        profile_path(root, str(profile["model"]))
        for profile in profiles.values()
        if profile.get("model")
    }
    cli = "third_party/llama.cpp/build/bin/llama-cli"
    server = "third_party/llama.cpp/build/bin/llama-server"
    build_dir = "third_party/llama.cpp/build"

    for candidate in sorted(models_dir.rglob("*.gguf")):
        if not candidate.is_file() or candidate.name.lower().startswith(("mmproj-", "mtp-")):
            continue
        shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", candidate.name, re.IGNORECASE)
        if shard and shard.group(1) != "00001":
            continue
        resolved = candidate.resolve()
        if resolved in configured_models:
            continue
        relative = candidate.relative_to(root).as_posix()
        digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
        profile_id = f"local-{digest}"
        if profile_id in profiles:
            continue
        display_name = candidate.relative_to(models_dir).as_posix()
        api_model = f"n3komata-{profile_id}"
        common_args = [
            "--model", "{model}", "--gpu-layers", "all",
            "--ctx-size", "{context}", "--predict", "{max_tokens}",
            "--batch-size", "4", "--ubatch-size", "4",
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--jinja", "--reasoning", "{reasoning}",
        ]
        profiles[profile_id] = {
            "name": display_name,
            "description": "Local GGUF using generic Vulkan defaults (context 4096, batch 4). Add a Profile for model-specific settings.",
            "model_type": "local",
            "engine": "regular",
            "model": relative,
            "cli": cli,
            "server": server,
            "build_dir": build_dir,
            "build_targets": ["llama-cli", "llama-server"],
            "default_context": 4096,
            "default_max_tokens": 1024,
            "default_reasoning": "auto",
            "api_model": api_model,
            "cli_args": common_args,
            "server_args": [*common_args[:2], "--alias", "{api_model}", *common_args[2:], "--no-ui"],
            "env": {},
            "profile_channel": "local",
        }
    return profiles


def profile_path(root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def make_environment(profile: dict[str, Any], root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for prefix in profile.get("clear_env_prefixes", []):
        for key in tuple(env):
            if key.startswith(prefix):
                env.pop(key, None)
    for key, value in profile.get("env", {}).items():
        rendered = str(value).replace("{root}", str(root))
        path_value = Path(rendered)
        if key.endswith("_FILE") and not path_value.is_absolute():
            rendered = str((root / path_value).resolve())
        env[key] = rendered
    return env


def format_args(values: list[str], *, root: Path, profile: dict[str, Any],
                context: int, max_tokens: int, reasoning: str, engine_port: int = 0) -> list[str]:
    replacements = {
        "root": str(root),
        "model": str(profile_path(root, profile["model"])),
        "context": str(context),
        "max_tokens": str(max_tokens),
        "reasoning": reasoning,
        "api_model": str(profile.get("api_model", "n3komata-local")),
        "engine_port": str(engine_port),
    }
    return [value.format_map(replacements) for value in values]


def engine_binary(root: Path, profile: dict[str, Any], kind: str) -> Path:
    key = "cli" if kind == "cli" else "server"
    raw = profile.get(key)
    if not raw:
        raise RuntimeError(f"Profile has no {kind} engine configured")
    return profile_path(root, raw)


def check_profile(root: Path, profile_id: str, profile: dict[str, Any], kind: str) -> list[str]:
    missing: list[str] = []
    model = profile_path(root, profile["model"])
    binary = engine_binary(root, profile, kind)
    if not model.is_file():
        missing.append(f"model: {model}")
    if not binary.is_file():
        missing.append(f"{kind} engine: {binary}")
    if profile.get("engine") == "moe":
        pin_value = profile.get("env", {}).get("N3KOMATA_MOE_PIN_POPULARITY_FILE")
        if pin_value:
            pin_path = profile_path(root, str(pin_value))
            if not pin_path.is_file():
                missing.append(f"pinned Expert list: {pin_path}")
    return missing


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="n3k",
        description="N3komata: a simple BC-250 interface to the llama.cpp engine.",
    )
    ap.add_argument("--version", action="version", version=f"n3komata {__version__}")
    subs = ap.add_subparsers(dest="command", required=True)

    subs.add_parser("models", help="list release and experimental model profiles")

    show = subs.add_parser("show", help="show a profile's effective settings")
    show.add_argument("profile")

    doctor = subs.add_parser("doctor", help="check the model, engine and profile files")
    doctor.add_argument("profile", nargs="?")

    build = subs.add_parser("build", help="build the selected profile's llama.cpp engine")
    build.add_argument("profile", nargs="?", default="qwen38")
    build.add_argument("--target", choices=("cli", "server", "all"), default="all")
    build.add_argument("-j", "--jobs", type=int, default=max(1, os.cpu_count() or 1))

    for command, help_text in (("chat", "start an interactive terminal chat"),
                               ("web", "start the local N3komata chat page")):
        run = subs.add_parser(command, help=help_text)
        run.add_argument("profile")
        run.add_argument("--ctx", "--context", dest="context", type=int)
        run.add_argument("--max-tokens", type=int)
        run.add_argument("--reasoning", choices=("on", "off", "auto"))
        run.add_argument("--dry-run", action="store_true", help="print the launch plan without starting it")
        if command == "web":
            run.add_argument("--host", default="127.0.0.1", help="frontend bind address (default: loopback only)")
            run.add_argument("--port", type=int, default=9931)
    return ap


def resolve_profile(profiles: dict[str, dict[str, Any]], profile_id: str) -> dict[str, Any]:
    try:
        return profiles[profile_id]
    except KeyError:
        names = ", ".join(sorted(profiles))
        raise RuntimeError(f"Unknown profile '{profile_id}'. Available profiles: {names}") from None


def command_models(profiles: dict[str, dict[str, Any]]) -> int:
    for channel, heading in (("release", "Release profiles"),
                             ("experimental", "Experimental profiles"),
                             ("local", "Local GGUF models")):
        entries = [(key, value) for key, value in profiles.items()
                   if value.get("profile_channel", "release") == channel]
        if not entries:
            continue
        print(f"{heading}:")
        for profile_id, profile in entries:
            print(f"  {profile_id:36} {profile.get('name', '')}")
            if profile.get("description"):
                print(f"  {'':26} {profile['description']}")
    return 0


def command_show(root: Path, profile_id: str, profile: dict[str, Any]) -> int:
    model = profile_path(root, profile["model"])
    print(f"Profile: {profile_id} — {profile.get('name', profile_id)}")
    print(f"Engine:  {profile.get('engine', 'unknown')}")
    print(f"Model:   {model}")
    print(f"Context: {profile.get('default_context', 'model default')}")
    print(f"Reasoning: {profile.get('default_reasoning', 'auto')}")
    print("Launch settings: profile-managed")
    for key, value in profile.get("env", {}).items():
        shown = value
        if key.endswith("_FILE"):
            shown = profile_path(root, str(value))
        print(f"  {key}={shown}")
    return 0


def command_doctor(root: Path, profiles: dict[str, dict[str, Any]], profile_id: str | None) -> int:
    selected = [profile_id] if profile_id else sorted(profiles)
    failed = False
    for current in selected:
        profile = resolve_profile(profiles, current)
        print(f"{current}: {profile.get('name', current)}")
        for kind in ("cli", "server"):
            binary = engine_binary(root, profile, kind)
            print(f"  {'OK' if binary.is_file() else 'MISSING'} {kind}: {binary}")
            failed |= not binary.is_file()
        model = profile_path(root, profile["model"])
        print(f"  {'OK' if model.is_file() else 'MISSING'} model: {model}")
        failed |= not model.is_file()
        pin_value = profile.get("env", {}).get("N3KOMATA_MOE_PIN_POPULARITY_FILE")
        if pin_value:
            pin_path = profile_path(root, str(pin_value))
            print(f"  {'OK' if pin_path.is_file() else 'MISSING'} Expert pin list: {pin_path}")
            failed |= not pin_path.is_file()
    if shutil.which("vulkaninfo"):
        print("  OK Vulkan inspection tool: vulkaninfo")
    else:
        print("  INFO vulkaninfo is not installed; runtime Vulkan status was not checked")
    return 1 if failed else 0


def command_build(root: Path, profile_id: str, profile: dict[str, Any], target: str, jobs: int) -> int:
    build_dir = profile_path(root, profile["build_dir"])
    targets = profile.get("build_targets", [])
    if target != "all":
        targets = [f"llama-{target}"]
    if jobs < 1:
        raise RuntimeError("--jobs must be at least 1")
    if not targets:
        raise RuntimeError(f"Profile '{profile_id}' has no build targets")
    command = ["cmake", "--build", str(build_dir), "--target", *targets, f"-j{jobs}"]
    print(shlex.join(command))
    return subprocess.run(command, cwd=root, check=False).returncode


def command_chat(root: Path, profile_id: str, profile: dict[str, Any], args: argparse.Namespace) -> int:
    missing = check_profile(root, profile_id, profile, "cli")
    if missing:
        for item in missing:
            print(f"Missing {item}", file=sys.stderr)
        print(f"Build the engine with: ./n3k build {profile_id} --target cli", file=sys.stderr)
        return 2
    context = args.context or int(profile.get("default_context", 0))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(profile.get("default_max_tokens", -1))
    reasoning = args.reasoning or str(profile.get("default_reasoning", "auto"))
    binary = engine_binary(root, profile, "cli")
    command = [str(binary), *format_args(profile["cli_args"], root=root, profile=profile,
                                         context=context, max_tokens=max_tokens, reasoning=reasoning)]
    env = make_environment(profile, root)
    if args.dry_run:
        print(shlex.join(command))
        for key, value in profile.get("env", {}).items():
            print(f"{key}={env.get(key, value)}")
        return 0
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


def command_web(root: Path, profile_id: str, profile: dict[str, Any],
                profiles: dict[str, dict[str, Any]], args: argparse.Namespace) -> int:
    missing = check_profile(root, profile_id, profile, "server")
    if missing:
        for item in missing:
            print(f"Missing {item}", file=sys.stderr)
        print(f"Build the engine with: ./n3k build {profile_id} --target server", file=sys.stderr)
        return 2
    context = args.context or int(profile.get("default_context", 0))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(profile.get("default_max_tokens", -1))
    reasoning = args.reasoning or str(profile.get("default_reasoning", "auto"))
    from .web import serve_frontend

    return serve_frontend(
        root=root,
        profile_id=profile_id,
        profile=profile,
        profiles=profiles,
        context=context,
        max_tokens=max_tokens,
        reasoning=reasoning,
        context_override=args.context is not None,
        max_tokens_override=args.max_tokens is not None,
        reasoning_override=args.reasoning is not None,
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    root = repository_root()
    try:
        profiles = discover_local_profiles(root, load_profiles(root))
        if args.command == "models":
            return command_models(profiles)
        if args.command == "doctor":
            return command_doctor(root, profiles, args.profile)
        profile = resolve_profile(profiles, args.profile)
        if args.command == "show":
            return command_show(root, args.profile, profile)
        if args.command == "build":
            return command_build(root, args.profile, profile, args.target, args.jobs)
        if args.command == "chat":
            return command_chat(root, args.profile, profile, args)
        if args.command == "web":
            return command_web(root, args.profile, profile, profiles, args)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"n3k: {exc}", file=sys.stderr)
        return 2
    ap.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
