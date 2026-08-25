"""Content-based compatibility identities and creation provenance.

Cache compatibility is derived from result-affecting source and dependency
content.  Repository state is recorded separately as creation provenance and
never participates in cache UIDs.
"""

from __future__ import annotations

import codecs
import hashlib
import importlib.util
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple


DATA_IMPLEMENTATION_MODULES = (
    ("load_dataset", "load_dataset"),
    ("preprocessing", "preprocessing"),
    ("filtering", "filtering"),
)

WEIGHT_IMPLEMENTATION_MODULES = (
    ("models", "models"),
    ("run", "run"),
    ("utils", "utils"),
    ("data_augmentation", "data_augmentation"),
)

DATA_DEPENDENCIES = (
    "numpy",
    "scipy",
    "neurokit2",
    "wfdb",
    "pandas",
)

WEIGHT_DEPENDENCIES = (
    "torch",
    "torchvision",
    "scikit-learn",
)


class ImplementationSourceError(RuntimeError):
    """Raised when required result-affecting Python source cannot be read."""


def canonical_json_bytes(value) -> bytes:
    """Return deterministic UTF-8 JSON bytes for identity-compatible values."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def canonical_json_sha256(value) -> str:
    """Return the SHA-256 digest of canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonicalize_python_source(source_bytes: bytes) -> bytes:
    """Normalize UTF-8 Python source for cross-platform content hashing."""
    if not isinstance(source_bytes, bytes):
        raise TypeError("Python source must be supplied as bytes.")

    if source_bytes.startswith(codecs.BOM_UTF8):
        source_bytes = source_bytes[len(codecs.BOM_UTF8):]

    source_text = source_bytes.decode("utf-8")
    source_text = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return source_text.encode("utf-8")


def python_source_identity(logical_name: str, source_bytes: bytes) -> dict:
    """Hash canonical source together with its logical, path-free name."""
    canonical_source = canonicalize_python_source(source_bytes)
    digest_input = {
        "logical_name": logical_name,
        "source": canonical_source.decode("utf-8"),
    }
    return {
        "logical_name": logical_name,
        "sha256": canonical_json_sha256(digest_input),
    }


def read_module_source_bytes(module_name: str) -> bytes:
    """Read module source through its loader, with a filesystem fallback.

    Loader ``get_data`` supports both normal source loaders and zipimport.
    ``get_source`` is the secondary loader route.  Direct filesystem reading
    is used only when the resolved origin is a real file.
    """
    try:
        specification = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as error:
        raise ImplementationSourceError(
            f"Required implementation source for {module_name!r} "
            f"could not be resolved: {error}"
        ) from error

    if specification is None:
        raise ImplementationSourceError(
            f"Required implementation source for {module_name!r} "
            "could not be resolved."
        )

    loader = specification.loader
    origin = specification.origin
    failures = []

    if loader is not None and origin and hasattr(loader, "get_data"):
        try:
            source_bytes = loader.get_data(origin)
            if isinstance(source_bytes, bytes):
                return source_bytes
            failures.append("loader get_data returned non-bytes content")
        except (OSError, ImportError, AttributeError) as error:
            failures.append(f"loader get_data failed: {error}")

    if loader is not None and hasattr(loader, "get_source"):
        try:
            source_text = loader.get_source(module_name)
            if isinstance(source_text, str):
                return source_text.encode("utf-8")
            failures.append("loader get_source returned no source")
        except (OSError, ImportError, AttributeError) as error:
            failures.append(f"loader get_source failed: {error}")

    if origin and origin not in {"built-in", "frozen"}:
        try:
            origin_path = Path(origin)
            if origin_path.is_file():
                return origin_path.read_bytes()
            failures.append("resolved origin is not a readable file")
        except OSError as error:
            failures.append(f"filesystem fallback failed: {error}")

    detail = "; ".join(failures) or "no source-reading route was available"
    raise ImplementationSourceError(
        f"Required implementation source for {module_name!r} "
        f"could not be read: {detail}."
    )


def build_implementation_group(
    group_name: str,
    modules: Sequence[Tuple[str, str]],
    source_reader: Optional[Callable[[str], bytes]] = None,
) -> dict:
    """Build named component and aggregate source identities for a group."""
    if source_reader is None:
        source_reader = read_module_source_bytes

    components = {}
    for logical_name, module_name in modules:
        try:
            source_bytes = source_reader(module_name)
        except ImplementationSourceError:
            raise
        except Exception as error:
            raise ImplementationSourceError(
                f"Required implementation source for {module_name!r} "
                f"could not be read: {error}"
            ) from error

        try:
            components[logical_name] = python_source_identity(
                logical_name,
                source_bytes,
            )
        except (TypeError, UnicodeError) as error:
            raise ImplementationSourceError(
                f"Required implementation source for {module_name!r} "
                f"could not be canonicalized as UTF-8 Python source: {error}"
            ) from error

    aggregate_input = {
        name: component["sha256"]
        for name, component in components.items()
    }
    return {
        "group": group_name,
        "components": components,
        "aggregate_sha256": canonical_json_sha256(aggregate_input),
    }


def build_data_implementation_identity(
    source_reader: Optional[Callable[[str], bytes]] = None,
) -> dict:
    """Return the data-loader/preprocessing implementation identity."""
    return build_implementation_group(
        "data",
        DATA_IMPLEMENTATION_MODULES,
        source_reader=source_reader,
    )


def build_weight_implementation_identity(
    source_reader: Optional[Callable[[str], bytes]] = None,
) -> dict:
    """Return the aggregate data plus model/training implementation identity."""
    data_identity = build_data_implementation_identity(
        source_reader=source_reader,
    )
    training_identity = build_implementation_group(
        "weight_training",
        WEIGHT_IMPLEMENTATION_MODULES,
        source_reader=source_reader,
    )
    aggregate_input = {
        "data_implementation": data_identity["aggregate_sha256"],
        "weight_training_implementation": training_identity["aggregate_sha256"],
    }
    return {
        "data_implementation": data_identity,
        "weight_training_implementation": training_identity,
        "aggregate_sha256": canonical_json_sha256(aggregate_input),
    }


def build_dependency_group(
    group_name: str,
    distribution_names: Iterable[str],
    version_resolver: Optional[Callable[[str], str]] = None,
) -> dict:
    """Build a canonical installed-distribution mapping and aggregate digest."""
    if version_resolver is None:
        version_resolver = metadata.version

    versions = {}
    for distribution_name in distribution_names:
        try:
            versions[distribution_name] = version_resolver(distribution_name)
        except metadata.PackageNotFoundError:
            versions[distribution_name] = "unavailable"

    return {
        "group": group_name,
        "distributions": versions,
        "aggregate_sha256": canonical_json_sha256(versions),
    }


def build_data_dependency_identity(
    version_resolver: Optional[Callable[[str], str]] = None,
) -> dict:
    """Return dependency identity for data loading and preprocessing."""
    return build_dependency_group(
        "data",
        DATA_DEPENDENCIES,
        version_resolver=version_resolver,
    )


def build_weight_dependency_identity(
    version_resolver: Optional[Callable[[str], str]] = None,
) -> dict:
    """Return complete data plus training dependency identity."""
    data_identity = build_data_dependency_identity(
        version_resolver=version_resolver,
    )
    training_identity = build_dependency_group(
        "weight_training",
        WEIGHT_DEPENDENCIES,
        version_resolver=version_resolver,
    )
    aggregate_input = {
        "data_dependencies": data_identity["aggregate_sha256"],
        "weight_training_dependencies": training_identity["aggregate_sha256"],
    }
    return {
        "data_dependencies": data_identity,
        "weight_training_dependencies": training_identity,
        "aggregate_sha256": canonical_json_sha256(aggregate_input),
    }


def build_data_compatibility_identity() -> dict:
    """Return implementation and dependency content for data cache identity."""
    return {
        "implementation": build_data_implementation_identity(),
        "dependencies": build_data_dependency_identity(),
    }


def build_weight_compatibility_identity() -> dict:
    """Return implementation and dependency content for weight cache identity."""
    return {
        "implementation": build_weight_implementation_identity(),
        "dependencies": build_weight_dependency_identity(),
    }


def _run_git(repository_root: Path, arguments: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_creation_provenance(
    repository_root: Optional[os.PathLike] = None,
    git_runner: Optional[Callable[[Path, Sequence[str]], Optional[str]]] = None,
) -> dict:
    """Collect Git creation metadata without making it cache-compatible state."""
    if repository_root is None:
        repository_root = Path(__file__).resolve().parent
    else:
        repository_root = Path(repository_root)

    if git_runner is None:
        git_runner = _run_git

    commit = git_runner(repository_root, ("rev-parse", "HEAD"))
    branch = git_runner(
        repository_root,
        ("rev-parse", "--abbrev-ref", "HEAD"),
    )
    working_tree_status = git_runner(
        repository_root,
        ("status", "--porcelain"),
    )

    available_count = sum(
        value is not None
        for value in (
            commit,
            branch,
            working_tree_status,
        )
    )
    if available_count == 3:
        status = "available"
    elif available_count:
        status = "partial"
    else:
        status = "unavailable"

    return {
        "git": {
            "status": status,
            "commit": commit or "unavailable",
            "branch": branch or "unavailable",
            "dirty": (
                bool(working_tree_status)
                if working_tree_status is not None
                else "unavailable"
            ),
        }
    }
