"""joblib's three conveniences as readable pure Python (utils/compute).

Absorption of ``joblib/joblib`` (reviewed file by file at commit
``878c9e0``, v1.7.dev0): ``hashing.py``'s ``Hasher``/``NumpyHasher``
(the value-semantics hashing -- strings hash by value, functions by
module+qualname reference, ndarray bytes plus a ``(dtype, shape,
strides)`` tag, dict keys sorted, sets via a sorted wrapper),
``memory.py``'s ``Memory``/``MemorizedFunc`` with ``func_inspect.py``'s
``filter_args`` and ``_store_backends.py``'s ``FileSystemStoreBackend``
(the ``<location>/<module>/<func>/<hash[:3]>/<hash[3:]>/output.pkl``
tree, the ``metadata.json`` with ``duration``/``time``/``input_args``,
the code-change wipe, the auto ``.gitignore``), ``parallel.py``'s
``Parallel``/``delayed`` with ``_parallel_backends.py``'s
``AutoBatchingMixin`` (``MIN_IDEAL_BATCH_DURATION = 0.2`` s,
``MAX_IDEAL_BATCH_DURATION = 2`` s, the doubling-capped batch growth,
the 0.8/0.2 exponentially weighted duration) and ``_utils.py``'s
AST-safe ``eval_expr`` for ``pre_dispatch``, and ``numpy_pickle.py`` +
``compressor.py``'s dump/load (the compressor registry with its magic
prefixes, the extension-override quirk, ``compress=True`` meaning zlib
level 3).

Honest boundaries (the differences that matter):

* :func:`hash_object` is **not byte-identical** to ``joblib.hash`` --
  joblib digests a custom protocol-3 pickle stream with its own memo
  quirks (strings un-memoized, unorderable dict keys substituted by
  recursive hashes); this module digests a canonical byte encoding
  with the same *semantics* (value vs reference, sorted keys, array
  bytes + dtype/shape/strides tag).  Two equal values hash equal, a
  rename changes a function's hash -- the contract joblib promises.
* ``Parallel`` runs a **threading backend or a sequential fallback**
  only -- no loky process forking, no memmapped reducer, no dask.
  ``prefer``/``require`` are honored as far as threading allows
  (``require="sharedmem"`` forces threads, as in joblib).
* ``dump``/``load`` write a **plain pickle stream** through the
  stdlib codecs -- joblib's ``NumpyArrayWrapper`` fast path (raw
  ``.npy`` bytes embedded after the pickle stream, 16-byte aligned,
  memmap-able) is not replicated; numpy arrays pickle normally.
  Compression is whole-buffer, not streamed.
* ``DiskCache`` has no ``bytes_limit`` eviction, no mmap mode and no
  store-backend registry -- one filesystem backend, joblib's layout.

Examples
--------
>>> from pharmaml.utils.compute import effective_n_jobs, eval_pre_dispatch
>>> effective_n_jobs(1), effective_n_jobs(None)
(1, 1)
>>> effective_n_jobs(-1) == effective_n_jobs(-2) + 1   # -1 = all CPUs
True
>>> effective_n_jobs(0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: n_jobs == 0 in Parallel has no meaning
>>> eval_pre_dispatch("2 * n_jobs", 4)     # joblib's default expression
8
>>> eval_pre_dispatch("2 ** 99", 1)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: pre_dispatch expression exceeds the magnitude cap (10**6)

>>> from pharmaml.utils.compute import Parallel, delayed
>>> from math import sqrt
>>> Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
>>> Parallel(n_jobs=1)(delayed(sqrt)(i ** 2) for i in range(5))  # sequential path
[0.0, 1.0, 2.0, 3.0, 4.0]
>>> Parallel(n_jobs=2, return_as="generator")(delayed(sqrt)(i ** 2) for i in range(3)).__next__()
0.0
>>> Parallel(n_jobs=2, return_as="tuple")(delayed(sqrt)(4) for _ in range(1))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: return_as must be one of 'list', 'generator' or 'generator_unordered', got 'tuple'
"""

from __future__ import annotations

import ast
import bz2
import functools
import gzip
import hashlib
import inspect
import json
import lzma
import math
import os
import pickle
import shutil
import time
import types
import uuid
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError, PersistenceError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "DiskCache",
    "Parallel",
    "cpu_count",
    "delayed",
    "dump",
    "effective_n_jobs",
    "eval_pre_dispatch",
    "format_time",
    "hash_object",
    "load",
    "memstr_to_bytes",
    "short_format_time",
]


# --------------------------------------------------------------------- #
# cpu counting + time formatting (joblib's _parallel_backends / logger)
# --------------------------------------------------------------------- #

def cpu_count() -> int:
    """The number of usable CPUs (joblib's ``cpu_count``, trimmed).

    joblib honours the cgroup cpuset before trusting
    ``os.cpu_count()``; this port trusts the OS (the container
    trimming is an operational nicety, not a semantic one).  Returns
    at least one CPU no matter what the environment reports.
    """
    count = os.cpu_count() or 1
    return max(int(count), 1)


def effective_n_jobs(n_jobs: int | None) -> int:
    """Resolve ``n_jobs`` the way joblib's backends do.

    ``None`` means unset (one worker); ``0`` is the documented
    nonsense that raises; negatives count down from all CPUs
    (``-1`` -> all, ``-2`` -> all but one, ``n`` -> ``cpus + 1 + n``
    floored at one); ``1`` short-circuits to the sequential path.

    Examples
    --------
    >>> from pharmaml.utils.compute import cpu_count, effective_n_jobs
    >>> effective_n_jobs(4)
    4
    >>> effective_n_jobs(-2) == max(cpu_count() - 1, 1)
    True
    >>> effective_n_jobs(-99)          # the floor
    1
    """
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise DataValidationError("n_jobs == 0 in Parallel has no meaning")
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    return n_jobs


def format_time(seconds: float) -> str:
    """``"12.3s, 0.2min"`` -- joblib's ``logger.format_time``."""
    return f"{seconds:.1f}s, {seconds / 60.0:.1f}min"


def short_format_time(seconds: float) -> str:
    """joblib's ``logger.short_format_time`` (minutes past 60 s)."""
    if seconds > 60.0:
        return f"{seconds / 60.0:4.1f}min"
    return f"{seconds:5.1f}s"


def memstr_to_bytes(text: str | int) -> int:
    """Parse ``"10G"``/``"500M"``/``"4K"`` into bytes (joblib's disk.py).

    Examples
    --------
    >>> from pharmaml.utils.compute import memstr_to_bytes
    >>> memstr_to_bytes("1K"), memstr_to_bytes("10M"), memstr_to_bytes("2G")
    (1024, 10485760, 2147483648)
    >>> memstr_to_bytes(4096)
    4096
    >>> memstr_to_bytes("10Z")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: wrong size format: '10Z' (units are K, M or G)
    """
    if isinstance(text, (int, float)):
        return int(text)
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    cleaned = str(text).strip().upper()
    if cleaned and cleaned[-1] in units and cleaned[:-1].strip().isdigit():
        return int(cleaned[:-1].strip()) * units[cleaned[-1]]
    if cleaned.isdigit():
        return int(cleaned)
    raise DataValidationError(
        f"wrong size format: {text!r} (units are K, M or G)"
    )


# --------------------------------------------------------------------- #
# hashing (joblib/hashing.py concepts)
# --------------------------------------------------------------------- #

_HASH_NAMES = ("md5", "sha1")


def _canonical_chunks(obj: Any, chunks: list[bytes]) -> None:
    """Append the canonical byte encoding of ``obj`` to ``chunks``.

    The joblib semantics, rule by rule:

    * strings/bytes/numbers hash **by value** (equal values -> equal
      digest regardless of identity);
    * dicts hash with **sorted keys** (joblib sorts the pickle's
      ``_batch_setitems``; unorderable keys fall back to a canonical
      byte ordering here -- joblib substitutes the recursive hash,
      same goal, different mechanism);
    * sets hash as their **sorted elements** (joblib's
      ``_ConsistentSet``);
    * functions, builtins and classes hash **by reference** -- module
      and qualified name only, nothing of the code body (so an alias
      hashes equal and a rename invalidates);
    * numpy arrays hash as their C-order raw bytes plus a
      ``(dtype, shape, strides)`` tag -- F-order arrays contribute
      the transposed bytes but keep their own strides in the tag, so
      C and F layouts of the same numbers hash differently (a
      deliberate joblib behavior, kept);
    * anything else falls back to a protocol-3 pickle blob (joblib
      has no fallback either -- unpicklable objects raise).
    """
    if obj is None:
        chunks.append(b"N:")
    elif isinstance(obj, bool):
        chunks.append(b"b1:" if obj else b"b0:")
    elif isinstance(obj, int):
        chunks.append(f"i:{obj};".encode("ascii"))
    elif isinstance(obj, float):
        chunks.append(f"f:{obj!r};".encode("ascii"))
    elif isinstance(obj, str):
        encoded = obj.encode("utf-8")
        chunks.append(f"s:{len(encoded)}:".encode("ascii") + encoded)
    elif isinstance(obj, (bytes, bytearray)):
        payload = bytes(obj)
        chunks.append(f"y:{len(payload)}:".encode("ascii") + payload)
    elif isinstance(obj, np.generic):
        # numpy scalars keep their dtype tag (joblib pickles them with
        # it too, so np.float32(2.0) never collides with the int/float 2)
        chunks.append(f"g:{obj.dtype.str};".encode("ascii"))
        _canonical_chunks(obj.item(), chunks)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:  # nested pickling, list semantics
            _canonical_chunks(obj.tolist(), chunks)
            return
        if obj.shape == ():
            view = obj.reshape(1)
        elif obj.flags.c_contiguous:
            view = obj
        elif obj.flags.f_contiguous:
            view = obj.T
        else:
            view = np.ascontiguousarray(obj)
        chunks.append(b"a:")
        chunks.append(np.ascontiguousarray(view).view(np.uint8).tobytes())
        tag = f"t:{obj.dtype.str}|{obj.shape}|{obj.strides};".encode("ascii")
        chunks.append(tag)
    elif isinstance(obj, np.dtype):
        chunks.append(b"_HASHED_DTYPE")
        chunks.append(pickle.dumps(obj, protocol=3))
    elif isinstance(obj, dict):
        chunks.append(b"D:%d:" % len(obj))
        try:
            ordered = sorted(obj.items())
        except TypeError:  # unorderable mixed keys: canonical order
            ordered = sorted(obj.items(), key=lambda kv: _digest_chunks([kv[0]]))
        for key, value in ordered:
            _canonical_chunks(key, chunks)
            _canonical_chunks(value, chunks)
    elif isinstance(obj, (set, frozenset)):
        chunks.append(b"S:%d:" % len(obj))
        try:
            ordered = sorted(obj)
        except TypeError:
            ordered = sorted(obj, key=lambda element: _digest_chunks([element]))
        for element in ordered:
            _canonical_chunks(element, chunks)
    elif isinstance(obj, (list, tuple)):
        chunks.append(b"L:" if isinstance(obj, list) else b"T:")
        chunks.append(b"%d:" % len(obj))
        for item in obj:
            _canonical_chunks(item, chunks)
    elif isinstance(
        obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)
    ):
        module = getattr(obj, "__module__", None) or "__builtin__"
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        chunks.append(f"F:{module}.{qualname};".encode())
    else:
        try:
            blob = pickle.dumps(obj, protocol=3)
        except Exception as exc:
            raise DataValidationError(
                f"cannot hash {type(obj).__name__} object: {exc}"
            ) from exc
        chunks.append(b"P:")
        chunks.append(blob)


def _digest_chunks(chunks: Sequence[bytes]) -> bytes:
    joined = b"".join(chunks)
    return hashlib.md5(joined, usedforsecurity=False).digest()


def hash_object(obj: Any, hash_name: str = "md5") -> str:
    """A deterministic content hash (joblib's ``hash`` semantics).

    Returns the lowercase hex digest -- 32 characters for md5, 40 for
    sha1 -- of the canonical encoding built by
    :func:`_canonical_chunks` (see its docstring for the exact
    value-vs-reference rules).  Unlike Python's builtin ``hash`` this
    is stable across processes and interpreters: no ``PYTHONHASHSEED``
    involvement, ever.

    Examples
    --------
    >>> from pharmaml.utils.compute import hash_object
    >>> len(hash_object("a")), len(hash_object("a", "sha1"))
    (32, 40)
    >>> hash_object("a") == hash_object("a")           # by value
    True
    >>> hash_object({"a": 1, "b": 2}) == hash_object({"b": 2, "a": 1})
    True
    >>> hash_object([1, 2]) == hash_object([2, 1])     # order matters
    False
    >>> hash_object(set((1, 2, 3))) == hash_object(set((3, 2, 1)))
    True
    >>> hash_object("nope", "sha256")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: hash_name must be one of ('md5', 'sha1'), got 'sha256'
    >>> def twin(): return 1
    >>> alias = twin
    >>> hash_object(twin) == hash_object(alias)        # reference semantics
    True
    >>> def twin(): return 2      # same name, new body: new reference? no --
    >>> # the module+qualname are unchanged, so the digest is too (joblib
    >>> # hashes functions by reference, never by body).
    >>> import numpy as np
    >>> arr = np.arange(4)
    >>> hash_object(arr) == hash_object(arr.copy())
    True
    >>> grid = np.arange(6).reshape(2, 3)
    >>> hash_object(grid) == hash_object(np.asfortranarray(grid))  # strides differ
    False
    >>> hash_object(np.float32(2.0)) == hash_object(2.0)           # dtype tag
    False
    """
    if hash_name not in _HASH_NAMES:
        raise DataValidationError(
            f"hash_name must be one of {_HASH_NAMES}, got {hash_name!r}"
        )
    chunks: list[bytes] = []
    _canonical_chunks(obj, chunks)
    digest = hashlib.new(hash_name, b"".join(chunks), usedforsecurity=False)
    return digest.hexdigest()


# --------------------------------------------------------------------- #
# argument filtering (joblib/func_inspect.py's filter_args)
# --------------------------------------------------------------------- #

def filter_args(
    func: Callable[..., Any],
    ignore: Sequence[str] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map a call onto ``{parameter name: value}`` (joblib's rules).

    Positional args land on their parameter names, keywords by name,
    defaults are filled in, ``*args``/``**kwargs`` overflow collects
    under the literal keys ``"*"`` and ``"**"`` (joblib's exact
    convention), and every name in ``ignore`` is dropped before the
    dict is hashed -- that is how ``Memory.cache(ignore=["debug"])``
    keeps a debug flag out of the cache key.

    Examples
    --------
    >>> from pharmaml.utils.compute import filter_args
    >>> def f(x, y=10, *rest, **flags): pass
    >>> filter_args(f, None, (1,), {"z": 2})
    {'x': 1, 'y': 10, '*': [], '**': {'z': 2}}
    >>> filter_args(f, ["y"], (1,))
    {'x': 1, '*': [], '**': {}}
    >>> filter_args(f, ["missing"], (1,))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: ignore refers to unknown parameters: ['missing'] (valid: 'x', 'y', '*', '**')
    >>> filter_args(f, "y", (1,))
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: ignore must be a list of parameter names, got a plain string
    """
    if isinstance(ignore, str):
        raise DataValidationError(
            "ignore must be a list of parameter names, got a plain string"
        )
    kwargs = dict(kwargs or {})
    try:
        signature = inspect.signature(func)
        bound = signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise DataValidationError(f"cannot inspect the call: {exc}") from exc
    bound.apply_defaults()
    filtered: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        parameter = signature.parameters[name]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            filtered["*"] = list(value)
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            filtered["**"] = dict(value)
        else:
            filtered[name] = value
    if ignore:
        unknown = [name for name in ignore if name not in filtered]
        if unknown:
            valid = ", ".join(repr(name) for name in filtered) or "<none>"
            raise DataValidationError(
                f"ignore refers to unknown parameters: {unknown} (valid: {valid})"
            )
        for name in ignore:
            filtered.pop(name, None)
    return filtered


def _code_fingerprint(func: Callable[..., Any]) -> str:
    """A stable fingerprint of a function's code object.

    joblib's ``_code_fingerprint`` digest: ``co_name``,
    ``co_code.hex()``, the name tuples, the arg counts and flags, and
    a recursive repr of the constants -- deliberately *excluding*
    ``co_firstlineno``, ``co_linetable`` and ``co_filename`` so moving
    the function inside its file (or editing lines above it) does not
    invalidate the cache.
    """
    code = getattr(func, "__code__", None)
    if code is None:  # builtins: fall back to the reference name
        return hash_object(func)
    parts: list[Any] = [
        code.co_name,
        code.co_code.hex(),
        sorted(code.co_names),
        sorted(code.co_varnames),
        sorted(code.co_cellvars),
        sorted(code.co_freevars),
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_flags,
        _repr_consts(code.co_consts),
    ]
    exception_table = getattr(code, "co_exceptiontable", None)
    if exception_table is not None:
        parts.append(exception_table.hex())
    return hash_object(parts)


def _repr_consts(consts: tuple[Any, ...]) -> list[str]:
    """Recursive repr of code constants (joblib's own helper)."""
    rendered: list[str] = []
    for const in consts:
        if isinstance(const, tuple):
            rendered.append("(" + ", ".join(_repr_consts(const)) + ")")
        else:
            rendered.append(repr(const))
    return rendered


# --------------------------------------------------------------------- #
# the disk cache (joblib/memory.py's Memory)
# --------------------------------------------------------------------- #

class _MemorizedFunc:
    """The ``Memory.cache`` wrapper: one cached function, one tree.

    On-disk layout (joblib's, 1.7 sharding included)::

        <location>/<module dirs>/<funcname>/func_fingerprint.json
        <location>/<module dirs>/<funcname>/<hash[:3]>/<hash[3:]>/
            output.pkl        # the return value (our dump/load)
            metadata.json     # {"duration", "time", "input_args"}

    The argument hash covers :func:`filter_args` of the call (minus
    the ``ignore``d names); the *function identity* is tracked
    separately by a code fingerprint -- when the body changes, the
    whole function directory is wiped (joblib's exact invalidation
    rule) and the result recomputed.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        location: Path,
        *,
        ignore: Sequence[str] | None = None,
        verbose: int = 0,
        compress: bool | int | str = False,
    ) -> None:
        self.func = func
        self._location = location
        self._ignore = list(ignore) if ignore else None
        self._verbose = int(verbose)
        self._compress = compress
        self._func_dir = self._location.joinpath(
            *self._module_parts(func), getattr(func, "__name__", "callable")
        )
        self._checked_identity = False

    # -- layout helpers --------------------------------------------- #

    @staticmethod
    def _module_parts(func: Callable[..., Any]) -> list[str]:
        module = getattr(func, "__module__", None) or "builtin"
        return [part for part in module.replace(".", os.sep).split(os.sep) if part]

    def _ensure_identity(self) -> None:
        """Wipe this function's cache when its code changed (once per process)."""
        if self._checked_identity:
            return
        self._checked_identity = True
        self._func_dir.mkdir(parents=True, exist_ok=True)
        marker = self._func_dir / "func_fingerprint.json"
        current = _code_fingerprint(self.func)
        if marker.is_file():
            try:
                stored = json.loads(marker.read_text(encoding="utf-8"))["fingerprint"]
            except (json.JSONDecodeError, KeyError):
                stored = None
            if stored != current:
                if self._verbose:
                    _LOG.warning(
                        "function %s changed -- flushing its cache directory",
                        getattr(self.func, "__name__", "callable"),
                    )
                self.clear(warn=False)
        marker.write_text(
            json.dumps({"fingerprint": current}, indent=2), encoding="utf-8"
        )
        # joblib also drops a .gitignore so the cache never lands in git
        gitignore = self._location / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("# Created automatically (the joblib convention).\n*\n")

    def _call_dir(self, args_id: str) -> Path:
        return self._func_dir / args_id[:3] / args_id[3:]

    def _args_id(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        return hash_object(filter_args(self.func, self._ignore, args, kwargs))

    # -- the cache protocol ------------------------------------------ #

    def check_call_in_cache(self, *args: Any, **kwargs: Any) -> bool:
        """``True`` when the exact call already has an entry (no side effects)."""
        directory = self._call_dir(self._args_id(args, kwargs))
        return (directory / "output.pkl").is_file()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_identity()
        directory = self._call_dir(self._args_id(args, kwargs))
        payload = directory / "output.pkl"
        if payload.is_file():
            try:
                return load(payload)
            except PersistenceError:  # a torn entry: recompute below
                if self._verbose:
                    _LOG.warning("cache entry %s unreadable -- recomputing", directory)
        output, _metadata = self._compute_and_persist(directory, args, kwargs)
        return output

    def call(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Force execution, persist, and return ``(output, metadata)``."""
        self._ensure_identity()
        directory = self._call_dir(self._args_id(args, kwargs))
        return self._compute_and_persist(directory, args, kwargs)

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        """``Memory.eval``: run through the cache (recompute on miss only)."""
        return self(*args, **kwargs)

    def clear(self, warn: bool = True) -> None:
        """Delete this function's cache subtree (joblib's per-func clear)."""
        if warn and self._verbose:
            _LOG.warning("Flushing the cache of %s", self._func_dir)
        if self._func_dir.is_dir():
            shutil.rmtree(self._func_dir, ignore_errors=True)
        self._func_dir.mkdir(parents=True, exist_ok=True)
        marker = self._func_dir / "func_fingerprint.json"
        marker.write_text(
            json.dumps({"fingerprint": _code_fingerprint(self.func)}, indent=2),
            encoding="utf-8",
        )

    def _compute_and_persist(
        self, directory: Path, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        started = time.time()
        output = self.func(*args, **kwargs)
        duration = time.time() - started
        directory.mkdir(parents=True, exist_ok=True)
        # concurrency-safe write: temp file in the same dir, atomic rename
        temporary = directory / f"output.pkl.{uuid.uuid4().hex}-{os.getpid()}"
        dump(output, temporary, compress=self._compress)
        os.replace(temporary, directory / "output.pkl")
        argument_dict = filter_args(self.func, self._ignore, args, kwargs)
        metadata = {
            "duration": duration,
            "time": started,
            "input_args": {name: repr(value) for name, value in argument_dict.items()},
        }
        meta_temporary = directory / f"metadata.json.{uuid.uuid4().hex}"
        meta_temporary.write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        os.replace(meta_temporary, directory / "metadata.json")
        return output, metadata


class DiskCache:
    """``joblib.Memory`` -- transparent memoization to disk.

    Parameters
    ----------
    location:
        The cache root.  ``None`` disables caching entirely (calls
        pass straight through -- joblib's transparent
        ``NotMemorizedFunc``).  A *string* location gains a
        ``joblib`` subdirectory (joblib's own quirk, kept); a
        :class:`~pathlib.Path` is used exactly as given.
    verbose:
        0 silences the cache chatter, 1+ logs wipes and rebuilds.
    compress:
        Passed to :func:`dump` for the persisted outputs (``False``
        plain, ``True``/``3`` zlib level 3, ``"gzip"``/``"bz2"``/``"xz"``
        codecs).

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from pharmaml.utils.compute import DiskCache
    >>> tmp = Path(tempfile.mkdtemp())
    >>> cache = DiskCache(tmp)
    >>> calls = []
    >>> def double(x):
    ...     calls.append(x)
    ...     return 2 * x
    >>> cached = cache.cache(double)
    >>> cached(21)
    42
    >>> len(calls)
    1
    >>> cached(21)              # the HIT: no recompute
    42
    >>> len(calls)
    1
    >>> cached(20)              # a new argument: a MISS
    40
    >>> len(calls)
    2

    The metadata sidecar joblib writes for every call (the entry tree
    is ``<location>/joblib/<module dirs>/<func>/<hash[:3]>/<hash[3:]>/``;
    functions defined at a prompt live under ``__main__``):

    >>> import json
    >>> from pharmaml.utils.compute import filter_args, hash_object
    >>> args_id = hash_object(filter_args(double, None, (21,), {}))
    >>> entry = next(
    ...     path.parent for path in sorted(tmp.rglob("output.pkl"))
    ...     if json.loads((path.parent / "metadata.json").read_text())["input_args"]
    ...     == {"x": "21"}
    ... )
    >>> sorted(json.loads((entry / "metadata.json").read_text()))
    ['duration', 'input_args', 'time']
    >>> json.loads((entry / "metadata.json").read_text())["input_args"]
    {'x': '21'}
    >>> cached.check_call_in_cache(21)
    True
    >>> cached.check_call_in_cache(99)
    False

    ``ignore`` keeps an argument out of the key (joblib's debug flag):

    >>> counted = cache.cache(double, ignore=["x"])
    >>> counted(1)
    2
    >>> counted(2)              # different x, SAME key -> a HIT
    2
    >>> len(calls)
    3

    ``call`` forces a recompute and returns the metadata too:

    >>> output, metadata = cached.call(21)
    >>> (output, len(calls), sorted(metadata))
    (42, 4, ['duration', 'input_args', 'time'])

    A ``None`` location is the transparent passthrough:

    >>> passthrough = DiskCache(None)
    >>> passthrough.cache(double)(5)
    10

    >>> cached.clear(warn=False)
    >>> cached.check_call_in_cache(21)
    False
    >>> cache.clear(warn=False)
    >>> import shutil
    >>> shutil.rmtree(tmp)
    """

    def __init__(
        self,
        location: str | Path | None,
        *,
        verbose: int = 0,
        compress: bool | int | str = False,
    ) -> None:
        root: Path | None
        if isinstance(location, str):
            root = Path(location) / "joblib"  # joblib appends this for str locations
        else:
            root = Path(location) if location is not None else None
        self.location = root
        self.verbose = int(verbose)
        self.compress = compress
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            info = root / "store_backend_info.json"
            if not info.exists():
                info.write_text(
                    json.dumps(
                        {"cache_version": 1, "engine": "pharmaml.utils.compute"},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            gitignore = root / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "# Created automatically (the joblib convention).\n*\n"
                )

    def cache(
        self,
        func: Callable[..., Any] | None = None,
        *,
        ignore: Sequence[str] | None = None,
        verbose: int | None = None,
    ) -> Callable[..., Any]:
        """The decorator: wrap ``func`` with disk memoization.

        Usable bare (``@memory.cache``) or with options
        (``@memory.cache(ignore=["debug"])``), exactly like joblib.
        """
        if func is None:  # partial application: @cache(ignore=[...])

            def wrapper(inner: Callable[..., Any]) -> Callable[..., Any]:
                return self.cache(inner, ignore=ignore, verbose=verbose)

            return wrapper
        if self.location is None:  # transparent: no store, plain function
            return func
        return _MemorizedFunc(
            func,
            self.location,
            ignore=ignore,
            verbose=self.verbose if verbose is None else verbose,
            compress=self.compress,
        )

    def clear(self, warn: bool = True) -> None:
        """Wipe the whole store (joblib's ``Memory.clear``)."""
        if self.location is None:
            return
        if warn and self.verbose:
            _LOG.warning("Flushing completely the cache in %s", self.location)
        if self.location.is_dir():
            shutil.rmtree(self.location, ignore_errors=True)
        self.location.mkdir(parents=True, exist_ok=True)
        (self.location / "store_backend_info.json").write_text(
            json.dumps(
                {"cache_version": 1, "engine": "pharmaml.utils.compute"}, indent=2
            ),
            encoding="utf-8",
        )
        (self.location / ".gitignore").write_text(
            "# Created automatically (the joblib convention).\n*\n"
        )

    def eval(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """``Memory.eval``: ``func(*args)`` unless the cache is warm."""
        return self.cache(func)(*args, **kwargs)


# --------------------------------------------------------------------- #
# pre_dispatch expression evaluation (joblib/_utils.py's eval_expr)
# --------------------------------------------------------------------- #

_EXPR_MAX_LENGTH = 30
_EXPR_MAX_VALUE = 10**6


def eval_pre_dispatch(expression: str | int, n_jobs: int = 1) -> int:
    """Evaluate a ``pre_dispatch`` expression safely.

    joblib's ``eval_expr`` contract: basic arithmetic only
    (``+ - * / // % **`` and unary minus), the literal ``n_jobs``
    substituted with the effective worker count, expressions capped
    at 30 characters and results at ``10**6`` -- nothing else in the
    AST is allowed, so ``eval`` is never touched.

    Examples
    --------
    >>> from pharmaml.utils.compute import eval_pre_dispatch
    >>> eval_pre_dispatch("2 * n_jobs", 4)
    8
    >>> eval_pre_dispatch(5)
    5
    >>> eval_pre_dispatch("n_jobs", 7)
    7
    >>> eval_pre_dispatch("__import__('os')", 1)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: disallowed syntax in pre_dispatch expression: __import__('os')
    >>> eval_pre_dispatch("1 + " * 20, 1)
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: pre_dispatch expression is too long (30 characters max)
    """
    if isinstance(expression, int):
        return int(expression)
    text = str(expression).strip()
    if len(text) > _EXPR_MAX_LENGTH:
        raise DataValidationError(
            f"pre_dispatch expression is too long ({_EXPR_MAX_LENGTH} characters max)"
        )
    text = text.replace("n_jobs", str(int(n_jobs)))

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if right * math.log10(max(abs(left), 1e-9)) > math.log10(_EXPR_MAX_VALUE):
                raise DataValidationError(
                    "pre_dispatch expression exceeds the magnitude cap (10**6)"
                )
            return left**right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        raise DataValidationError(
            f"disallowed syntax in pre_dispatch expression: {text}"
        )

    try:
        tree = ast.parse(text, mode="eval")
        value = evaluate(tree)
    except SyntaxError as exc:
        raise DataValidationError(
            f"pre_dispatch expression is not valid arithmetic: {text!r}"
        ) from exc
    result = int(value)
    if not 0 <= result <= _EXPR_MAX_VALUE:
        raise DataValidationError(
            f"pre_dispatch must resolve to 0..{_EXPR_MAX_VALUE}, got {result}"
        )
    return result


# --------------------------------------------------------------------- #
# Parallel / delayed (joblib/parallel.py + AutoBatchingMixin)
# --------------------------------------------------------------------- #

def delayed(function: Callable[..., Any]) -> Callable[..., Any]:
    """Capture a call as ``(function, args, kwargs)`` for :class:`Parallel`.

    joblib's exact shape (the ``check_pickle`` argument died in
    joblib 1.0); the returned closure carries the wrapped function's
    name via ``functools.wraps``.
    """
    def delayed_function(*args: Any, **kwargs: Any) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
        return function, args, kwargs

    try:
        delayed_function = functools.wraps(function)(delayed_function)
    except AttributeError:  # some callables have no __name__
        pass
    return delayed_function


class _AutoBatcher:
    """``AutoBatchingMixin`` verbatim: the self-tuning batch size.

    Constants and rules straight from joblib's
    ``_parallel_backends.py``: a batch should take between 0.2 s
    (below which dispatch overhead is not amortized) and 2 s (above
    which stragglers starve the pool).  The batch size starts at 1,
    may at most **double** per adjustment, and the batch duration is
    an exponentially weighted average with 0.8/0.2 weights that only
    updates when the completed batch matches the current size.
    """

    MIN_IDEAL_BATCH_DURATION = 0.2
    MAX_IDEAL_BATCH_DURATION = 2.0

    def __init__(self) -> None:
        self._effective_batch_size = 1
        self._smoothed_batch_duration = 0.0

    def compute_batch_size(self) -> int:
        old_batch_size = self._effective_batch_size
        batch_duration = self._smoothed_batch_duration
        if 0 < batch_duration < self.MIN_IDEAL_BATCH_DURATION:
            ideal = int(old_batch_size * self.MIN_IDEAL_BATCH_DURATION / batch_duration)
            ideal *= 2  # limit oscillations between min and max
            batch_size = min(2 * old_batch_size, ideal)  # no runaway growth
            batch_size = max(batch_size, 1)
            self._effective_batch_size = batch_size
        elif batch_duration > self.MAX_IDEAL_BATCH_DURATION and old_batch_size >= 2:
            ideal = int(old_batch_size * self.MIN_IDEAL_BATCH_DURATION / batch_duration)
            ideal *= 2
            batch_size = max(2 * ideal, 1)  # shrink fast to limit starving
            self._effective_batch_size = batch_size
        else:
            batch_size = old_batch_size
        return batch_size

    def batch_completed(self, batch_size: int, duration: float) -> None:
        if batch_size == self._effective_batch_size:
            old = self._smoothed_batch_duration
            self._smoothed_batch_duration = 0.8 * old + 0.2 * duration


class Parallel:
    """``joblib.Parallel`` on a threading budget.

    Honored faithfully: the ``n_jobs`` normalization
    (:func:`effective_n_jobs`), ``delayed`` call tuples, the
    submission-order result guarantee (except
    ``return_as="generator_unordered"``), the ``pre_dispatch``
    arithmetic-window semantics via :func:`eval_pre_dispatch`, the
    auto-batching heuristic (:class:`_AutoBatcher`) and the verbose
    ``[Parallel(n_jobs=2)]: Done 4 out of 10 tasks`` line format.

    Honest deviations: the only backends are ``"threading"`` (the
    default here -- joblib defaults to the loky *process* executor)
    and ``"sequential"``; there is no memmapping of large arrays, no
    dask, and ``require="sharedmem"`` resolves to threads (as it does
    in joblib) while ``prefer="processes"`` logs that processes are
    unavailable and stays on threads.

    Parameters
    ----------
    n_jobs:
        Workers (``None``/1 sequential, ``-1`` all CPUs, ``-2`` all
        but one, ``0`` raises).
    return_as:
        ``"list"`` (default), ``"generator"`` (ordered, lazy) or
        ``"generator_unordered"`` (completion order).
    pre_dispatch:
        ``"all"``, an int, or an arithmetic expression such as
        joblib's default ``"2 * n_jobs"`` -- the number of tasks
        submitted before the first harvest.
    batch_size:
        ``"auto"`` (the 0.2 s/2 s self-tuner) or a positive int.
    timeout:
        Per-task seconds; a straggler raises
        :class:`concurrent.futures.TimeoutError`.

    Examples
    --------
    >>> from pharmaml.utils.compute import Parallel, delayed
    >>> from math import sqrt
    >>> Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    >>> results = Parallel(n_jobs=2, return_as="generator")(delayed(sqrt)(i * i) for i in range(4))
    >>> [round(value, 6) for value in results]
    [0.0, 1.0, 2.0, 3.0]
    >>> Parallel(n_jobs=1, verbose=0)(delayed(abs)(-value) for value in range(3))
    [0, 1, 2]
    """

    _VALID_RETURN_AS = ("list", "generator", "generator_unordered")
    _VALID_BACKENDS = ("sequential", "threading")

    def __init__(
        self,
        n_jobs: int | None = None,
        *,
        backend: str | None = None,
        return_as: str = "list",
        verbose: int = 0,
        timeout: float | None = None,
        pre_dispatch: str | int = "2 * n_jobs",
        batch_size: str | int = "auto",
        prefer: str | None = None,
        require: str | None = None,
    ) -> None:
        if return_as not in self._VALID_RETURN_AS:
            raise DataValidationError(
                "return_as must be one of 'list', 'generator' or "
                f"'generator_unordered', got {return_as!r}"
            )
        if isinstance(batch_size, str) and batch_size != "auto":
            raise DataValidationError(
                f"batch_size must be 'auto' or a positive int, got {batch_size!r}"
            )
        if isinstance(batch_size, int) and batch_size < 1:
            raise DataValidationError(
                f"batch_size must be 'auto' or a positive int, got {batch_size}"
            )
        resolved_backend = backend or "threading"
        if prefer == "threads":
            resolved_backend = "threading"
        elif prefer == "processes":  # honest: no process backend exists here
            _LOG.warning(
                "prefer='processes' requested but this Parallel only threads; "
                "continuing with the threading backend"
            )
        if require == "sharedmem":
            resolved_backend = "threading"
        elif require not in (None, "sharedmem"):
            raise DataValidationError(f"unknown require constraint: {require!r}")
        if resolved_backend not in self._VALID_BACKENDS:
            raise DataValidationError(
                f"backend must be one of {self._VALID_BACKENDS} "
                f"(no process forking in this port), got {backend!r}"
            )
        self.n_jobs = n_jobs
        self.backend = resolved_backend
        self.return_as = return_as
        self.verbose = int(verbose)
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch
        self.batch_size = batch_size

    # -- the call ---------------------------------------------------- #

    def __call__(self, iterable: Iterable[Any]) -> Any:
        """Run the ``delayed`` tasks; ordered results by submission."""
        if self.return_as == "list":
            return list(self._iterate(iterable))
        return self._iterate(iterable)

    def _iterate(self, iterable: Iterable[Any]) -> Iterator[Any]:
        workers = effective_n_jobs(self.n_jobs)
        tasks: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = [
            task for task in iterable
        ]
        started = time.time()
        if workers == 1 or self.backend == "sequential":
            yield from self._run_sequential(tasks, started)
            return
        yield from self._run_threaded(tasks, workers, started)

    def _emit(self, done: int, total: int, started: float) -> None:
        if self.verbose <= 0:
            return
        elapsed = time.time() - started
        if done >= total:
            print(
                f"[Parallel(n_jobs={effective_n_jobs(self.n_jobs)})]: "
                f"Done {done} out of {total} tasks | elapsed: {elapsed:.1f}s finished"
            )
        else:
            print(
                f"[Parallel(n_jobs={effective_n_jobs(self.n_jobs)})]: "
                f"Done {done} out of {total} tasks | elapsed: {elapsed:.1f}s "
                f"remaining: {(elapsed / max(done, 1)) * (total - done):.1f}s"
            )

    def _run_sequential(
        self,
        tasks: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
        started: float,
    ) -> Iterator[Any]:
        auto = _AutoBatcher()
        done = 0
        index = 0
        while index < len(tasks):
            size = (
                auto.compute_batch_size()
                if self.batch_size == "auto"
                else int(self.batch_size)
            )
            batch = tasks[index : index + size]
            index += len(batch)
            batch_started = time.time()
            for function, args, kwargs in batch:
                yield function(*args, **kwargs)
                done += 1
                self._emit(done, len(tasks), started)
            if self.batch_size == "auto":
                auto.batch_completed(len(batch), time.time() - batch_started)

    def _run_threaded(
        self,
        tasks: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
        workers: int,
        started: float,
    ) -> Iterator[Any]:
        pre_dispatch: int | None
        if self.pre_dispatch == "all":
            pre_dispatch = None
        else:
            pre_dispatch = max(eval_pre_dispatch(self.pre_dispatch, workers), 1)

        def run_batch(
            batch: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
        ) -> list[Any]:
            return [function(*args, **kwargs) for function, args, kwargs in batch]

        auto = _AutoBatcher()
        done = 0
        index = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: list[Future[list[Any]]] = []
            # the sliding pre_dispatch window: never more than
            # pre_dispatch submitted-but-unharvested tasks (joblib's
            # dispatch loop, "all" meaning unbounded)
            while index < len(tasks) or pending:
                while index < len(tasks) and (
                    pre_dispatch is None or len(pending) < pre_dispatch
                ):
                    size = (
                        auto.compute_batch_size()
                        if self.batch_size == "auto"
                        else int(self.batch_size)
                    )
                    batch = tasks[index : index + size]
                    index += len(batch)
                    pending.append(pool.submit(run_batch, batch))
                if self.return_as == "generator_unordered":
                    # completion order: harvest whichever finishes first
                    from concurrent.futures import wait

                    ready, _pending_set = wait(
                        pending, return_when="FIRST_COMPLETED"
                    )
                    for future in ready:
                        batch_started = time.time()
                        for value in self._result(future):
                            yield value
                            done += 1
                            self._emit(done, len(tasks), started)
                        if self.batch_size == "auto":
                            auto.batch_completed(1, time.time() - batch_started)
                        pending = [
                            other for other in pending if other not in ready
                        ]
                else:
                    future = pending.pop(0)  # submission order, always
                    batch_started = time.time()
                    for value in self._result(future):
                        yield value
                        done += 1
                        self._emit(done, len(tasks), started)
                    if self.batch_size == "auto":
                        auto.batch_completed(1, time.time() - batch_started)

    def _result(self, future: Future[list[Any]]) -> list[Any]:
        if self.timeout is None:
            return future.result()
        return future.result(timeout=self.timeout)


# --------------------------------------------------------------------- #
# dump / load (joblib/numpy_pickle.py + compressor.py)
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Codec:
    """One entry of the compressor registry (compressor.py's table)."""

    name: str
    extension: str
    magic: bytes
    default_level: int
    compress: Callable[[bytes, int], bytes]
    decompress: Callable[[bytes], bytes]


_CODECS: dict[str, _Codec] = {
    "zlib": _Codec(
        "zlib", ".z", b"\x78", 3,
        lambda data, level: zlib.compress(data, level), zlib.decompress,
    ),
    "gzip": _Codec(
        "gzip", ".gz", b"\x1f\x8b", 3,
        lambda data, level: gzip.compress(data, compresslevel=level, mtime=0),
        gzip.decompress,
    ),
    "bz2": _Codec(
        "bz2", ".bz2", b"BZ", 9, bz2.compress, bz2.decompress,
    ),
    "lzma": _Codec(
        "lzma", ".lzma", b"\x5d\x00", 6,
        lambda data, level: lzma.compress(data, format=lzma.FORMAT_ALONE, preset=level),
        lambda data: lzma.decompress(data, format=lzma.FORMAT_ALONE),
    ),
    "xz": _Codec(
        "xz", ".xz", b"\xfd7zXZ", 6,
        lambda data, level: lzma.compress(data, format=lzma.FORMAT_XZ, preset=level),
        lambda data: lzma.decompress(data, format=lzma.FORMAT_XZ),
    ),
}

#: The magic prefixes joblib's ``_detect_compressor`` sniffs on load,
#: longest-first so the xz header never matches a shorter prefix.
_MAGIC_ORDER: list[tuple[str, bytes]] = [
    (name, codec.magic) for name, codec in _CODECS.items()
]


def _parse_compress(
    compress: bool | int | str | tuple[str, int | None],
    filename: Path | str | None,
) -> tuple[str | None, int]:
    """Resolve the ``compress`` parameter exactly like ``numpy_pickle``.

    Returns ``(codec_name_or_None, level)`` with ``None`` meaning
    "write the plain pickle".  The rules, ported verbatim: ``True``
    -> zlib at its default level 3; an int 0-9 -> zlib at that level
    (0/``False``/``None`` -> plain); a string -> that codec at its
    own default (gzip 3, bz2 9, lzma/xz preset 6); a
    ``(codec, level)`` tuple -> explicit.  And the famous quirk: a
    filename ending in a known codec extension selects that codec
    **even when ``compress=0``** (the level resets to the codec
    default), unless ``compress`` is an explicit tuple.
    """
    compress_method = "zlib"
    compress_level: int | None
    if compress is True:
        compress_level = None
    elif isinstance(compress, tuple):
        if len(compress) != 2:
            raise DataValidationError(
                f"compress tuple must be (method, level), got {compress!r}"
            )
        compress_method, compress_level = compress  # type: ignore[misc,assignment]
    elif isinstance(compress, str):
        compress_method = compress
        compress_level = None
    else:
        compress_level = int(compress) if compress else 0
    if compress_method not in _CODECS:
        valid = ", ".join(sorted(_CODECS))
        raise DataValidationError(
            f"compress method must be one of {valid}, got {compress_method!r}"
        )
    if (
        compress_level is not None
        and compress_level is not False
        and not (isinstance(compress_level, int) and 0 <= compress_level <= 9)
    ):
        raise DataValidationError(
            f"compress level must be None or 0..9, got {compress_level!r}"
        )
    if compress_level is None or compress_level is False:
        compress_level = _CODECS[compress_method].default_level
    compress_level = int(compress_level)  # type: ignore[assignment]

    # the extension override (numpy_pickle.py's filename sniffing)
    if filename is not None and not isinstance(compress, tuple):
        text = str(filename)
        for name, codec in _CODECS.items():
            if text.endswith(codec.extension):
                compress_method = name
                if compress_level == 0:
                    compress_level = codec.default_level
                break
    if compress_level == 0:
        return None, 0
    return compress_method, compress_level


def dump(
    value: Any,
    filename: str | Path,
    compress: bool | int | str | tuple[str, int | None] = 0,
    protocol: int | None = None,
) -> list[Path]:
    """Persist an object, optionally compressed (joblib's ``dump``).

    Writes a plain pickle stream (protocol
    :data:`pickle.DEFAULT_PROTOCOL` when ``None``, as joblib does)
    either raw or through one of the registered codecs.  Returns
    ``[Path(filename)]`` -- joblib's historic list shape.

    The ``compress`` parameter follows joblib exactly (see
    :func:`_parse_compress`), including the extension quirk: a
    ``.gz``/``.z``/``.bz2``/``.lzma``/``.xz`` suffix picks its codec
    even with ``compress=0``.  Unlike joblib there is no
    numpy-array fast path (no ``.npy`` payload embedded after the
    pickle stream, no memmap on load) and no lz4 (joblib needs the
    external ``python-lz4`` for it too).

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from pharmaml.utils.compute import dump, load
    >>> tmp = Path(tempfile.mkdtemp())
    >>> payload = {"rows": list(range(400)), "text": "pharmaml " * 200}
    >>> plain = dump(payload, tmp / "blob.joblib")
    >>> [path.name for path in plain]
    ['blob.joblib']
    >>> restored = load(plain[0])
    >>> restored["rows"][:3]
    [0, 1, 2]
    >>> restored["text"][:9]
    'pharmaml '

    Compression: ``True`` is zlib level 3, an int sets the zlib
    level, a string picks a codec, a tuple is explicit:

    >>> zlib_path = dump(payload, tmp / "blob.z", compress=True)
    >>> len(zlib_path[0].read_bytes()) < len((tmp / "blob.joblib").read_bytes())
    True
    >>> load(zlib_path[0])["rows"][:3]
    [0, 1, 2]
    >>> gz = dump(payload, tmp / "blob.gz", compress=0)   # extension wins!
    >>> gz[0].read_bytes()[:2] == bytes((0x1F, 0x8B))   # gzip magic
    True
    >>> load(gz[0])["rows"][:3]
    [0, 1, 2]
    >>> xz = dump(payload, tmp / "blob.xz", compress=("xz", 6))
    >>> xz[0].read_bytes()[:6] == bytes((0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00))
    True
    >>> dump(payload, tmp / "blob.br", compress="brotli")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: compress method must be one of bz2, gzip, lzma, xz, zlib, got 'brotli'
    >>> import shutil
    >>> shutil.rmtree(tmp)
    """
    path = Path(filename)
    if protocol is None:
        protocol = pickle.DEFAULT_PROTOCOL
    blob = pickle.dumps(value, protocol=protocol)
    method, level = _parse_compress(compress, path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if method is None:
            path.write_bytes(blob)
        else:
            codec = _CODECS[method]
            path.write_bytes(codec.compress(blob, level))
    except OSError as exc:
        raise PersistenceError(f"cannot write {path}: {exc}") from exc
    return [path]


def load(filename: str | Path) -> Any:
    """Load what :func:`dump` wrote (joblib's ``load``).

    The codec is chosen by **magic bytes**, never by extension
    (joblib's ``_detect_compressor``): ``\\x78`` zlib, ``\\x1f\\x8b``
    gzip, ``BZ`` bzip2, ``\\xfd7zXZ`` xz, ``]\\x00`` lzma-alone;
    anything else is taken as a raw pickle stream.
    """
    path = Path(filename)
    if not path.is_file():
        raise PersistenceError(f"File not found: {path}")
    blob = path.read_bytes()
    payload = blob
    for name, magic in _MAGIC_ORDER:
        if blob[: len(magic)] == magic:
            payload = _CODECS[name].decompress(blob)
            break
    try:
        return pickle.loads(payload)
    except Exception as exc:
        raise PersistenceError(f"cannot load {path}: {exc}") from exc
