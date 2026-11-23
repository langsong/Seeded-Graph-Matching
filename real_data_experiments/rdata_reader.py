"""Minimal reader for R save() files (.Rdata / .Rd), so no R install is needed.

Covers what the SGM data files use: numeric, integer, logical and character vectors
and matrices, lists, factors, and the attributes that shape them (dim, names,
levels). Anything else (functions, environments, S4 objects) raises an error
instead of being misread. Uncompressed, gzip, bzip2 and xz files are accepted.

    objects = read_rdata("Data/herm_Graph.Rd")   # {"Ac": ndarray, "neurons": [...], ...}
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import struct
from pathlib import Path

import numpy as np

# Serialisation codes from R's serialize.c.
REFSXP, NILVALUE_SXP = 255, 254
SYMSXP, LISTSXP, CHARSXP = 1, 2, 9
LGLSXP, INTSXP, REALSXP, CPLXSXP, STRSXP, VECSXP, EXPRSXP, RAWSXP = 10, 13, 14, 15, 16, 19, 20, 24
NA_INTEGER = -(2 ** 31)


class _Stream:
    def __init__(self, raw: bytes):
        self.raw, self.pos = raw, 0

    def take(self, n: int) -> bytes:
        out = self.raw[self.pos:self.pos + n]
        if len(out) != n:
            raise ValueError("Unexpected end of R data file.")
        self.pos += n
        return out

    def int(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def array(self, dtype: str, n: int) -> np.ndarray:
        return np.frombuffer(self.take(np.dtype(dtype).itemsize * n), dtype=dtype)


def _flags(word: int) -> tuple[int, bool, bool]:
    return word & 0xFF, bool(word & (1 << 9)), bool(word & (1 << 10))


class _Reader:
    def __init__(self, stream: _Stream):
        self.s, self.refs = stream, []

    def length(self) -> int:
        n = self.s.int()
        if n == -1:  # long vector: two 32-bit halves
            n = (self.s.int() << 32) + self.s.int()
        return n

    def item(self):
        sexptype, has_attr, has_tag = _flags(word := self.s.int())
        if sexptype == NILVALUE_SXP:
            return None
        if sexptype == REFSXP:
            return self.refs[(word >> 8 or self.s.int()) - 1]
        if sexptype == SYMSXP:
            symbol = self.item()
            self.refs.append(symbol)
            return symbol
        if sexptype == CHARSXP:
            n = self.s.int()
            return None if n == -1 else self.s.take(n).decode("utf-8", errors="replace")
        if sexptype == LISTSXP:
            return self._pairlist(has_attr, has_tag)

        if sexptype in (LGLSXP, INTSXP):
            raw = self.s.array(">i4", self.length()).astype(np.int64)
            values = raw.astype(float)
            values[raw == NA_INTEGER] = np.nan
            if sexptype == INTSXP and not (raw == NA_INTEGER).any():
                values = raw
        elif sexptype == REALSXP:
            values = self.s.array(">f8", self.length()).astype(np.float64)
        elif sexptype == CPLXSXP:
            parts = self.s.array(">f8", 2 * self.length()).astype(np.float64)
            values = parts[0::2] + 1j * parts[1::2]
        elif sexptype == RAWSXP:
            values = np.frombuffer(self.s.take(self.length()), dtype=np.uint8).copy()
        elif sexptype == STRSXP:
            values = [self.item() for _ in range(self.length())]
        elif sexptype in (VECSXP, EXPRSXP):
            values = [self.item() for _ in range(self.length())]
        else:
            raise ValueError(f"R object type {sexptype} is not supported by this reader.")
        attributes = self.item() if has_attr else None
        return _shape(values, attributes or {})

    def _pairlist(self, has_attr: bool, has_tag: bool) -> dict:
        out = {}
        while True:
            if has_attr:
                self.item()
            tag = self.item() if has_tag else None
            out[tag if tag is not None else f"[[{len(out) + 1}]]"] = self.item()
            sexptype, has_attr, has_tag = _flags(self.s.int())
            if sexptype == NILVALUE_SXP:
                return out
            if sexptype != LISTSXP:
                raise ValueError(f"Unsupported pairlist tail (type {sexptype}).")


def _shape(values, attributes: dict):
    levels = attributes.get("levels")
    if levels is not None and isinstance(values, np.ndarray):  # factor -> labels
        return [None if np.isnan(code) else levels[int(code) - 1] for code in values.astype(float)]
    dim = attributes.get("dim")
    if dim is not None and isinstance(values, np.ndarray):
        values = values.reshape([int(d) for d in np.atleast_1d(dim)], order="F")  # R is column-major
    names = attributes.get("names")
    if isinstance(values, list) and names is not None and all(isinstance(n, str) and n for n in names) \
            and len(set(names)) == len(names) and any(not isinstance(v, str) for v in values):
        return dict(zip(names, values))  # a named list
    return values


def read_rdata(path: str | Path) -> dict:
    """Return the objects saved in an R save() file, keyed by object name."""

    raw = Path(path).read_bytes()
    for opener in (gzip.decompress, bz2.decompress, lzma.decompress):
        try:
            raw = opener(raw)
            break
        except Exception:
            pass
    if raw[:5] not in (b"RDX2\n", b"RDX3\n"):
        raise ValueError(f"{path} is not an R save() file (header {raw[:5]!r}).")
    stream = _Stream(raw[5:])
    if stream.take(2) != b"X\n":
        raise ValueError("Only R's default binary (XDR) format is supported.")
    version = stream.int()
    stream.int(), stream.int()  # versions of R that wrote / can read the file
    if version == 3:
        stream.take(stream.int())  # native encoding name
    return _Reader(stream).item()
