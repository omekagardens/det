"""
Bytecode Cache - Precompiled Existence-Lang Bytecode
=====================================================

Manages .exb (Existence Bytecode) files for fast creature loading.

File Format (.exb):
    [Header - 32 bytes]
    - Magic: "EXB\x00" (4 bytes)
    - Version: uint16 (2 bytes)
    - Flags: uint16 (2 bytes)
    - Source mtime: float64 (8 bytes)
    - Source size: uint32 (4 bytes)
    - Source hash: uint32 (4 bytes, CRC32)
    - Reserved: 8 bytes

    [Payload]
    - Pickled CompiledCreatureData dict

Usage:
    from det.lang.bytecode_cache import BytecodeCache

    cache = BytecodeCache()

    # Load creature (auto-compiles if needed)
    creature_data = cache.load("path/to/creature.ex")

    # Force recompile
    creature_data = cache.load("path/to/creature.ex", force_recompile=True)

    # Compile all .ex files in directory
    cache.compile_directory("path/to/creatures/")
"""

import struct
import pickle
import zlib
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .parser import Parser
from .eis_compiler import EISCompiler
from ..eis.creature_runner import compile_creature_for_runner, CompiledCreatureData


# File format constants
MAGIC = b'EXB\x00'
VERSION = 1
HEADER_SIZE = 32
HEADER_FORMAT = '<4sHHdII8s'  # magic, version, flags, mtime, size, hash, reserved


@dataclass
class BytecodeHeader:
    """Header information from .exb file."""
    magic: bytes
    version: int
    flags: int
    source_mtime: float
    source_size: int
    source_hash: int

    def is_valid(self) -> bool:
        """Check if header is valid."""
        return self.magic == MAGIC and self.version == VERSION


class BytecodeCache:
    """
    Manages bytecode caching for Existence-Lang creatures.

    Provides fast loading by caching compiled bytecode in .exb files.
    Automatically recompiles when source files change.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize bytecode cache.

        Args:
            cache_dir: Directory for .exb files. If None, uses same directory as .ex files.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'recompiles': 0,
        }

    def _get_exb_path(self, ex_path: Path) -> Path:
        """Get the .exb path for a given .ex file."""
        if self.cache_dir:
            return self.cache_dir / (ex_path.stem + '.exb')
        return ex_path.with_suffix('.exb')

    def _compute_source_hash(self, source: bytes) -> int:
        """Compute CRC32 hash of source."""
        return zlib.crc32(source) & 0xFFFFFFFF

    def _read_header(self, exb_path: Path) -> Optional[BytecodeHeader]:
        """Read header from .exb file."""
        try:
            with open(exb_path, 'rb') as f:
                header_bytes = f.read(HEADER_SIZE)
                if len(header_bytes) < HEADER_SIZE:
                    return None

                magic, version, flags, mtime, size, hash_, reserved = struct.unpack(
                    HEADER_FORMAT, header_bytes
                )

                return BytecodeHeader(
                    magic=magic,
                    version=version,
                    flags=flags,
                    source_mtime=mtime,
                    source_size=size,
                    source_hash=hash_,
                )
        except (IOError, struct.error):
            return None

    def _write_exb(self, exb_path: Path, source_mtime: float, source_size: int,
                   source_hash: int, creature_data: Dict[str, CompiledCreatureData]):
        """Write compiled bytecode to .exb file."""
        # Serialize creature data
        payload = pickle.dumps(creature_data, protocol=pickle.HIGHEST_PROTOCOL)

        # Build header
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            0,  # flags (reserved)
            source_mtime,
            source_size,
            source_hash,
            b'\x00' * 8,  # reserved
        )

        # Write file
        exb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exb_path, 'wb') as f:
            f.write(header)
            f.write(payload)

    def _read_exb(self, exb_path: Path) -> Optional[Dict[str, CompiledCreatureData]]:
        """Read compiled bytecode from .exb file."""
        try:
            with open(exb_path, 'rb') as f:
                # Skip header
                f.seek(HEADER_SIZE)
                payload = f.read()
                return pickle.loads(payload)
        except (IOError, pickle.UnpicklingError):
            return None

    def is_cache_valid(self, ex_path: Path) -> bool:
        """
        Check if cached .exb is valid for given .ex file.

        Returns True if:
        - .exb file exists
        - Header is valid (magic, version)
        - Source mtime matches
        - Source size matches
        - Source hash matches
        """
        ex_path = Path(ex_path)
        exb_path = self._get_exb_path(ex_path)

        if not exb_path.exists():
            return False

        header = self._read_header(exb_path)
        if not header or not header.is_valid():
            return False

        # Check source file
        try:
            stat = ex_path.stat()
            if stat.st_mtime != header.source_mtime:
                return False
            if stat.st_size != header.source_size:
                return False

            # Verify hash
            with open(ex_path, 'rb') as f:
                source_hash = self._compute_source_hash(f.read())
            if source_hash != header.source_hash:
                return False

            return True
        except IOError:
            return False

    def compile(self, ex_path: Path, force: bool = False) -> Dict[str, CompiledCreatureData]:
        """
        Compile .ex file to bytecode.

        Args:
            ex_path: Path to .ex source file
            force: If True, recompile even if cache is valid

        Returns:
            Dict mapping creature names to CompiledCreatureData
        """
        ex_path = Path(ex_path)
        exb_path = self._get_exb_path(ex_path)

        # Read source
        with open(ex_path, 'rb') as f:
            source_bytes = f.read()
        source = source_bytes.decode('utf-8')

        stat = ex_path.stat()
        source_mtime = stat.st_mtime
        source_size = stat.st_size
        source_hash = self._compute_source_hash(source_bytes)

        # Parse and compile
        parser = Parser(source, ex_path.name)
        ast = parser.parse()
        compiler = EISCompiler()
        compiled = compiler.compile(ast)
        creature_data = compile_creature_for_runner(compiled)

        # Write .exb
        self._write_exb(exb_path, source_mtime, source_size, source_hash, creature_data)

        self._stats['recompiles'] += 1
        return creature_data

    def load(self, ex_path: Path, force_recompile: bool = False) -> Dict[str, CompiledCreatureData]:
        """
        Load creature from .ex file, using cached .exb if valid.

        Args:
            ex_path: Path to .ex source file
            force_recompile: If True, always recompile from source

        Returns:
            Dict mapping creature names to CompiledCreatureData
        """
        ex_path = Path(ex_path)
        exb_path = self._get_exb_path(ex_path)

        # Check cache
        if not force_recompile and self.is_cache_valid(ex_path):
            creature_data = self._read_exb(exb_path)
            if creature_data:
                self._stats['hits'] += 1
                return creature_data

        # Cache miss - compile
        self._stats['misses'] += 1
        return self.compile(ex_path)

    def load_exb(self, exb_path: Path) -> Optional[Dict[str, CompiledCreatureData]]:
        """
        Load directly from .exb file without validation.

        Use this when you know the .exb is valid (e.g., shipped with application).

        Args:
            exb_path: Path to .exb bytecode file

        Returns:
            Dict mapping creature names to CompiledCreatureData, or None if invalid
        """
        exb_path = Path(exb_path)

        header = self._read_header(exb_path)
        if not header or not header.is_valid():
            return None

        return self._read_exb(exb_path)

    def compile_directory(self, directory: Path, recursive: bool = False) -> Dict[str, bool]:
        """
        Compile all .ex files in a directory.

        Args:
            directory: Directory containing .ex files
            recursive: If True, search subdirectories

        Returns:
            Dict mapping file paths to success status
        """
        directory = Path(directory)
        pattern = '**/*.ex' if recursive else '*.ex'

        results = {}
        for ex_path in directory.glob(pattern):
            try:
                self.compile(ex_path, force=True)
                results[str(ex_path)] = True
            except Exception as e:
                results[str(ex_path)] = False
                print(f"Failed to compile {ex_path}: {e}")

        return results

    def clear_cache(self, directory: Optional[Path] = None):
        """
        Remove all .exb files from directory.

        Args:
            directory: Directory to clear. If None, uses cache_dir.
        """
        target = Path(directory) if directory else self.cache_dir
        if not target:
            return

        for exb_path in target.glob('**/*.exb'):
            try:
                exb_path.unlink()
            except IOError:
                pass

    @property
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset cache statistics."""
        self._stats = {'hits': 0, 'misses': 0, 'recompiles': 0}


# Convenience functions

_default_cache: Optional[BytecodeCache] = None

def get_cache() -> BytecodeCache:
    """Get the default bytecode cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = BytecodeCache()
    return _default_cache

def load_creature(ex_path: Path, force_recompile: bool = False) -> Dict[str, CompiledCreatureData]:
    """Load creature using default cache."""
    return get_cache().load(ex_path, force_recompile)

def compile_creature(ex_path: Path) -> Dict[str, CompiledCreatureData]:
    """Compile creature and cache bytecode."""
    return get_cache().compile(ex_path, force=True)
