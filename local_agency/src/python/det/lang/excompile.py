#!/usr/bin/env python3
"""
excompile - Existence-Lang Bytecode Compiler
============================================

Precompile .ex files to .exb bytecode for fast loading.

Usage:
    python -m det.lang.excompile <file.ex>           # Compile single file
    python -m det.lang.excompile <directory>         # Compile all .ex in directory
    python -m det.lang.excompile --all               # Compile all standard creatures
    python -m det.lang.excompile --info <file.exb>   # Show bytecode info
    python -m det.lang.excompile --clean <directory> # Remove all .exb files
"""

import argparse
import sys
import time
import struct
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from det.lang.bytecode_cache import (
    BytecodeCache, MAGIC, VERSION, HEADER_FORMAT, HEADER_SIZE
)


def compile_file(ex_path: Path, verbose: bool = True) -> bool:
    """Compile a single .ex file."""
    cache = BytecodeCache()

    try:
        start = time.perf_counter()
        cache.compile(ex_path, force=True)
        elapsed = (time.perf_counter() - start) * 1000

        exb_path = ex_path.with_suffix('.exb')
        if verbose:
            src_size = ex_path.stat().st_size
            exb_size = exb_path.stat().st_size
            ratio = exb_size / src_size * 100
            print(f"  {ex_path.name} -> {exb_path.name}")
            print(f"    Source: {src_size:,} bytes")
            print(f"    Bytecode: {exb_size:,} bytes ({ratio:.1f}%)")
            print(f"    Time: {elapsed:.2f}ms")
        return True
    except Exception as e:
        print(f"  ERROR: {ex_path.name}: {e}", file=sys.stderr)
        return False


def compile_directory(directory: Path, verbose: bool = True) -> tuple:
    """Compile all .ex files in directory."""
    ex_files = list(directory.glob('*.ex'))

    if not ex_files:
        print(f"No .ex files found in {directory}")
        return 0, 0

    print(f"Compiling {len(ex_files)} files in {directory}...")
    print()

    success = 0
    failed = 0

    for ex_path in sorted(ex_files):
        if compile_file(ex_path, verbose):
            success += 1
        else:
            failed += 1
        if verbose:
            print()

    return success, failed


def show_info(exb_path: Path):
    """Show information about .exb file."""
    if not exb_path.exists():
        print(f"File not found: {exb_path}", file=sys.stderr)
        return

    try:
        with open(exb_path, 'rb') as f:
            header_bytes = f.read(HEADER_SIZE)
            payload = f.read()

        magic, version, flags, mtime, size, hash_, reserved = struct.unpack(
            HEADER_FORMAT, header_bytes
        )

        print(f"Bytecode Info: {exb_path.name}")
        print(f"  Magic: {magic}")
        print(f"  Version: {version}")
        print(f"  Flags: {flags:#x}")
        print(f"  Source mtime: {mtime}")
        print(f"  Source size: {size:,} bytes")
        print(f"  Source hash: {hash_:#010x}")
        print(f"  Payload size: {len(payload):,} bytes")
        print(f"  Total size: {exb_path.stat().st_size:,} bytes")

        # Check if source exists
        ex_path = exb_path.with_suffix('.ex')
        if ex_path.exists():
            cache = BytecodeCache()
            is_valid = cache.is_cache_valid(ex_path)
            print(f"  Cache valid: {is_valid}")
        else:
            print(f"  Source file: NOT FOUND")

    except Exception as e:
        print(f"Error reading {exb_path}: {e}", file=sys.stderr)


def clean_directory(directory: Path):
    """Remove all .exb files from directory."""
    exb_files = list(directory.glob('*.exb'))

    if not exb_files:
        print(f"No .exb files found in {directory}")
        return

    print(f"Removing {len(exb_files)} .exb files...")
    for exb_path in exb_files:
        try:
            exb_path.unlink()
            print(f"  Removed: {exb_path.name}")
        except Exception as e:
            print(f"  ERROR: {exb_path.name}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Existence-Lang Bytecode Compiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('path', nargs='?', help='File or directory to compile')
    parser.add_argument('--all', action='store_true',
                        help='Compile all standard creatures in ../existence/')
    parser.add_argument('--info', metavar='FILE', help='Show bytecode info')
    parser.add_argument('--clean', metavar='DIR', help='Remove all .exb files')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode (less output)')

    args = parser.parse_args()

    if args.info:
        show_info(Path(args.info))
        return

    if args.clean:
        clean_directory(Path(args.clean))
        return

    if args.all:
        # Find existence directory relative to this script
        script_dir = Path(__file__).parent.parent.parent
        existence_dir = script_dir.parent / 'existence'
        if not existence_dir.exists():
            print(f"Existence directory not found: {existence_dir}", file=sys.stderr)
            sys.exit(1)

        success, failed = compile_directory(existence_dir, not args.quiet)
        print(f"Compiled: {success} succeeded, {failed} failed")
        sys.exit(0 if failed == 0 else 1)

    if args.path:
        path = Path(args.path)
        if path.is_file():
            if compile_file(path, not args.quiet):
                sys.exit(0)
            else:
                sys.exit(1)
        elif path.is_dir():
            success, failed = compile_directory(path, not args.quiet)
            print(f"Compiled: {success} succeeded, {failed} failed")
            sys.exit(0 if failed == 0 else 1)
        else:
            print(f"Path not found: {path}", file=sys.stderr)
            sys.exit(1)

    parser.print_help()


if __name__ == '__main__':
    main()
