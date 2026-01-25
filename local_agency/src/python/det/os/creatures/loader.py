"""
Creature Loader
===============

Loads creatures into the DET-OS runtime from various sources:
- Pure Existence-Lang creatures (.ex files) - PREFERRED (Phase 20)
- JIT compilation from .ex source files
- Bytecode loading from .exb compiled files
- Legacy built-in creatures (Python wrappers) - DEPRECATED

Usage:
    loader = CreatureLoader(runtime)

    # Load creature (prefers .ex over Python wrapper)
    memory = loader.load("memory")  # -> loads memory.ex
    tool = loader.load("tool")      # -> loads tool.ex

    # Force Python wrapper (deprecated)
    memory = loader.load("memory", force_builtin=True)

    # Load from source (JIT)
    custom = loader.load_file("/path/to/custom.ex")

    # Load from bytecode
    fast = loader.load_bytecode("/path/to/creature.exb")

Note: As of Phase 20, pure Existence-Lang creatures are preferred.
Python wrappers are kept for backward compatibility but are deprecated.
"""

import os
import json
import struct
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Type, Union
from enum import Enum

from .base import CreatureWrapper
from .memory import MemoryCreature, MemoryType, spawn_memory_creature
from .tool import ToolCreature, spawn_tool_creature
from .reasoner import ReasonerCreature, spawn_reasoner_creature
from .planner import PlannerCreature, spawn_planner_creature
from ..existence.runtime import ExistenceKernelRuntime, CreatureState


class LoadMode(Enum):
    """How a creature was loaded."""
    BUILTIN = "builtin"      # Built-in Python wrapper
    JIT = "jit"              # JIT compiled from .ex source
    BYTECODE = "bytecode"    # Loaded from .exb bytecode


@dataclass
class CreatureSpec:
    """Specification for a loadable creature."""
    name: str
    creature_type: str
    initial_f: float = 10.0
    initial_a: float = 0.5
    description: str = ""
    protocols: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    bytecode_path: Optional[str] = None


@dataclass
class LoadedCreature:
    """A creature that has been loaded into the runtime."""
    spec: CreatureSpec
    wrapper: CreatureWrapper
    load_mode: LoadMode
    source_hash: Optional[str] = None  # Hash of source for cache invalidation


# Bytecode file format (.exb)
# Header: MAGIC (4) + VERSION (2) + FLAGS (2) + METADATA_LEN (4) + CODE_LEN (4)
# Metadata: JSON blob with creature spec
# Code: EIS bytecode instructions

EXB_MAGIC = b'DETC'  # DET Creature
EXB_VERSION = 1


class CreatureLoader:
    """
    Loads and manages creatures in the DET-OS runtime.

    Supports:
    - Built-in creatures with Python wrappers
    - JIT compilation from .ex source files
    - Bytecode loading from .exb files
    - Creature caching and registry
    """

    # DEPRECATED: Built-in creature types (Python wrappers)
    # As of Phase 20, pure Existence-Lang creatures (.ex files) are preferred.
    # These wrappers are kept for backward compatibility only.
    # Use loader.load("name") which will find the .ex file first.
    BUILTIN_CREATURES: Dict[str, Dict[str, Any]] = {
        "memory": {
            "spawn_fn": spawn_memory_creature,
            "wrapper_class": MemoryCreature,
            "default_f": 50.0,
            "default_a": 0.5,
            "description": "[DEPRECATED] Store and recall memories via bonds",
            "protocols": ["store", "recall", "get_instructions"],
            "el_file": "memory.ex",  # Preferred EL implementation
        },
        "tool": {
            "spawn_fn": spawn_tool_creature,
            "wrapper_class": ToolCreature,
            "default_f": 30.0,
            "default_a": 0.6,
            "description": "[DEPRECATED] Execute commands in sandboxed environment",
            "protocols": ["execute"],
            "el_file": "tool.ex",
        },
        "reasoner": {
            "spawn_fn": spawn_reasoner_creature,
            "wrapper_class": ReasonerCreature,
            "default_f": 40.0,
            "default_a": 0.7,
            "description": "[DEPRECATED] Chain-of-thought reasoning",
            "protocols": ["reason"],
            "el_file": "reasoner.ex",
        },
        "planner": {
            "spawn_fn": spawn_planner_creature,
            "wrapper_class": PlannerCreature,
            "default_f": 35.0,
            "default_a": 0.65,
            "description": "[DEPRECATED] Task decomposition and planning",
            "protocols": ["plan"],
            "el_file": "planner.ex",
        }
    }

    # Mapping from creature names to their .ex file names
    EL_CREATURES: Dict[str, str] = {
        "memory": "memory.ex",
        "tool": "tool.ex",
        "reasoner": "reasoner.ex",
        "planner": "planner.ex",
        "llm": "llm.ex",
        "terminal": "terminal.ex",
    }

    def __init__(self, runtime: ExistenceKernelRuntime,
                 cache_dir: Optional[str] = None):
        """
        Initialize the creature loader.

        Args:
            runtime: The DET-OS runtime to load creatures into
            cache_dir: Optional directory for bytecode cache
        """
        self.runtime = runtime
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Registry of loaded creatures
        self.loaded: Dict[str, LoadedCreature] = {}

        # Compiler for JIT loading (lazy import)
        self._compiler = None

        # Search paths for .ex files
        self.search_paths: List[Path] = [
            Path(__file__).parent.parent.parent.parent.parent / "existence",  # src/existence
        ]

    @property
    def compiler(self):
        """Lazy-load the Existence-Lang compiler."""
        if self._compiler is None:
            try:
                from det.lang import parse
                from det.lang.eis_compiler import EISCompiler
                self._compiler = {
                    'parse': parse,
                    'compile': EISCompiler()
                }
            except ImportError:
                self._compiler = None
        return self._compiler

    def list_available(self) -> List[Dict[str, Any]]:
        """List all available creatures (EL files preferred, then deprecated built-ins)."""
        available = []
        seen_names = set()

        # Phase 20: First scan for .ex files (preferred)
        for path in self.search_paths:
            if path.exists():
                for ex_file in path.glob("*.ex"):
                    if ex_file.name not in ["kernel.ex", "physics.ex"]:
                        name = ex_file.stem
                        if name not in seen_names:
                            seen_names.add(name)
                            # Check if this has a deprecated Python wrapper
                            has_wrapper = name in self.BUILTIN_CREATURES
                            available.append({
                                "name": name,
                                "type": "existence-lang",
                                "path": str(ex_file),
                                "description": f"Pure EL creature from {ex_file.name}",
                                "has_deprecated_wrapper": has_wrapper,
                                "loaded": name in self.loaded
                            })

        # Add built-in creatures that don't have .ex equivalents
        for name, info in self.BUILTIN_CREATURES.items():
            if name not in seen_names:
                available.append({
                    "name": name,
                    "type": "builtin-deprecated",
                    "description": info["description"],
                    "protocols": info["protocols"],
                    "loaded": name in self.loaded
                })

        return available

    def list_loaded(self) -> List[Dict[str, Any]]:
        """List currently loaded creatures."""
        return [
            {
                "name": name,
                "cid": lc.wrapper.cid,
                "type": lc.spec.creature_type,
                "mode": lc.load_mode.value,
                "F": lc.wrapper.F,
                "a": lc.wrapper.a,
                "protocols": lc.spec.protocols
            }
            for name, lc in self.loaded.items()
        ]

    def load(self, name: str,
             initial_f: Optional[float] = None,
             initial_a: Optional[float] = None,
             force_builtin: bool = False,
             **kwargs) -> CreatureWrapper:
        """
        Load a creature by name.

        As of Phase 20, prefers Existence-Lang (.ex) files over Python wrappers.
        Use force_builtin=True to load the deprecated Python wrapper instead.

        Args:
            name: Creature name (e.g., "memory", "tool") or path to .ex file
            initial_f: Override initial F value
            initial_a: Override initial a value
            force_builtin: Force loading Python wrapper (deprecated)
            **kwargs: Additional arguments for the spawn function

        Returns:
            CreatureWrapper for the loaded creature
        """
        # Check if already loaded
        if name in self.loaded:
            return self.loaded[name].wrapper

        # Check if it's a path
        if name.endswith('.ex') or name.endswith('.exb') or '/' in name:
            path = Path(name)
            if path.suffix == '.exb':
                return self.load_bytecode(str(path))
            else:
                return self.load_file(str(path))

        # Phase 20: Prefer .ex files over built-in Python wrappers
        if not force_builtin:
            # Check if there's an EL creature available
            el_filename = self.EL_CREATURES.get(name, f"{name}.ex")
            for search_path in self.search_paths:
                ex_path = search_path / el_filename
                if ex_path.exists():
                    import warnings
                    if name in self.BUILTIN_CREATURES:
                        warnings.warn(
                            f"Loading {name} from {ex_path.name} (pure EL). "
                            f"Python wrapper is deprecated. Use force_builtin=True to override.",
                            DeprecationWarning,
                            stacklevel=2
                        )
                    return self.load_file(str(ex_path), name=name)

        # Fall back to built-in creatures (deprecated)
        if name in self.BUILTIN_CREATURES:
            import warnings
            warnings.warn(
                f"Using deprecated Python wrapper for {name}. "
                f"Consider using the pure EL implementation.",
                DeprecationWarning,
                stacklevel=2
            )
            return self._load_builtin(name, initial_f, initial_a, **kwargs)

        raise ValueError(f"Unknown creature: {name}")

    def _load_builtin(self, name: str,
                      initial_f: Optional[float] = None,
                      initial_a: Optional[float] = None,
                      **kwargs) -> CreatureWrapper:
        """Load a built-in creature."""
        info = self.BUILTIN_CREATURES[name]

        f = initial_f if initial_f is not None else info["default_f"]
        a = initial_a if initial_a is not None else info["default_a"]

        # Call spawn function
        wrapper = info["spawn_fn"](
            self.runtime,
            name=name,
            initial_f=f,
            initial_a=a,
            **kwargs
        )

        # Create spec
        spec = CreatureSpec(
            name=name,
            creature_type=name,
            initial_f=f,
            initial_a=a,
            description=info["description"],
            protocols=info["protocols"]
        )

        # Register
        self.loaded[name] = LoadedCreature(
            spec=spec,
            wrapper=wrapper,
            load_mode=LoadMode.BUILTIN
        )

        return wrapper

    def load_file(self, path: str,
                  name: Optional[str] = None,
                  use_cache: bool = True) -> CreatureWrapper:
        """
        Load a creature from an .ex source file (JIT compilation).

        Args:
            path: Path to the .ex file
            name: Optional name override (defaults to filename stem)
            use_cache: Whether to use/update bytecode cache

        Returns:
            CreatureWrapper for the loaded creature
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Creature file not found: {path}")

        name = name or path.stem

        # Check if already loaded
        if name in self.loaded:
            return self.loaded[name].wrapper

        # Read source
        source = path.read_text()
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        # Check bytecode cache
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{name}_{source_hash}.exb"
            if cache_path.exists():
                return self.load_bytecode(str(cache_path), name=name)

        # JIT compile
        if not self.compiler:
            raise RuntimeError("Existence-Lang compiler not available for JIT loading")

        # Parse and compile
        ast = self.compiler['parse'](source)
        bytecode = self.compiler['compile'].compile(ast)

        # Extract creature info from AST
        creature_info = self._extract_creature_info(ast, name)

        # Cache bytecode if cache dir exists
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_bytecode(cache_path, creature_info, bytecode)

        # Create creature in runtime
        wrapper = self._instantiate_creature(creature_info, bytecode)

        # Create spec
        spec = CreatureSpec(
            name=name,
            creature_type=creature_info.get('type', name),
            initial_f=creature_info.get('initial_f', 10.0),
            initial_a=creature_info.get('initial_a', 0.5),
            description=creature_info.get('description', f'Loaded from {path.name}'),
            protocols=creature_info.get('protocols', []),
            source_path=str(path)
        )

        # Register
        self.loaded[name] = LoadedCreature(
            spec=spec,
            wrapper=wrapper,
            load_mode=LoadMode.JIT,
            source_hash=source_hash
        )

        return wrapper

    def load_bytecode(self, path: str,
                      name: Optional[str] = None) -> CreatureWrapper:
        """
        Load a creature from compiled bytecode (.exb file).

        Args:
            path: Path to the .exb file
            name: Optional name override

        Returns:
            CreatureWrapper for the loaded creature
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Bytecode file not found: {path}")

        # Read and parse bytecode file
        with open(path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != EXB_MAGIC:
                raise ValueError(f"Invalid bytecode file: bad magic")

            version = struct.unpack('<H', f.read(2))[0]
            if version > EXB_VERSION:
                raise ValueError(f"Unsupported bytecode version: {version}")

            flags = struct.unpack('<H', f.read(2))[0]
            metadata_len = struct.unpack('<I', f.read(4))[0]
            code_len = struct.unpack('<I', f.read(4))[0]

            # Read metadata
            metadata_bytes = f.read(metadata_len)
            metadata = json.loads(metadata_bytes.decode('utf-8'))

            # Read bytecode
            bytecode = f.read(code_len)

        name = name or metadata.get('name', path.stem)

        # Check if already loaded
        if name in self.loaded:
            return self.loaded[name].wrapper

        # Create creature in runtime
        wrapper = self._instantiate_creature(metadata, bytecode)

        # Create spec
        spec = CreatureSpec(
            name=name,
            creature_type=metadata.get('type', name),
            initial_f=metadata.get('initial_f', 10.0),
            initial_a=metadata.get('initial_a', 0.5),
            description=metadata.get('description', f'Loaded from {path.name}'),
            protocols=metadata.get('protocols', []),
            bytecode_path=str(path)
        )

        # Register
        self.loaded[name] = LoadedCreature(
            spec=spec,
            wrapper=wrapper,
            load_mode=LoadMode.BYTECODE,
            source_hash=metadata.get('source_hash')
        )

        return wrapper

    def compile_to_bytecode(self, source_path: str,
                            output_path: Optional[str] = None) -> str:
        """
        Compile an .ex file to bytecode (.exb).

        Args:
            source_path: Path to source .ex file
            output_path: Optional output path (defaults to same name with .exb)

        Returns:
            Path to the compiled .exb file
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if not self.compiler:
            raise RuntimeError("Existence-Lang compiler not available")

        # Read and compile
        source = source_path.read_text()
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        ast = self.compiler['parse'](source)
        compiled = self.compiler['compile'].compile(ast)

        # Extract creature info from AST
        creature_info = self._extract_creature_info(ast, source_path.stem)
        creature_info['source_hash'] = source_hash

        # Extract bytecode from CompiledProgram
        # The compiled result has creatures dict with CompiledCreature objects
        bytecode = self._serialize_compiled_program(compiled)

        # Determine output path
        if output_path is None:
            output_path = source_path.with_suffix('.exb')
        else:
            output_path = Path(output_path)

        # Write bytecode file
        self._save_bytecode(output_path, creature_info, bytecode)

        return str(output_path)

    def _serialize_compiled_program(self, compiled) -> bytes:
        """Serialize a CompiledProgram to bytes for storage."""
        # Collect all bytecode sections from all creatures
        sections = []

        if hasattr(compiled, 'creatures') and compiled.creatures:
            for name, creature in compiled.creatures.items():
                # Each CompiledCreature has: init_code, agency_code, grace_code, participate_code
                creature_data = {
                    'name': name,
                    'init': creature.init_code if hasattr(creature, 'init_code') else b'',
                    'agency': creature.agency_code if hasattr(creature, 'agency_code') else b'',
                    'grace': creature.grace_code if hasattr(creature, 'grace_code') else b'',
                    'participate': creature.participate_code if hasattr(creature, 'participate_code') else b'',
                }
                sections.append(creature_data)

        # Simple serialization: JSON header + concatenated bytecode
        # Format: [num_creatures:4][creature_entries...]
        # Each entry: [name_len:2][name][init_len:4][init][agency_len:4][agency]...
        result = bytearray()
        result.extend(struct.pack('<I', len(sections)))

        for section in sections:
            name_bytes = section['name'].encode('utf-8')
            result.extend(struct.pack('<H', len(name_bytes)))
            result.extend(name_bytes)

            for key in ['init', 'agency', 'grace', 'participate']:
                code = section[key] or b''
                result.extend(struct.pack('<I', len(code)))
                result.extend(code)

        return bytes(result)

    def _extract_creature_info(self, ast, default_name: str) -> Dict[str, Any]:
        """Extract creature metadata from parsed AST."""
        info = {
            'name': default_name,
            'type': default_name,
            'initial_f': 10.0,
            'initial_a': 0.5,
            'description': '',
            'protocols': []
        }

        # Try to extract from AST
        if hasattr(ast, 'creatures') and ast.creatures:
            creature = ast.creatures[0]  # Take first creature
            if hasattr(creature, 'name'):
                info['name'] = creature.name
                info['type'] = creature.name

            # Look for var declarations
            if hasattr(creature, 'vars'):
                for var in creature.vars:
                    if var.name == 'F' and hasattr(var, 'initial'):
                        info['initial_f'] = float(var.initial)
                    elif var.name == 'a' and hasattr(var, 'initial'):
                        info['initial_a'] = float(var.initial)

            # Look for kernel definitions (protocols)
            if hasattr(creature, 'kernels'):
                for kernel in creature.kernels:
                    if hasattr(kernel, 'name'):
                        info['protocols'].append(kernel.name.lower())

        return info

    def _save_bytecode(self, path: Path, metadata: Dict, bytecode: bytes):
        """Save bytecode to .exb file."""
        metadata_bytes = json.dumps(metadata).encode('utf-8')

        with open(path, 'wb') as f:
            # Write header
            f.write(EXB_MAGIC)
            f.write(struct.pack('<H', EXB_VERSION))
            f.write(struct.pack('<H', 0))  # flags
            f.write(struct.pack('<I', len(metadata_bytes)))
            f.write(struct.pack('<I', len(bytecode)))

            # Write metadata and code
            f.write(metadata_bytes)
            f.write(bytecode)

    def _instantiate_creature(self, info: Dict, bytecode: bytes) -> CreatureWrapper:
        """Create a creature instance in the runtime."""
        # Spawn creature in runtime
        cid = self.runtime.spawn(
            info.get('name', 'creature'),
            initial_f=info.get('initial_f', 10.0),
            initial_a=info.get('initial_a', 0.5),
            program=bytecode
        )
        self.runtime.creatures[cid].state = CreatureState.RUNNING

        # Create wrapper
        # For now, use base CreatureWrapper
        # In the future, could dynamically generate wrapper based on protocols
        wrapper = CreatureWrapper(self.runtime, cid)

        return wrapper

    def unload(self, name: str) -> bool:
        """
        Unload a creature from the runtime.

        Args:
            name: Name of the creature to unload

        Returns:
            True if unloaded, False if not found
        """
        if name not in self.loaded:
            return False

        lc = self.loaded[name]

        # Kill the creature in runtime
        if lc.wrapper.creature:
            self.runtime.kill(lc.wrapper.cid, "unloaded")

        del self.loaded[name]
        return True

    def reload(self, name: str) -> CreatureWrapper:
        """
        Reload a creature (unload and load again).

        Useful for picking up source changes for JIT-loaded creatures.

        Args:
            name: Name of the creature to reload

        Returns:
            New CreatureWrapper
        """
        if name not in self.loaded:
            raise ValueError(f"Creature not loaded: {name}")

        lc = self.loaded[name]

        # Get original load parameters
        spec = lc.spec
        source_path = spec.source_path
        bytecode_path = spec.bytecode_path

        # Unload
        self.unload(name)

        # Reload based on original load mode
        if source_path:
            return self.load_file(source_path, name=name, use_cache=False)
        elif bytecode_path:
            return self.load_bytecode(bytecode_path, name=name)
        else:
            # Built-in
            return self.load(name, initial_f=spec.initial_f, initial_a=spec.initial_a)

    def bond_to(self, creature_name: str, peer: CreatureWrapper,
                coherence: float = 1.0) -> int:
        """
        Create a bond between a loaded creature and another creature.

        Args:
            creature_name: Name of the loaded creature
            peer: Creature to bond with
            coherence: Bond coherence (default 1.0 for reliable IPC)

        Returns:
            Channel ID for the bond
        """
        if creature_name not in self.loaded:
            raise ValueError(f"Creature not loaded: {creature_name}")

        lc = self.loaded[creature_name]

        # Create bidirectional bond
        channel_id = lc.wrapper.bond_with(peer.cid, coherence)
        peer.bonds[lc.wrapper.cid] = channel_id

        return channel_id


# Convenience function for CLI
def create_loader(runtime: ExistenceKernelRuntime,
                  cache_dir: Optional[str] = None) -> CreatureLoader:
    """Create a creature loader for the given runtime."""
    return CreatureLoader(runtime, cache_dir)
