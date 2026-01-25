#!/usr/bin/env python3
"""
DET-OS CLI - LLM as Existence-Lang Creature
============================================

This CLI runs the LLM agent as a creature within the DET-OS kernel.
The agent has resource (F), agency (a), and participates in DET physics.

Architecture:
    User Input
        ↓
    LLMAgent Creature (F, a, bonds)
        ↓ bond
    MemoryCreature (stores/recalls)
        ↓
    DET-OS Kernel (kernel.ex)
        ↓
    Substrate v2 (C/Metal)
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent))

from det.os.existence.bootstrap import DETOSBootstrap, BootConfig, BootState
from det.os.existence.runtime import CreatureState
from det.os.creatures.base import CreatureWrapper
from det.os.creatures.memory import MemoryCreature, MemoryType, spawn_memory_creature
from det.os.creatures.loader import CreatureLoader, LoadMode


class LLMCreature(CreatureWrapper):
    """
    LLM Agent as a DET-OS creature.

    The LLM creature:
    - Has resource (F) that depletes when generating tokens
    - Has agency (a) that determines capabilities
    - Creates bonds to memory creatures
    - Is subject to presence-based scheduling
    """

    def __init__(self, runtime, cid: int, ollama_url: str, model: str):
        super().__init__(runtime, cid)
        self.ollama_url = ollama_url
        self.model = model
        self.client = None
        self.conversation_history = []

        # Token costs (F consumed per token)
        self.cost_per_input_token = 0.01
        self.cost_per_output_token = 0.05

        # Memory creature reference (set after bonding)
        self.memory_cid: Optional[int] = None

        # Initialize Ollama client
        self._init_ollama()

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            from det.llm import OllamaClient
            self.client = OllamaClient(base_url=self.ollama_url, model=self.model)
        except ImportError:
            # Fallback to requests-based client
            self.client = SimpleOllamaClient(self.ollama_url, self.model)

    def bond_to_memory(self, memory: MemoryCreature, coherence: float = 0.9) -> int:
        """Create a high-coherence bond to memory creature."""
        channel_id = self.bond_with(memory.cid, coherence)
        self.memory_cid = memory.cid

        # Memory also needs to know about this bond
        memory.bonds[self.cid] = channel_id

        return channel_id

    def store_memory(self, content: str, memory_type: str = "context",
                     importance: int = 5, metadata: Optional[Dict] = None) -> bool:
        """Store a memory via bond to memory creature."""
        if self.memory_cid is None:
            return False

        msg = {
            "type": "store",
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata or {}
        }
        return self.send_to(self.memory_cid, msg)

    def recall_memories(self, query: str, limit: int = 5,
                        memory_types: Optional[List[str]] = None) -> bool:
        """Request memory recall via bond. Response comes async."""
        if self.memory_cid is None:
            return False

        msg = {
            "type": "recall",
            "query": query,
            "limit": limit
        }
        if memory_types:
            msg["memory_types"] = memory_types
        return self.send_to(self.memory_cid, msg)

    def get_instructions(self) -> bool:
        """Request standing instructions from memory creature."""
        if self.memory_cid is None:
            return False
        return self.send_to(self.memory_cid, {"type": "get_instructions"})

    def get_memory_responses(self) -> List[Dict]:
        """Get any pending responses from memory creature."""
        if self.memory_cid is None:
            return []

        return self.receive_all_from(self.memory_cid)

    def can_think(self, estimated_tokens: int = 100) -> bool:
        """Check if we have enough F to think."""
        estimated_cost = estimated_tokens * self.cost_per_output_token
        return self.F >= estimated_cost

    def think(self, user_input: str, context_memories: Optional[List[str]] = None) -> tuple:
        """
        Process user input and generate response.

        Args:
            user_input: The user's message
            context_memories: Optional list of recalled memories to include in context

        Returns (response, stats) where stats includes token costs.
        """
        if not self.is_alive:
            return "[Creature has died - no resources]", {"error": "dead"}

        if not self.client or not self.client.is_available():
            return "[Ollama not available]", {"error": "no_ollama"}

        # Estimate input cost
        input_tokens = len(user_input.split()) * 1.3  # rough estimate
        input_cost = input_tokens * self.cost_per_input_token

        # Check if we can afford to think
        if self.F < input_cost:
            return f"[Insufficient resource: F={self.F:.2f}, need {input_cost:.2f}]", {
                "error": "insufficient_F",
                "F": self.F,
                "needed": input_cost
            }

        # Deduct input cost
        self.creature.F -= input_cost

        # Build messages with optional memory context
        messages = list(self.conversation_history)

        if context_memories:
            memory_context = "\n".join(f"- {m}" for m in context_memories)
            system_msg = f"Relevant context from memory:\n{memory_context}"
            messages.insert(0, {"role": "system", "content": system_msg})

        messages.append({"role": "user", "content": user_input})

        # Generate response
        start_time = time.time()
        try:
            result = self.client.chat(
                messages=messages,
                temperature=self._compute_temperature()
            )
            # Handle both dict (OllamaClient) and string (SimpleOllamaClient) responses
            if isinstance(result, dict):
                response = result.get("message", {}).get("content", str(result))
            else:
                response = result
        except Exception as e:
            return f"[Error: {e}]", {"error": str(e)}

        elapsed = time.time() - start_time

        # Estimate output cost
        output_tokens = len(response.split()) * 1.3
        output_cost = output_tokens * self.cost_per_output_token

        # Deduct output cost
        self.creature.F -= output_cost

        # Update conversation history (without memory context)
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Compile stats
        stats = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "F_remaining": self.F,
            "elapsed": elapsed,
            "used_memories": len(context_memories) if context_memories else 0,
        }

        return response, stats

    def _compute_temperature(self) -> float:
        """
        Compute LLM temperature based on agency.
        Higher agency = more exploratory (higher temp)
        Lower agency = more conservative (lower temp)
        """
        base_temp = 0.7
        # Agency modulates temperature: a=1.0 -> temp=1.0, a=0.5 -> temp=0.5
        return base_temp * self.a

    def extract_memories(self, user_input: str, response: str) -> List[Dict]:
        """
        Use LLM to extract key facts/preferences/instructions from the exchange.
        Returns list of {content, memory_type, importance} dicts.
        """
        if not self.client or not self.client.is_available():
            return []

        # Cost check - extraction is a small call
        if self.F < 1.0:
            return []

        extraction_prompt = f"""Analyze this exchange and extract any key information worth remembering.

USER: {user_input}
ASSISTANT: {response}

For each piece of information, classify it as:
- fact: Factual information (names, dates, technical details)
- preference: User preferences or likes/dislikes
- instruction: Something the user wants you to remember doing
- context: Relevant background context

Output ONLY a JSON array. Each item should have:
- "content": The information to remember (concise, self-contained)
- "type": One of fact/preference/instruction/context
- "importance": 1-10 (10 = critical to remember)

If nothing is worth remembering, output: []

Examples of good extractions:
- {{"content": "User's name is Sam", "type": "fact", "importance": 9}}
- {{"content": "User prefers concise responses", "type": "preference", "importance": 7}}
- {{"content": "Always check code for security issues", "type": "instruction", "importance": 8}}

Output JSON array only, no explanation:"""

        try:
            result = self.client.chat(
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1  # Low temp for consistent extraction
            )
            if isinstance(result, dict):
                text = result.get("message", {}).get("content", "[]")
            else:
                text = result

            # Deduct small cost for extraction
            self.creature.F -= 0.5

            # Parse JSON
            import json
            # Find JSON array in response
            text = text.strip()
            if text.startswith("```"):
                # Remove markdown code blocks
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("```"))
                text = text.strip()

            memories = json.loads(text)
            if not isinstance(memories, list):
                return []

            # Validate and normalize
            valid = []
            for m in memories:
                if isinstance(m, dict) and "content" in m:
                    valid.append({
                        "content": str(m["content"]),
                        "memory_type": m.get("type", "context"),
                        "importance": min(10, max(1, int(m.get("importance", 5))))
                    })
            return valid

        except Exception:
            return []


class SimpleOllamaClient:
    """Simple Ollama client using requests."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except:
            return False

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        import requests
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=120
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


def format_creature_state(creature: CreatureWrapper) -> str:
    """Format creature state for display."""
    state = creature.get_state_dict()
    if state.get("status") == "dead":
        return "\033[91m[DEAD]\033[0m"

    F = state["F"]
    a = state["a"]
    P = state["presence"]

    # Color code F based on level
    if F > 50:
        f_color = "\033[92m"  # green
    elif F > 10:
        f_color = "\033[93m"  # yellow
    else:
        f_color = "\033[91m"  # red

    return f"F:{f_color}{F:.1f}\033[0m a:{a:.2f} P:{P:.1f}"


def format_kernel_state(bootstrap) -> str:
    """Format kernel state for display."""
    stats = bootstrap.get_stats()
    return f"tick:{stats['tick']} creatures:{stats['num_creatures']} F_total:{stats['total_F']:.0f}"


def run_cli(
    model: str = "llama3.2:3b",
    ollama_url: str = "http://localhost:11434",
    initial_f: float = 100.0,
    initial_a: float = 0.8,
    tick_rate: float = 10.0,
    debug: bool = False
):
    """
    Run the DET-OS CLI with LLM as a creature.
    """
    print("\n" + "=" * 60)
    print("  DET-OS CLI - LLM as Existence-Lang Creature")
    print("  Deep Existence Theory Operating System")
    print("=" * 60)

    # Boot DET-OS kernel
    print("\nBooting DET-OS kernel...")
    config = BootConfig(
        total_F=100000.0,
        grace_pool=1000.0,
        tick_rate=tick_rate,
        debug=debug
    )

    bootstrap = DETOSBootstrap(config=config)
    if not bootstrap.boot():
        print("\033[91mKernel boot failed!\033[0m")
        for entry in bootstrap.boot_log:
            print(f"  {entry}")
        return 1

    print(f"  Kernel running: {format_kernel_state(bootstrap)}")

    # Spawn LLM creature
    print(f"\nSpawning LLM creature (F={initial_f}, a={initial_a})...")
    llm_cid = bootstrap.spawn("llm_agent", initial_f=initial_f, initial_a=initial_a)
    bootstrap.runtime.creatures[llm_cid].state = CreatureState.RUNNING
    llm = LLMCreature(bootstrap.runtime, llm_cid, ollama_url, model)

    # Spawn Memory creature and bond with perfect coherence (no message loss)
    print("Spawning Memory creature (F=50, a=0.5)...")
    memory = spawn_memory_creature(bootstrap.runtime, "memory", initial_f=50.0, initial_a=0.5)
    channel_id = llm.bond_to_memory(memory, coherence=1.0)
    print(f"  Bond created: LLM <--[ch:{channel_id}, C:1.0]--> Memory")

    # Initialize creature loader for loading additional creatures
    loader = CreatureLoader(bootstrap.runtime)
    # Register the already-spawned memory creature in the loader's registry
    from det.os.creatures.loader import LoadedCreature, CreatureSpec
    loader.loaded["memory"] = LoadedCreature(
        spec=CreatureSpec(
            name="memory",
            creature_type="memory",
            initial_f=50.0,
            initial_a=0.5,
            description="Store and recall memories via bonds",
            protocols=["store", "recall", "get_instructions"]
        ),
        wrapper=memory,
        load_mode=LoadMode.BUILTIN
    )

    # Check Ollama
    print(f"Connecting to Ollama ({ollama_url})...", end=" ")
    if llm.client and llm.client.is_available():
        print(f"OK (model: {model})")
    else:
        print("\033[91mFAILED\033[0m")
        print("  Ollama not running. Start with: ollama serve")
        print("  Continuing without LLM (state exploration only)")

    print(f"\nLLM Creature: {format_creature_state(llm)}")
    print(f"Memory Creature: {format_creature_state(memory)}")
    print(f"Kernel: {format_kernel_state(bootstrap)}")

    print("\nCommands:")
    print("  /state      - Show LLM creature state")
    print("  /memory     - Show memory creature state")
    print("  /memories   - List stored memories")
    print("  /kernel     - Show kernel state")
    print("  /store <t>  - Store text (use: /store [type:importance] text)")
    print("  /recall <q> - Recall memories matching query")
    print("  /instruct   - Store as instruction (high priority)")
    print("  /tick [n]   - Advance kernel by n ticks")
    print("  /inject f   - Inject F resource into LLM creature")
    print("  /list       - List all creatures")
    print("  /bonds      - Show bonds between creatures")
    print("  /creatures  - List available/loaded creatures")
    print("  /load <n>   - Load a creature (name or path to .ex/.exb)")
    print("  /unload <n> - Unload a creature")
    print("  /bond <n>   - Bond loaded creature with LLM")
    print("  /compile <p>- Compile .ex to .exb bytecode")
    print("  /help       - Show this help")
    print("  /quit       - Exit")
    print("\nMemory types: fact, preference, instruction, context, episode")
    print("-" * 60)

    # Background tick thread
    import threading
    tick_running = True

    def tick_loop():
        while tick_running and bootstrap.state == BootState.RUNNING:
            try:
                bootstrap.tick()
                # Note: Don't process memory messages here - let main thread handle it
                # to avoid race conditions with store/recall commands
                time.sleep(1.0 / tick_rate)
            except:
                break

    tick_thread = threading.Thread(target=tick_loop, daemon=True)
    tick_thread.start()

    # Main REPL loop
    while True:
        try:
            # Show creature state in prompt
            state_str = format_creature_state(llm)
            user_input = input(f"\n[{state_str}] \033[96mYou>\033[0m ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if cmd in ("quit", "exit", "q"):
                    print("\nHalting kernel...")
                    tick_running = False
                    bootstrap.halt()
                    print("Goodbye!")
                    break

                elif cmd == "state":
                    state = llm.get_state_dict()
                    print("\nLLM Creature State:")
                    for k, v in state.items():
                        print(f"  {k}: {v}")

                elif cmd == "memory":
                    stats = memory.get_stats()
                    print("\nMemory Creature State:")
                    for k, v in stats.items():
                        print(f"  {k}: {v}")

                elif cmd == "kernel":
                    stats = bootstrap.get_stats()
                    print("\nKernel State:")
                    for k, v in stats.items():
                        if k != "boot_log":
                            print(f"  {k}: {v}")

                elif cmd == "store":
                    if not args:
                        print("Usage: /store [type:importance] <text>")
                        print("  Types: fact, preference, instruction, context")
                        print("  Importance: 1-10 (default 5)")
                        print("  Example: /store fact:9 User's name is Sam")
                        continue

                    # Parse optional type:importance prefix
                    mem_type = "context"
                    importance = 5
                    text = args

                    if " " in args:
                        first, rest = args.split(" ", 1)
                        if ":" in first:
                            parts = first.split(":")
                            mem_type = parts[0].lower()
                            if mem_type not in ("fact", "preference", "instruction", "context", "episode"):
                                mem_type = "context"
                            if len(parts) > 1 and parts[1].isdigit():
                                importance = int(parts[1])
                            text = rest

                    if debug:
                        print(f"  DEBUG: type={mem_type}, importance={importance}")

                    success = llm.store_memory(text, memory_type=mem_type, importance=importance)

                    if success:
                        print(f"Storing [{mem_type}:{importance}]: \"{text[:40]}{'...' if len(text) > 40 else ''}\"")
                        time.sleep(0.1)
                        memory.process_messages()
                        responses = llm.get_memory_responses()
                        for r in responses:
                            if r.get("type") == "store_ack" and r.get("success"):
                                print(f"  Stored. Memories: {len(memory.memories)}")
                                break
                    else:
                        print(f"Failed to store (F={llm.F:.1f})")

                elif cmd == "instruct":
                    # Shortcut for storing instructions with high importance
                    if not args:
                        print("Usage: /instruct <instruction text>")
                        print("  Stores as instruction with importance=9")
                        continue

                    success = llm.store_memory(args, memory_type="instruction", importance=9)
                    if success:
                        print(f"Storing instruction: \"{args[:50]}...\"" if len(args) > 50 else f"Storing instruction: \"{args}\"")
                        time.sleep(0.1)
                        memory.process_messages()
                        llm.get_memory_responses()  # Clear responses
                        print(f"  Instruction stored. Total memories: {len(memory.memories)}")
                    else:
                        print("Failed to store instruction")

                elif cmd == "memories":
                    # List stored memories
                    type_filter = None
                    if args:
                        try:
                            type_filter = MemoryType(args.lower())
                        except ValueError:
                            print(f"Unknown type: {args}")
                            print("  Types: fact, preference, instruction, context, episode")
                            continue

                    mems = memory.list_memories(limit=15, memory_type=type_filter)
                    if not mems:
                        print("No memories stored" + (f" of type '{args}'" if args else ""))
                    else:
                        print(f"\nMemories ({len(mems)} shown):")
                        for m in mems:
                            t = m["memory_type"]
                            imp = m["importance"]
                            content = m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"]
                            print(f"  [{t}:{imp}] {content}")

                elif cmd == "recall":
                    if not args:
                        print("Usage: /recall <query>")
                        continue

                    # Prioritize facts, preferences, instructions
                    success = llm.recall_memories(args, limit=5,
                                                   memory_types=["fact", "preference", "instruction", "context"])

                    if success:
                        print(f"Recalling: \"{args}\"")
                        time.sleep(0.1)
                        memory.process_messages()
                        responses = llm.get_memory_responses()
                        found_response = False
                        for r in responses:
                            if r.get("type") == "response":
                                found_response = True
                                memories = r.get("memories", [])
                                if memories:
                                    print(f"  Found {len(memories)} memories:")
                                    for i, m in enumerate(memories, 1):
                                        t = m.get("memory_type", "?")
                                        imp = m.get("importance", 5)
                                        content = m["content"]
                                        preview = content[:55] + "..." if len(content) > 55 else content
                                        print(f"    {i}. [{t}:{imp}] {preview}")
                                else:
                                    print("  No matching memories found")
                        if not found_response:
                            print("  No response received (memory count: {})".format(len(memory.memories)))
                    else:
                        print(f"Failed to send recall request (coherence={llm.get_bond_coherence(memory.cid):.2f})")

                elif cmd == "tick":
                    n = int(args) if args else 1
                    for _ in range(n):
                        bootstrap.tick()
                        memory.process_messages()
                    print(f"Advanced {n} tick(s). {format_kernel_state(bootstrap)}")

                elif cmd == "inject":
                    amount = float(args) if args else 10.0
                    llm.inject_resource(amount)
                    print(f"Injected {amount} F. New state: {format_creature_state(llm)}")

                elif cmd == "spawn":
                    name = args if args else f"creature_{len(bootstrap.runtime.creatures)}"
                    cid = bootstrap.spawn(name, initial_f=10.0, initial_a=0.5)
                    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING
                    print(f"Spawned creature '{name}' with cid={cid}")

                elif cmd == "list":
                    print("\nCreatures:")
                    for cid, c in bootstrap.runtime.creatures.items():
                        markers = []
                        if cid == llm.cid:
                            markers.append("LLM")
                        if cid == memory.cid:
                            markers.append("MEM")
                        marker = f" [{', '.join(markers)}]" if markers else ""
                        print(f"  [{cid}] {c.name}: F={c.F:.1f} a={c.a:.2f} state={c.state.name}{marker}")

                elif cmd == "bonds":
                    print("\nBonds:")
                    for ch_id, ch in bootstrap.runtime.channels.items():
                        c_a = bootstrap.runtime.creatures.get(ch.creature_a)
                        c_b = bootstrap.runtime.creatures.get(ch.creature_b)
                        name_a = c_a.name if c_a else f"#{ch.creature_a}"
                        name_b = c_b.name if c_b else f"#{ch.creature_b}"
                        print(f"  [{ch_id}] {name_a} <--[C:{ch.coherence:.2f}]--> {name_b}")

                elif cmd == "creatures":
                    # List available and loaded creatures
                    available = loader.list_available()
                    loaded = loader.list_loaded()

                    print("\nAvailable creatures:")
                    for c in available:
                        status = "\033[92m[loaded]\033[0m" if c.get("loaded") else ""
                        ctype = c.get("type", "?")
                        desc = c.get("description", "")[:40]
                        print(f"  {c['name']:12} ({ctype:8}) {desc} {status}")

                    if loaded:
                        print("\nLoaded creatures:")
                        for c in loaded:
                            print(f"  {c['name']:12} cid={c['cid']:2} F={c['F']:.1f} a={c['a']:.2f} mode={c['mode']}")

                elif cmd == "load":
                    if not args:
                        print("Usage: /load <name|path>")
                        print("  Load a creature by name (memory, tool, reasoner, planner)")
                        print("  Or by path to .ex (JIT) or .exb (bytecode) file")
                        print("  Example: /load tool")
                        print("  Example: /load /path/to/custom.ex")
                        continue

                    try:
                        wrapper = loader.load(args)
                        lc = loader.loaded.get(args.split('/')[-1].replace('.ex', '').replace('.exb', ''))
                        if lc:
                            print(f"Loaded '{lc.spec.name}' (cid={wrapper.cid}, mode={lc.load_mode.value})")
                            print(f"  F={wrapper.F:.1f} a={wrapper.a:.2f}")
                            if lc.spec.protocols:
                                print(f"  Protocols: {', '.join(lc.spec.protocols)}")

                            # Ask if user wants to bond with LLM
                            print(f"  Use '/bond {lc.spec.name}' to create a bond with LLM")
                        else:
                            print(f"Loaded creature (cid={wrapper.cid})")
                    except Exception as e:
                        print(f"Failed to load creature: {e}")

                elif cmd == "unload":
                    if not args:
                        print("Usage: /unload <name>")
                        continue

                    if args == "memory":
                        print("Cannot unload the primary memory creature")
                        continue

                    if loader.unload(args):
                        print(f"Unloaded '{args}'")
                    else:
                        print(f"Creature not loaded: {args}")

                elif cmd == "compile":
                    if not args:
                        print("Usage: /compile <path.ex> [output.exb]")
                        print("  Compile an Existence-Lang file to bytecode")
                        continue

                    parts = args.split()
                    source_path = parts[0]
                    output_path = parts[1] if len(parts) > 1 else None

                    try:
                        result = loader.compile_to_bytecode(source_path, output_path)
                        print(f"Compiled to: {result}")
                    except Exception as e:
                        print(f"Compilation failed: {e}")

                elif cmd == "bond":
                    # Bond a loaded creature with LLM
                    if not args:
                        print("Usage: /bond <creature_name> [coherence]")
                        print("  Create a bond between a loaded creature and LLM")
                        continue

                    parts = args.split()
                    creature_name = parts[0]
                    coherence = float(parts[1]) if len(parts) > 1 else 1.0

                    if creature_name not in loader.loaded:
                        print(f"Creature not loaded: {creature_name}")
                        print("  Use /creatures to see loaded creatures")
                        continue

                    try:
                        channel_id = loader.bond_to(creature_name, llm, coherence)
                        print(f"Bond created: LLM <--[ch:{channel_id}, C:{coherence}]--> {creature_name}")
                    except Exception as e:
                        print(f"Failed to create bond: {e}")

                elif cmd == "help":
                    print("\nCommands:")
                    print("  /state      - Show LLM creature state")
                    print("  /memory     - Show memory creature state")
                    print("  /memories   - List stored memories (/memories [type])")
                    print("  /kernel     - Show kernel state")
                    print("  /store      - Store memory (/store [type:importance] text)")
                    print("  /instruct   - Store instruction (/instruct text)")
                    print("  /recall     - Recall memories (/recall query)")
                    print("  /tick [n]   - Advance kernel by n ticks")
                    print("  /inject f   - Inject F resource into LLM creature")
                    print("  /list       - List all creatures in kernel")
                    print("  /bonds      - Show bonds between creatures")
                    print("\nCreature Loading:")
                    print("  /creatures  - List available/loaded creatures")
                    print("  /load <n>   - Load creature (name or path to .ex/.exb)")
                    print("  /unload <n> - Unload a creature")
                    print("  /bond <n>   - Bond loaded creature with LLM (/bond name [coherence])")
                    print("  /compile <p>- Compile .ex to .exb bytecode")
                    print("\n  /quit       - Exit")
                    print("\nMemory types: fact, preference, instruction, context, episode")
                    print("Importance: 1-10 (higher = more important to remember)")
                    print("Creature types: memory, tool, reasoner, planner")

                else:
                    print(f"Unknown command: /{cmd}")

                continue

            # Regular input - send to LLM creature
            if not llm.is_alive:
                print("\033[91mLLM creature has died. Use /inject to revive.\033[0m")
                continue

            # First, recall relevant memories (prioritize instructions, facts, preferences)
            context_memories = []
            memory_info = []
            if llm.memory_cid:
                # Get standing instructions first
                llm.get_instructions()
                time.sleep(0.05)
                memory.process_messages()
                responses = llm.get_memory_responses()
                instructions = []
                for r in responses:
                    if r.get("type") == "instructions":
                        instructions = r.get("instructions", [])

                # Then recall relevant memories
                llm.recall_memories(user_input, limit=5,
                                    memory_types=["fact", "preference", "instruction", "context"])
                time.sleep(0.05)
                memory.process_messages()
                responses = llm.get_memory_responses()
                for r in responses:
                    if r.get("type") == "response":
                        for m in r.get("memories", []):
                            content = m["content"]
                            mtype = m.get("memory_type", "context")
                            # Skip instructions we already have
                            if mtype != "instruction" or content not in instructions:
                                context_memories.append(content)
                                memory_info.append(f"[{mtype}] {content[:30]}...")

                # Add instructions to context
                context_memories = instructions + context_memories

            print("\n\033[93mThinking...\033[0m", end=" ", flush=True)
            response, stats = llm.think(user_input, context_memories)

            if "error" in stats:
                print(f"\n{response}")
            else:
                mem_note = f", memories: {stats['used_memories']}" if stats.get('used_memories') else ""
                print(f"({stats['total_cost']:.2f} F{mem_note})")
                print(f"\n\033[92mLLM>\033[0m {response}")

                # Show which memories were used (if any)
                if memory_info and debug:
                    print(f"\n\033[90m[Used memories: {', '.join(memory_info[:3])}]\033[0m")

                print(f"\n\033[90m[tokens: {stats['input_tokens']}→{stats['output_tokens']}, "
                      f"cost: {stats['total_cost']:.2f} F, "
                      f"remaining: {stats['F_remaining']:.1f} F, "
                      f"time: {stats['elapsed']:.1f}s]\033[0m")

                # LLM-driven memory extraction
                # Ask LLM what facts/preferences/instructions should be remembered
                extracted = llm.extract_memories(user_input, response)
                if extracted:
                    for mem in extracted:
                        llm.store_memory(
                            mem["content"],
                            memory_type=mem["memory_type"],
                            importance=mem["importance"]
                        )
                    memory.process_messages()
                    llm.get_memory_responses()  # Clear acks
                    if debug:
                        print(f"\033[90m[Extracted {len(extracted)} memories]\033[0m")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Use /quit to exit.")
            continue
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\033[91mError: {e}\033[0m")
            if debug:
                import traceback
                traceback.print_exc()

    tick_running = False
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="DET-OS CLI - LLM as Existence-Lang Creature"
    )
    parser.add_argument(
        "--model", "-m",
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL"
    )
    parser.add_argument(
        "--initial-f", "-F",
        type=float,
        default=100.0,
        help="Initial resource (F) for LLM creature (default: 100)"
    )
    parser.add_argument(
        "--initial-a", "-a",
        type=float,
        default=0.8,
        help="Initial agency (a) for LLM creature (default: 0.8)"
    )
    parser.add_argument(
        "--tick-rate",
        type=float,
        default=10.0,
        help="Kernel tick rate in Hz (default: 10)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    sys.exit(run_cli(
        model=args.model,
        ollama_url=args.ollama_url,
        initial_f=args.initial_f,
        initial_a=args.initial_a,
        tick_rate=args.tick_rate,
        debug=args.debug
    ))


if __name__ == "__main__":
    main()
