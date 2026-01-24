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
        ↓
    DET-OS Kernel (kernel.ex)
        ↓
    Substrate v2 (C/Metal)
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent))

from det.os.existence.bootstrap import DETOSBootstrap, BootConfig, BootState
from det.os.existence.runtime import CreatureState


class LLMCreature:
    """
    LLM Agent as a DET-OS creature.

    The LLM creature:
    - Has resource (F) that depletes when generating tokens
    - Has agency (a) that determines capabilities
    - Creates bonds to memory creatures
    - Is subject to presence-based scheduling
    """

    def __init__(self, runtime, cid: int, ollama_url: str, model: str):
        self.runtime = runtime
        self.cid = cid
        self.ollama_url = ollama_url
        self.model = model
        self.client = None
        self.conversation_history = []

        # Token costs (F consumed per token)
        self.cost_per_input_token = 0.01
        self.cost_per_output_token = 0.05

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

    @property
    def creature(self):
        """Get the runtime creature object."""
        return self.runtime.creatures.get(self.cid)

    @property
    def F(self) -> float:
        """Current resource level."""
        c = self.creature
        return c.F if c else 0.0

    @property
    def a(self) -> float:
        """Current agency level."""
        c = self.creature
        return c.a if c else 0.0

    @property
    def presence(self) -> float:
        """Compute presence: P = F * a."""
        return self.F * self.a

    @property
    def is_alive(self) -> bool:
        """Check if creature is still alive."""
        c = self.creature
        return c is not None and c.is_alive()

    def can_think(self, estimated_tokens: int = 100) -> bool:
        """Check if we have enough F to think."""
        estimated_cost = estimated_tokens * self.cost_per_output_token
        return self.F >= estimated_cost

    def think(self, user_input: str) -> tuple[str, Dict[str, Any]]:
        """
        Process user input and generate response.

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

        # Add to conversation
        self.conversation_history.append({"role": "user", "content": user_input})

        # Generate response
        start_time = time.time()
        try:
            response = self.client.chat(
                messages=self.conversation_history,
                temperature=self._compute_temperature()
            )
        except Exception as e:
            return f"[Error: {e}]", {"error": str(e)}

        elapsed = time.time() - start_time

        # Estimate output cost
        output_tokens = len(response.split()) * 1.3
        output_cost = output_tokens * self.cost_per_output_token

        # Deduct output cost
        self.creature.F -= output_cost

        # Add response to history
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

    def inject_resource(self, amount: float):
        """Inject resource (grace) into the creature."""
        if self.creature:
            self.creature.F += amount

    def get_state(self) -> Dict[str, Any]:
        """Get creature state for display."""
        c = self.creature
        if not c:
            return {"status": "dead"}

        return {
            "cid": self.cid,
            "name": c.name,
            "F": c.F,
            "a": c.a,
            "presence": self.presence,
            "state": c.state.name,
            "conversation_turns": len(self.conversation_history) // 2,
        }


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


def format_creature_state(creature: LLMCreature) -> str:
    """Format creature state for display."""
    state = creature.get_state()
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

    # Activate the creature
    bootstrap.runtime.creatures[llm_cid].state = CreatureState.RUNNING

    # Create LLM creature wrapper
    llm = LLMCreature(bootstrap.runtime, llm_cid, ollama_url, model)

    # Check Ollama
    print(f"Connecting to Ollama ({ollama_url})...", end=" ")
    if llm.client and llm.client.is_available():
        print(f"OK (model: {model})")
    else:
        print("\033[91mFAILED\033[0m")
        print("  Ollama not running. Start with: ollama serve")
        print("  Continuing without LLM (state exploration only)")

    print(f"\nLLM Creature: {format_creature_state(llm)}")
    print(f"Kernel: {format_kernel_state(bootstrap)}")

    print("\nCommands:")
    print("  /state    - Show LLM creature state")
    print("  /kernel   - Show kernel state")
    print("  /tick [n] - Advance kernel by n ticks (default 1)")
    print("  /inject f - Inject F resource into LLM creature")
    print("  /spawn    - Spawn another creature")
    print("  /list     - List all creatures")
    print("  /help     - Show this help")
    print("  /quit     - Exit")
    print("\n" + "-" * 60)

    # Background tick thread
    import threading
    tick_running = True

    def tick_loop():
        while tick_running and bootstrap.state == BootState.RUNNING:
            try:
                bootstrap.tick()
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
                    state = llm.get_state()
                    print("\nLLM Creature State:")
                    for k, v in state.items():
                        print(f"  {k}: {v}")

                elif cmd == "kernel":
                    stats = bootstrap.get_stats()
                    print("\nKernel State:")
                    for k, v in stats.items():
                        if k != "boot_log":
                            print(f"  {k}: {v}")

                elif cmd == "tick":
                    n = int(args) if args else 1
                    for _ in range(n):
                        bootstrap.tick()
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
                        marker = " *" if cid == llm.cid else ""
                        print(f"  [{cid}] {c.name}: F={c.F:.1f} a={c.a:.2f} state={c.state.name}{marker}")

                elif cmd == "help":
                    print("\nCommands:")
                    print("  /state    - Show LLM creature state")
                    print("  /kernel   - Show kernel state")
                    print("  /tick [n] - Advance kernel by n ticks")
                    print("  /inject f - Inject F resource into LLM creature")
                    print("  /spawn    - Spawn another creature")
                    print("  /list     - List all creatures")
                    print("  /quit     - Exit")

                else:
                    print(f"Unknown command: /{cmd}")

                continue

            # Regular input - send to LLM creature
            if not llm.is_alive:
                print("\033[91mLLM creature has died. Use /inject to revive.\033[0m")
                continue

            print("\n\033[93mThinking...\033[0m", end=" ", flush=True)
            response, stats = llm.think(user_input)

            if "error" in stats:
                print(f"\n{response}")
            else:
                print(f"({stats['total_cost']:.2f} F)")
                print(f"\n\033[92mLLM>\033[0m {response}")
                print(f"\n\033[90m[tokens: {stats['input_tokens']}→{stats['output_tokens']}, "
                      f"cost: {stats['total_cost']:.2f} F, "
                      f"remaining: {stats['F_remaining']:.1f} F, "
                      f"time: {stats['elapsed']:.1f}s]\033[0m")

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
