#!/usr/bin/env python3
"""
DET Local Agency - CLI REPL
===========================

Command-line interface for interacting with the DET-LLM system.
Integrates memory management and internal dialogue.
"""

import sys
import argparse
import threading
import webbrowser
from typing import Optional
from pathlib import Path


def format_affect(valence: float, arousal: float, bondedness: float) -> str:
    """Format affect values as a visual bar."""
    def bar(val: float, label: str, color_pos: str = "\033[92m", color_neg: str = "\033[91m") -> str:
        if label == "V":
            width = 10
            center = width // 2
            filled = int(abs(val) * center)
            if val >= 0:
                bar_str = " " * center + color_pos + "=" * filled + "\033[0m" + " " * (center - filled)
            else:
                bar_str = " " * (center - filled) + color_neg + "=" * filled + "\033[0m" + " " * center
            return f"{label}[{bar_str}]"
        else:
            width = 10
            filled = int(val * width)
            return f"{label}[{color_pos}{'=' * filled}\033[0m{' ' * (width - filled)}]"

    v_bar = bar(valence, "V")
    a_bar = bar(arousal, "A", "\033[93m", "\033[93m")
    b_bar = bar(bondedness, "B", "\033[94m", "\033[94m")

    return f"{v_bar} {a_bar} {b_bar}"


def format_state(state: dict) -> str:
    """Format DET state for display."""
    lines = []

    affect = state.get("self_affect", {})
    v = affect.get("valence", 0)
    a = affect.get("arousal", 0)
    b = affect.get("bondedness", 0)
    lines.append(format_affect(v, a, b))

    agg = state.get("aggregates", {})
    lines.append(
        f"P:{agg.get('presence', 0):.2f} "
        f"C:{agg.get('coherence', 0):.2f} "
        f"F:{agg.get('resource', 0):.2f} "
        f"q:{agg.get('debt', 0):.2f}"
    )

    emotion = state.get("emotion", "neutral")
    lines.append(f"Emotion: {emotion}")

    return " | ".join(lines)


def run_repl(
    model: str = "llama3.2:3b",
    ollama_url: str = "http://localhost:11434",
    show_state: bool = True,
    use_dialogue: bool = True,
    storage_path: Optional[Path] = None
):
    """
    Run the interactive REPL with memory and dialogue integration.

    Args:
        model: Ollama model to use.
        ollama_url: Ollama API URL.
        show_state: Whether to show DET state after each response.
        use_dialogue: Use internal dialogue system for processing.
        storage_path: Path for memory storage.
    """
    from .core import DETCore, DETDecision
    from .llm import DETLLMInterface, OllamaClient
    from .memory import MemoryManager, MemoryDomain
    from .dialogue import InternalDialogue

    print("\n" + "=" * 60)
    print("  DET Local Agency - Interactive CLI v0.2")
    print("  Deep Existence Theory × Local LLM")
    print("=" * 60)

    # Initialize DET core
    print("\nInitializing DET core...", end=" ")
    try:
        core = DETCore()
        print(f"OK ({core.num_active} active nodes)")
        print("Warming up DET substrate...", end=" ")
        core.warmup(steps=50)
        p, c, f, q = core.get_aggregates()
        print(f"OK (P:{p:.2f} C:{c:.2f} F:{f:.2f})")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    # Initialize memory manager
    print("Initializing memory system...", end=" ")
    try:
        memory = MemoryManager(core, storage_path)
        stats = memory.get_domain_stats()
        total_memories = sum(s["entry_count"] for s in stats.values())
        print(f"OK ({total_memories} memories, {len(stats)} domains)")
    except Exception as e:
        print(f"FAILED: {e}")
        memory = None

    # Initialize Ollama
    print(f"Connecting to Ollama ({ollama_url})...", end=" ")
    client = OllamaClient(base_url=ollama_url, model=model)

    if not client.is_available():
        print("FAILED")
        print("\nOllama is not running. Please start it with: ollama serve")
        print("Then ensure the model is available: ollama pull", model)
        return 1

    print(f"OK (model: {model})")

    # Initialize interface and dialogue
    interface = DETLLMInterface(core, ollama_url=ollama_url, model=model)
    dialogue = InternalDialogue(core, client) if use_dialogue else None

    if dialogue:
        print("Internal dialogue system: ENABLED")
    else:
        print("Internal dialogue system: DISABLED")

    # Initial state
    print("\nInitial DET state:")
    print(format_state(core.inspect()))

    print("\nCommands:")
    print("  /state   - Show detailed DET state")
    print("  /inspect - Full system inspection (usage: /inspect [detailed])")
    print("  /grace   - Show grace status and inject (usage: /grace [node] [amount])")
    print("  /affect  - Show affect visualization")
    print("  /memory  - Show memory statistics")
    print("  /store   - Store a memory (usage: /store <text>)")
    print("  /recall  - Recall memories (usage: /recall <query>)")
    print("  /think   - Internal thinking (usage: /think <topic>)")
    print("  /train   - Autonomous training (usage: /train [duration] [--prompts N])")
    print("  /webapp  - Launch web visualization (usage: /webapp [port])")
    print("  /clear   - Clear conversation and memory")
    print("  /help    - Show this help")
    print("  /quit    - Exit")
    print("\n" + "-" * 60)

    # Track webapp server state (use dict for mutable reference in closures)
    webapp_state = {"thread": None, "running": False}

    while True:
        try:
            user_input = input("\n\033[96mYou>\033[0m ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if cmd in ("quit", "exit", "q"):
                    print("\nGoodbye!")
                    break

                elif cmd == "state":
                    state = core.inspect()
                    print("\n" + "=" * 50)
                    print("DET Core State")
                    print("=" * 50)
                    print(f"Tick: {state['tick']}  |  Emotion: {state['emotion']}")
                    print(f"\nAggregates:")
                    for k, v in state['aggregates'].items():
                        print(f"  {k}: {v:.4f}")
                    print(f"\nSelf Affect:")
                    for k, v in state['self_affect'].items():
                        print(f"  {k}: {v:.4f}")
                    print(f"\nGrace Status:")
                    print(f"  Nodes needing grace: {state['grace']['nodes_needing']}")
                    print(f"  Total grace needed: {state['grace']['total_needed']:.4f}")
                    print(f"\nDomains:")
                    for name, info in state['domains'].items():
                        print(f"  {name}: C={info['coherence']:.3f} F={info['avg_F']:.3f} ({info['count']} nodes)")
                    print(f"\nCounts:")
                    for k, v in state['counts'].items():
                        print(f"  {k}: {v}")
                    print("=" * 50)

                elif cmd == "inspect":
                    import json
                    detailed = args.lower() == "detailed"
                    state = core.inspect(detailed=detailed)
                    print("\n" + "=" * 60)
                    print(f"DET Full Inspection {'(DETAILED)' if detailed else ''}")
                    print("=" * 60)
                    print(json.dumps(state, indent=2))
                    print("=" * 60)

                elif cmd == "grace":
                    if not args:
                        # Show grace status
                        total = core.total_grace_needed()
                        print(f"\nTotal grace needed: {total:.4f}")
                        needing = []
                        for i in range(min(core.num_active, 100)):
                            if core.needs_grace(i):
                                node = core.get_node(i)
                                needing.append(f"  Node {i}: F={node.F:.3f}, q={node.q:.3f}")
                        if needing:
                            print(f"Nodes needing grace ({len(needing)}):")
                            for line in needing[:10]:
                                print(line)
                            if len(needing) > 10:
                                print(f"  ... and {len(needing)-10} more")
                        else:
                            print("No nodes currently need grace.")
                    else:
                        # Inject grace
                        parts = args.split()
                        if len(parts) >= 2:
                            node_id = int(parts[0])
                            amount = float(parts[1])
                            core.inject_grace(node_id, amount)
                            print(f"\nInjected {amount:.2f} grace to node {node_id}")
                        else:
                            print("\nUsage: /grace <node_id> <amount>")

                elif cmd == "affect":
                    v, a, b = core.get_self_affect()
                    print("\n" + format_affect(v, a, b))
                    print(f"\nValence:    {v:+.3f} (good/bad)")
                    print(f"Arousal:     {a:.3f} (activation)")
                    print(f"Bondedness:  {b:.3f} (attachment)")

                elif cmd == "memory":
                    if memory:
                        stats = memory.get_domain_stats()
                        print("\n" + "=" * 40)
                        print("Memory Domain Statistics")
                        print("=" * 40)
                        for domain_name, stat in stats.items():
                            print(f"\n{domain_name}:")
                            print(f"  Entries: {stat['entry_count']}")
                            print(f"  Coherence: {stat['coherence']:.3f}")
                            print(f"  Total accesses: {stat['total_accesses']}")
                            print(f"  Avg importance: {stat['avg_importance']:.3f}")
                        print("=" * 40)
                    else:
                        print("\nMemory system not available.")

                elif cmd == "store":
                    if not args:
                        print("\nUsage: /store <text to store>")
                    elif memory:
                        entry = memory.store(args)
                        print(f"\nStored to {entry.domain.name} domain (coherence: {entry.coherence_at_storage:.3f})")
                    else:
                        print("\nMemory system not available.")

                elif cmd == "recall":
                    if not args:
                        print("\nUsage: /recall <search query>")
                    elif memory:
                        results = memory.retrieve(args, limit=5)
                        if results:
                            print(f"\nFound {len(results)} memories:")
                            for i, entry in enumerate(results, 1):
                                preview = entry.content[:60] + "..." if len(entry.content) > 60 else entry.content
                                print(f"  {i}. [{entry.domain.name}] {preview}")
                        else:
                            print("\nNo matching memories found.")
                    else:
                        print("\nMemory system not available.")

                elif cmd == "think":
                    if not args:
                        print("\nUsage: /think <topic to think about>")
                    elif dialogue:
                        print(f"\n\033[90m[Thinking about: {args}...]\033[0m")
                        turns = dialogue.think(args, max_turns=3)
                        for i, turn in enumerate(turns, 1):
                            print(f"\n\033[93mThought {i}:\033[0m {turn.output_text[:200]}...")
                        print(f"\n\033[90m[Completed {len(turns)} thinking turns]\033[0m")
                    else:
                        print("\nDialogue system not available.")

                elif cmd == "train":
                    # Parse training arguments
                    duration = 60.0  # Default 1 minute
                    max_prompts = 0  # Unlimited

                    if args:
                        parts = args.split()
                        i = 0
                        while i < len(parts):
                            if parts[i] == "--prompts" and i + 1 < len(parts):
                                max_prompts = int(parts[i + 1])
                                i += 2
                            elif parts[i] == "--duration" and i + 1 < len(parts):
                                duration = float(parts[i + 1])
                                i += 2
                            else:
                                try:
                                    duration = float(parts[i])
                                except ValueError:
                                    pass
                                i += 1

                    print(f"\n{'='*60}")
                    print("  Starting Autonomous Training")
                    print(f"  Duration: {duration}s | Max prompts: {max_prompts or 'unlimited'}")
                    print(f"{'='*60}")
                    print("\nPress Ctrl+C to stop training early.\n")

                    try:
                        from .trainer import DETTrainer, TrainingConfig
                        import asyncio

                        config = TrainingConfig(
                            prompt_interval=5.0,
                            max_duration=duration,
                            max_prompts=max_prompts,
                            adapt_to_affect=True,
                        )

                        trainer = DETTrainer(
                            core=core,
                            client=client,
                            memory=memory,
                            config=config,
                        )

                        # Set up callbacks for live output
                        def on_prompt(prompt, response, domain):
                            print(f"\033[94m[{domain}]\033[0m {prompt[:50]}...")
                            print(f"  \033[93m→\033[0m {response[:60]}...")

                        def on_store(fact, domain):
                            print(f"\033[92m[STORED]\033[0m [{domain}] {fact[:50]}...")

                        def on_state_change(state):
                            if state.get("coherence", 1.0) < 0.3:
                                print(f"\033[91m[LOW COHERENCE]\033[0m {state['coherence']:.2f} - pausing...")

                        trainer.on_prompt = on_prompt
                        trainer.on_store = on_store
                        trainer.on_state_change = on_state_change

                        # Run training
                        stats = asyncio.run(trainer.train(duration=duration, prompts=max_prompts))

                        print(f"\n{'='*60}")
                        print("  Training Complete")
                        print(f"  Prompts: {stats.prompts_sent}")
                        print(f"  Facts stored: {stats.facts_stored}")
                        print(f"  Errors: {stats.errors}")
                        print(f"  Domains: {stats.domains_covered}")
                        print(f"{'='*60}")

                    except KeyboardInterrupt:
                        print("\n\nTraining stopped by user.")
                    except Exception as e:
                        print(f"\nTraining error: {e}")
                        import traceback
                        traceback.print_exc()

                elif cmd == "webapp":
                    if webapp_state["running"]:
                        print("\nWeb visualization is already running.")
                        print("Open http://127.0.0.1:8420 in your browser.")
                        continue

                    port = 8420
                    if args:
                        try:
                            port = int(args)
                        except ValueError:
                            print(f"\nInvalid port: {args}")
                            continue

                    try:
                        from .harness import create_harness
                        from . import WEBAPP_AVAILABLE

                        if not WEBAPP_AVAILABLE:
                            print("\nWebapp not available. Install: pip install fastapi uvicorn")
                            continue

                        harness = create_harness(core=core, start_paused=True)

                        def run_webapp():
                            try:
                                import uvicorn
                                from .webapp.server import DETWebApp

                                webapp = DETWebApp(core=core, harness=harness)
                                webapp.add_event_callback()
                                print("\nWebapp started in PAUSED state. Use controls to resume.")

                                webapp_state["running"] = True
                                # Configure with longer websocket timeouts
                                config = uvicorn.Config(
                                    webapp.app,
                                    host="127.0.0.1",
                                    port=port,
                                    log_level="warning",
                                    ws_ping_interval=20.0,
                                    ws_ping_timeout=60.0,
                                )
                                server = uvicorn.Server(config)
                                server.run()
                            except Exception as e:
                                print(f"\nWebapp error: {e}")
                            finally:
                                webapp_state["running"] = False

                        webapp_state["thread"] = threading.Thread(target=run_webapp, daemon=True)
                        webapp_state["thread"].start()

                        print(f"\nStarting web visualization on http://127.0.0.1:{port}")
                        print("Opening browser...")

                        # Give server a moment to start
                        import time
                        time.sleep(1.0)

                        webbrowser.open(f"http://127.0.0.1:{port}")

                    except ImportError as e:
                        print(f"\nWebapp not available: {e}")
                        print("Install with: pip install fastapi uvicorn")

                elif cmd == "clear":
                    interface.clear_history()
                    if dialogue:
                        dialogue.clear_history()
                    core.reset()
                    print("\nCleared conversation and reset DET state.")

                elif cmd == "somatic":
                    # Show somatic introspection
                    if dialogue and hasattr(dialogue, 'explain_last_somatic'):
                        if args == "list":
                            # List all somatic nodes
                            somatic_list = core.get_all_somatic()
                            if somatic_list:
                                print("\n" + "=" * 50)
                                print("Somatic Nodes (Physical I/O)")
                                print("=" * 50)
                                sensors = [s for s in somatic_list if s["is_sensor"]]
                                actuators = [s for s in somatic_list if s["is_actuator"]]
                                if sensors:
                                    print("\nSensors:")
                                    for s in sensors:
                                        print(f"  [{s['idx']}] {s['name']} ({s['type_name']}): {s['value']:.3f}")
                                if actuators:
                                    print("\nActuators:")
                                    for a in actuators:
                                        print(f"  [{a['idx']}] {a['name']} ({a['type_name']}): target={a['target']:.2f}, output={a['output']:.2f}")
                                print("=" * 50)
                            else:
                                print("\nNo somatic nodes configured.")
                        else:
                            # Show last somatic processing
                            explanation = dialogue.explain_last_somatic()
                            print("\n" + explanation)
                    else:
                        print("\nDialogue system with somatic bridge not available.")

                elif cmd == "help":
                    print("\nCommands:")
                    print("  /state   - Show detailed DET state")
                    print("  /inspect - Full system inspection (usage: /inspect [detailed])")
                    print("  /grace   - Show grace status and inject (usage: /grace [node] [amount])")
                    print("  /affect  - Show affect visualization")
                    print("  /memory  - Show memory statistics")
                    print("  /store   - Store a memory (usage: /store <text>)")
                    print("  /recall  - Recall memories (usage: /recall <query>)")
                    print("  /think   - Internal thinking (usage: /think <topic>)")
                    print("  /train   - Autonomous training (usage: /train [duration] [--prompts N])")
                    print("  /somatic - Show somatic processing details (usage: /somatic [list])")
                    print("  /webapp  - Launch web visualization (usage: /webapp [port])")
                    print("  /clear   - Clear conversation and memory")
                    print("  /help    - Show this help")
                    print("  /quit    - Exit")

                else:
                    print(f"\nUnknown command: /{cmd}")

                continue

            # Process through DET system
            print("\n\033[90m[Processing...]\033[0m")

            # Route the request to a domain if memory is available
            if memory:
                domain, confidence, coherence = memory.route_request(user_input)
                print(f"\033[90m[Routed to {domain.name} (conf: {confidence:.2f}, coh: {coherence:.2f})]\033[0m")

            # Use dialogue system or direct interface
            if dialogue:
                turn = dialogue.process(user_input)
                decision_name = turn.decision.name if hasattr(turn.decision, "name") else str(turn.decision)
                response = turn.output_text

                print(f"\033[90m[Decision: {decision_name}, Reformulations: {turn.reformulation_count}]\033[0m")

                # Store interaction to memory
                if memory and response:
                    memory.store(f"Q: {user_input}\nA: {response}", importance=0.6)
            else:
                result = interface.process_request(user_input)
                decision = result.get("decision")
                decision_name = decision.name if hasattr(decision, "name") else str(decision)
                response = result.get("response", "")

                print(f"\033[90m[{result['intent']}/{result['domain']} → {decision_name}]\033[0m")

            # Show response
            if response:
                print(f"\n\033[93mDET>\033[0m {response}")

            # Show state if enabled
            if show_state:
                state = core.inspect()
                print(f"\n\033[90m{format_state(state)}\033[0m")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
            continue

        except EOFError:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"\n\033[91mError: {e}\033[0m")
            import traceback
            traceback.print_exc()
            continue

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DET Local Agency - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  det-cli                     # Start with defaults
  det-cli --model mistral     # Use a different model
  det-cli --no-state          # Hide DET state display
  det-cli --no-dialogue       # Disable internal dialogue system

The DET core provides a cognitive substrate that modulates LLM behavior
based on coherence, agency, and emotional affect dynamics.

Commands in REPL:
  /state   - Show detailed DET state
  /affect  - Show affect visualization
  /memory  - Show memory statistics
  /store   - Store a memory
  /recall  - Recall memories
  /think   - Internal thinking on a topic
  /webapp  - Launch web visualization
  /clear   - Clear conversation
  /quit    - Exit
        """
    )

    parser.add_argument(
        "--model", "-m",
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)"
    )

    parser.add_argument(
        "--url", "-u",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )

    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Don't show DET state after responses"
    )

    parser.add_argument(
        "--no-dialogue",
        action="store_true",
        help="Disable internal dialogue system"
    )

    parser.add_argument(
        "--storage",
        type=Path,
        default=None,
        help="Path for memory storage (default: ~/.det_agency/memory)"
    )

    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit"
    )

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"DET Local Agency v{__version__}")
        return 0

    return run_repl(
        model=args.model,
        ollama_url=args.url,
        show_state=not args.no_state,
        use_dialogue=not args.no_dialogue,
        storage_path=args.storage,
    )


if __name__ == "__main__":
    sys.exit(main())
