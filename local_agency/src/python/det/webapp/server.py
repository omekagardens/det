"""
DET Web Server
==============

FastAPI-based web server for DET visualization and control.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.responses import HTMLResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Stubs for type hints when fastapi not installed
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None
    Request = None
    HTMLResponse = None

from ..core import DETCore
from ..harness import HarnessController, HarnessEvent, create_harness
from ..metrics import MetricsCollector, DETEventType, Profiler, create_metrics_collector, create_profiler
from ..llm import DETLLMInterface, OllamaClient


class ConnectionManager:
    """Manages WebSocket connections for broadcasting state updates."""

    def __init__(self):
        self.active_connections: List["WebSocket"] = []

    async def connect(self, websocket: "WebSocket"):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: "WebSocket"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                # Use timeout to prevent blocking on slow connections
                await asyncio.wait_for(
                    connection.send_json(message),
                    timeout=5.0
                )
            except (asyncio.TimeoutError, Exception):
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


class DETWebApp:
    """
    DET Web Application wrapper.

    Provides FastAPI app with WebSocket support for real-time visualization.
    """

    def __init__(
        self,
        core: Optional[DETCore] = None,
        harness: Optional[HarnessController] = None,
        update_interval: float = 0.1,
    ):
        """
        Initialize the web application.

        Args:
            core: DETCore instance (creates one if not provided).
            harness: HarnessController instance (creates one if not provided).
            update_interval: Interval for state updates (seconds).
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        self.core = core
        self.harness = harness or (create_harness(core=core) if core else None)
        self.update_interval = update_interval
        self.connection_manager = ConnectionManager()

        # Background task handle
        self._update_task: Optional[asyncio.Task] = None
        self._running = False

        # Event history for new connections
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 100

        # Phase 6.4: Metrics and profiling
        self.metrics = create_metrics_collector(max_samples=1000, max_events=500)
        self.profiler = create_profiler(window_size=100)
        self._metrics_interval = 1.0  # Sample metrics every second
        self._last_metrics_sample = 0.0

        # Visualization update throttling
        self._viz_interval = 0.5  # Full viz data every 0.5 seconds
        self._last_viz_update = 0.0
        self._max_viz_nodes = 100  # Limit nodes sent for performance

        # LLM interface for chat (lazy initialized)
        self._llm_interface: Optional[DETLLMInterface] = None
        self._ollama_url = "http://localhost:11434"
        self._model = "llama3.2:3b"

        # Setup FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> "FastAPI":
        """Create and configure the FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: "FastAPI"):
            # Startup
            self._running = True
            self._update_task = asyncio.create_task(self._broadcast_loop())
            yield
            # Shutdown
            self._running = False
            if self._update_task:
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    pass

        app = FastAPI(
            title="DET Mind Viewer",
            description="Real-time visualization of DET mind state",
            version="0.6.0",
            lifespan=lifespan,
        )

        # Static files and templates
        static_dir = Path(__file__).parent / "static"
        templates_dir = Path(__file__).parent / "templates"

        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        if templates_dir.exists():
            templates = Jinja2Templates(directory=str(templates_dir))
        else:
            templates = None

        # Routes
        @app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Main visualization page."""
            if templates:
                return templates.TemplateResponse("index.html", {"request": request})
            return HTMLResponse(content=self._get_fallback_html(), status_code=200)

        @app.get("/api/status")
        async def get_status():
            """Get current DET status."""
            return self._get_state()

        @app.get("/api/nodes")
        async def get_nodes():
            """Get all node states."""
            return self._get_nodes()

        @app.get("/api/bonds")
        async def get_bonds():
            """Get all bond states."""
            return self._get_bonds()

        @app.get("/api/node/{node_id}")
        async def get_node(node_id: int):
            """Get specific node state."""
            if self.harness:
                state = self.harness.get_node_state(node_id)
                if state:
                    return state
            return {"error": "Node not found"}

        @app.post("/api/step")
        async def do_step(n: int = 1, dt: float = 0.1):
            """Execute simulation steps."""
            if self.harness:
                count = self.harness.step_n(n, dt)
                return {"steps": count, "tick": self.core.tick if self.core else 0}
            return {"error": "No harness available"}

        @app.post("/api/pause")
        async def do_pause():
            """Pause simulation."""
            if self.harness:
                self.harness.pause()
                return {"paused": True}
            return {"error": "No harness available"}

        @app.post("/api/resume")
        async def do_resume():
            """Resume simulation."""
            if self.harness:
                self.harness.resume()
                return {"paused": False}
            return {"error": "No harness available"}

        @app.post("/api/speed")
        async def set_speed(speed: float = 1.0):
            """Set simulation speed."""
            if self.harness:
                self.harness.set_speed(speed)
                return {"speed": self.harness.speed}
            return {"error": "No harness available"}

        @app.post("/api/inject_f")
        async def inject_f(node: int, amount: float):
            """Inject resource F into a node."""
            if self.harness:
                result = self.harness.inject_f(node, amount)
                return {"success": result}
            return {"error": "No harness available"}

        @app.post("/api/snapshot")
        async def take_snapshot(name: str):
            """Take a state snapshot."""
            if self.harness:
                result = self.harness.take_snapshot(name)
                return {"success": result}
            return {"error": "No harness available"}

        @app.post("/api/restore")
        async def restore_snapshot(name: str):
            """Restore a state snapshot."""
            if self.harness:
                result = self.harness.restore_snapshot(name)
                return {"success": result}
            return {"error": "No harness available"}

        @app.get("/api/snapshots")
        async def list_snapshots():
            """List available snapshots."""
            if self.harness:
                return self.harness.list_snapshots()
            return []

        @app.get("/api/events")
        async def get_events(limit: int = 50):
            """Get recent events."""
            if self.harness:
                return self.harness.get_events(limit=limit)
            return []

        # Phase 6.4: Metrics endpoints
        @app.get("/api/metrics/dashboard")
        async def get_dashboard():
            """Get metrics dashboard data."""
            return self.metrics.get_dashboard()

        @app.get("/api/metrics/samples")
        async def get_samples(limit: int = 100):
            """Get recent metric samples."""
            return self.metrics.get_samples(limit=limit)

        @app.get("/api/metrics/timeline/{field}")
        async def get_timeline(field: str, limit: int = 200):
            """Get timeline data for a specific field."""
            return self.metrics.get_timeline(field, limit=limit)

        @app.get("/api/metrics/events")
        async def get_metric_events(limit: int = 100, event_type: Optional[str] = None):
            """Get DET events (escalation, compilation, etc.)."""
            et = DETEventType(event_type) if event_type else None
            return self.metrics.get_events(limit=limit, event_type=et)

        @app.get("/api/metrics/statistics")
        async def get_statistics():
            """Get statistical summary of all metrics."""
            return self.metrics.get_statistics()

        @app.get("/api/metrics/profiling")
        async def get_profiling():
            """Get performance profiling data."""
            return self.profiler.get_report()

        # Somatic (Physical I/O) endpoints
        @app.get("/api/somatic")
        async def get_somatic():
            """Get all somatic nodes."""
            if not self.core:
                return {"error": "No core available", "somatic": []}
            return {"somatic": self.core.get_all_somatic()}

        @app.post("/api/somatic/create")
        async def create_somatic(request: Request):
            """Create a new somatic node."""
            if not self.core:
                return {"error": "No core available", "success": False}
            data = await request.json()
            from ..core import SomaticType
            try:
                type_val = data.get("type", 15)  # Default to GENERIC_SENSOR
                # Handle both integer and string type values
                if isinstance(type_val, int):
                    somatic_type = SomaticType(type_val)
                else:
                    somatic_type = SomaticType[str(type_val).upper()]
                idx = self.core.create_somatic(
                    somatic_type=somatic_type,
                    name=data.get("name", "unnamed"),
                    is_virtual=data.get("is_virtual", True),
                    remote_id=data.get("remote_id", 0),
                    channel=data.get("channel", 0),
                )
                if idx >= 0:
                    return {"success": True, "somatic_idx": idx}
                return {"success": False, "error": "Failed to create somatic node"}
            except (KeyError, ValueError) as e:
                return {"success": False, "error": f"Invalid somatic type: {e}"}

        @app.post("/api/somatic/remove")
        async def remove_somatic(request: Request):
            """Remove a somatic node."""
            if not self.core:
                return {"error": "No core available", "success": False}
            data = await request.json()
            idx = data.get("idx", -1)
            success = self.core.remove_somatic(idx)
            return {"success": success}

        @app.post("/api/somatic/set_target")
        async def set_somatic_target(request: Request):
            """Set somatic actuator target."""
            if not self.core:
                return {"error": "No core available", "success": False}
            data = await request.json()
            idx = data.get("idx", -1)
            target = data.get("target", 0.0)
            self.core.set_somatic_target(idx, target)
            return {"success": True}

        @app.post("/api/somatic/update_value")
        async def update_somatic_value(request: Request):
            """Update somatic sensor value (for external input)."""
            if not self.core:
                return {"error": "No core available", "success": False}
            data = await request.json()
            idx = data.get("somatic_idx", data.get("idx", -1))
            value = data.get("value", 0.0)
            raw_value = data.get("raw_value", value)
            self.core.update_somatic_value(idx, value, raw_value)
            return {"success": True}

        @app.post("/api/somatic/simulate")
        async def simulate_somatic(request: Request):
            """Simulate one step of virtual somatic nodes."""
            if not self.core:
                return {"error": "No core available", "success": False}
            data = await request.json()
            dt = data.get("dt", 0.1)
            self.core.simulate_somatic(dt)
            return {"success": True}

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: "WebSocket"):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket)

            # Send initial state
            try:
                await websocket.send_json({
                    "type": "init",
                    "data": self._get_full_state(),
                })

                # Send event history
                for event in self._event_history[-20:]:
                    await websocket.send_json({
                        "type": "event",
                        "data": event,
                    })

                # Keep connection alive and handle messages
                while True:
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_json(),
                            timeout=30.0  # Longer timeout to handle slow operations
                        )
                        # Handle message in background to not block pings
                        asyncio.create_task(
                            self._handle_ws_message_safe(websocket, data)
                        )
                    except asyncio.TimeoutError:
                        # Send ping to keep alive
                        try:
                            await asyncio.wait_for(
                                websocket.send_json({"type": "ping"}),
                                timeout=5.0
                            )
                        except (asyncio.TimeoutError, Exception):
                            break  # Connection lost
                    except asyncio.CancelledError:
                        break  # Task cancelled, exit gracefully

            except WebSocketDisconnect:
                pass  # Normal disconnect
            except asyncio.CancelledError:
                pass  # Task cancelled
            except Exception as e:
                # Log unexpected errors but don't crash
                import sys
                print(f"WebSocket error: {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                self.connection_manager.disconnect(websocket)

        return app

    async def _handle_ws_message_safe(self, websocket: "WebSocket", data: Dict[str, Any]):
        """Safely handle WebSocket messages with error catching."""
        try:
            await self._handle_ws_message(websocket, data)
        except Exception as e:
            # Don't let message handling errors crash the connection
            import sys
            print(f"WebSocket message error: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                await websocket.send_json({
                    "type": "error",
                    "data": {"error": str(e)}
                })
            except Exception:
                pass  # Connection may be closed

    async def _handle_ws_message(self, websocket: "WebSocket", data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type", "")

        if msg_type == "pong":
            pass  # Keep-alive response

        elif msg_type == "step":
            if self.harness:
                n = data.get("n", 1)
                dt = data.get("dt", 0.1)
                self.harness.step_n(n, dt)

        elif msg_type == "pause":
            if self.harness:
                self.harness.pause()

        elif msg_type == "resume":
            if self.harness:
                self.harness.resume()

        elif msg_type == "speed":
            if self.harness:
                speed = data.get("speed", 1.0)
                self.harness.set_speed(speed)

        elif msg_type == "inject_f":
            if self.harness:
                node = data.get("node", 0)
                amount = data.get("amount", 0.1)
                self.harness.inject_f(node, amount)

        elif msg_type == "snapshot":
            if self.harness:
                name = data.get("name", f"snap_{int(time.time())}")
                self.harness.take_snapshot(name)

        elif msg_type == "chat":
            message = data.get("message", "")
            if message:
                await self._handle_chat(websocket, message)

    async def _handle_chat(self, websocket: "WebSocket", message: str):
        """Handle a chat message from the client."""
        try:
            # Initialize LLM interface if needed
            if self._llm_interface is None and self.core:
                self._llm_interface = DETLLMInterface(
                    self.core,
                    ollama_url=self._ollama_url,
                    model=self._model
                )

            if self._llm_interface is None:
                await websocket.send_json({
                    "type": "chat_error",
                    "data": {"error": "LLM interface not available"}
                })
                return

            # Check if Ollama is available
            client = OllamaClient(base_url=self._ollama_url, model=self._model)
            if not client.is_available():
                await websocket.send_json({
                    "type": "chat_error",
                    "data": {"error": "Ollama not running. Start with: ollama serve"}
                })
                return

            # Process through DET interface (run in thread pool to avoid blocking)
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._llm_interface.process_request(message)
            )

            # Send response
            decision = result.get("decision")
            decision_name = decision.name if hasattr(decision, "name") else str(decision)

            await websocket.send_json({
                "type": "chat_response",
                "data": {
                    "response": result.get("response", ""),
                    "decision": decision_name,
                    "intent": result.get("intent", ""),
                    "domain": result.get("domain", ""),
                }
            })

            # Log the event
            self.metrics.log_event(
                DETEventType.GATEKEEPER_DECISION,
                tick=self.core.tick if self.core else 0,
                decision=decision_name,
                intent=result.get("intent", ""),
            )

        except Exception as e:
            await websocket.send_json({
                "type": "chat_error",
                "data": {"error": str(e)}
            })

    async def _broadcast_loop(self):
        """Background task to broadcast state updates and run simulation."""
        while self._running:
            try:
                now = time.time()

                # Auto-step simulation if not paused
                if self.harness and not self.harness.paused and self.core:
                    # Profile the step
                    self.profiler.start_tick()
                    self.metrics.start_tick()

                    dt = 0.1 * self.harness.speed
                    self.core.step(dt)

                    self.profiler.end_tick()
                    self.metrics.end_tick()

                # Sample metrics periodically
                if now - self._last_metrics_sample >= self._metrics_interval:
                    self.metrics.sample(self.core)
                    self.profiler.sample_memory()
                    self._last_metrics_sample = now

                if self.connection_manager.connection_count > 0:
                    # Always send basic state
                    state = self._get_state()

                    # Only send full viz data periodically to reduce load
                    if now - self._last_viz_update >= self._viz_interval:
                        state["nodes"] = self._get_nodes_viz()
                        state["bonds"] = self._get_bonds()
                        self._last_viz_update = now

                    await self.connection_manager.broadcast({
                        "type": "state",
                        "data": state,
                    })

                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Broadcast error: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    def _get_state(self) -> Dict[str, Any]:
        """Get current DET state for broadcasting."""
        if not self.core:
            return {"error": "No core available"}

        return {
            "tick": self.core.tick,
            "num_active": self.core.num_active,
            "num_bonds": self.core.num_bonds,
            "num_somatic": self.core.num_somatic,
            "aggregates": self.harness.get_aggregates() if self.harness else {},
            "affect": self.harness.get_affect() if self.harness else {},
            "emotional_state": self.harness.get_emotional_state() if self.harness else "unknown",
            "self_cluster_size": len(self.harness.get_self_cluster()) if self.harness else 0,
            "paused": self.harness.paused if self.harness else False,
            "speed": self.harness.speed if self.harness else 1.0,
            "timestamp": time.time(),
        }

    def _get_full_state(self) -> Dict[str, Any]:
        """Get full state including nodes, bonds, and somatic."""
        state = self._get_state()
        state["nodes"] = self._get_nodes_viz()  # Use viz data with positions
        state["bonds"] = self._get_bonds()
        state["somatic"] = self.core.get_all_somatic() if self.core else []
        return state

    def _get_nodes(self) -> List[Dict[str, Any]]:
        """Get all node states."""
        if not self.core:
            return []

        nodes = []
        for i in range(self.core.num_active):
            node = self.core._core.contents.nodes[i]
            nodes.append({
                "index": i,
                "layer": "P" if i < 16 else "A",
                "F": node.F,
                "q": node.q,
                "a": node.a,
                "theta": node.theta,
                "P": node.P,
                "active": node.active,
                "v": node.affect.v,
                "r": node.affect.r,
                "b": node.affect.b,
            })
        return nodes

    def _get_nodes_viz(self) -> List[Dict[str, Any]]:
        """Get nodes with 3D visualization data."""
        if not self.core:
            return []

        import math

        # Get self-cluster for highlighting
        self.core._lib.det_core_identify_self(self.core._core)
        self_struct = self.core._core.contents.self
        self_cluster = set()
        for i in range(self_struct.num_nodes):
            self_cluster.add(self_struct.nodes[i])

        # Domain names for LLM areas
        domain_names = ["math", "language", "tool_use", "science"]

        nodes = []
        # Limit nodes for performance - prioritize P-layer and self-cluster
        num_to_send = min(self.core.num_active, self._max_viz_nodes)
        for i in range(num_to_send):
            node = self.core._core.contents.nodes[i]

            # Position based on layer and domain
            if i < 16:  # P-layer: center cluster
                angle = (i / 16) * 2 * math.pi + node.theta
                radius = 1.5
                y = 0.0
            else:  # A-layer: grouped by domain (4 quadrants)
                a_idx = i - 16
                domain = node.domain if hasattr(node, 'domain') else (a_idx // 64) % 4
                domain_angle = (domain / 4) * 2 * math.pi  # Base angle for domain
                local_idx = a_idx % 64
                local_angle = (local_idx / 64) * (math.pi / 2)  # Spread within quadrant
                angle = domain_angle + local_angle + node.theta * 0.5
                radius = 3.0 + (local_idx % 8) * 0.3  # Vary radius within domain
                y = (local_idx // 8 - 4) * 0.3  # Slight vertical spread

            x = math.cos(angle) * radius
            z = math.sin(angle) * radius

            # Color based on affect (valence: red-green, arousal: brightness)
            v = (node.affect.v + 1) / 2  # Normalize to 0-1
            brightness = 0.3 + node.affect.r * 0.7
            red = (1 - v) * brightness
            green = v * brightness
            blue = node.affect.b * brightness

            # Get domain for A-layer nodes
            domain_idx = 0
            if i >= 16:
                a_idx = i - 16
                domain_idx = (a_idx // 64) % 4

            nodes.append({
                "id": i,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "size": round(0.08 + node.a * 0.12, 3),  # Smaller nodes: 0.08-0.20
                "color": {"r": round(red, 3), "g": round(green, 3), "b": round(blue, 3)},
                "layer": "P" if i < 16 else "A",
                "domain": domain_names[domain_idx] if i >= 16 else "core",
                "in_self": i in self_cluster,
                "P": round(node.P, 3),
                "a": round(node.a, 3),
                "F": round(node.F, 3),
                "theta": round(node.theta, 3),
            })

        # Add somatic nodes in a separate cluster (below main cluster)
        somatic_list = self.core.get_all_somatic()
        for i, somatic in enumerate(somatic_list):
            # Position in a ring below the main cluster
            angle = (i / max(len(somatic_list), 1)) * 2 * math.pi
            radius = 2.0
            x = math.cos(angle) * radius
            y = -2.0  # Below main cluster
            z = math.sin(angle) * radius

            # Color based on type (sensors=blue, actuators=orange)
            if somatic.get("is_sensor"):
                red, green, blue_c = 0.2, 0.5, 0.9  # Blue for sensors
            else:
                red, green, blue_c = 0.9, 0.5, 0.2  # Orange for actuators

            # Brightness based on value
            brightness = 0.4 + somatic["value"] * 0.6
            red *= brightness
            green *= brightness
            blue_c *= brightness

            nodes.append({
                "id": f"somatic_{somatic['idx']}",
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "size": 0.15,  # Fixed size for somatic nodes
                "color": {"r": round(red, 3), "g": round(green, 3), "b": round(blue_c, 3)},
                "layer": "S",  # Somatic layer
                "domain": somatic["type_name"],
                "in_self": False,
                "P": round(somatic["value"], 3),
                "a": round(somatic["target"], 3),
                "F": 0.5,
                "theta": 0.0,
                "is_somatic": True,
                "somatic_name": somatic["name"],
            })

        return nodes

    def _get_bonds(self) -> List[Dict[str, Any]]:
        """Get all bond states."""
        if not self.core:
            return []

        bonds = []
        for i in range(self.core.num_bonds):
            bond = self.core._core.contents.bonds[i]
            if bond.C > 0.01:  # Only include active bonds
                # Calculate flux for info flow visualization
                flux = abs(bond.flux_ema) if hasattr(bond, 'flux_ema') else 0.0
                bonds.append({
                    "source": bond.i,
                    "target": bond.j,
                    "C": round(bond.C, 3),
                    "strength": round(bond.C, 3),
                    "pi": round(bond.pi, 3),
                    "flux": round(flux, 3),
                    "is_cross_layer": bool(bond.is_cross_layer),
                })
        return bonds

    def _get_fallback_html(self) -> str:
        """Return fallback HTML if templates not found."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>DET Mind Viewer</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; }
        .status { background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .error { color: #ff6b6b; }
    </style>
</head>
<body>
    <h1>DET Mind Viewer</h1>
    <div class="status">
        <p>Templates not found. Using fallback page.</p>
        <p>API endpoints available at /api/*</p>
        <p>WebSocket available at /ws</p>
    </div>
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onmessage = (event) => {
            console.log('DET State:', JSON.parse(event.data));
        };
    </script>
</body>
</html>
"""

    def add_event_callback(self):
        """Register event callback to track harness events."""
        if self.harness:
            def on_event(event: HarnessEvent):
                event_data = event.to_dict()
                self._event_history.append(event_data)
                if len(self._event_history) > self._max_history:
                    self._event_history.pop(0)

            self.harness.add_event_callback(on_event)


def create_app(
    core: Optional[DETCore] = None,
    harness: Optional[HarnessController] = None,
    update_interval: float = 0.1,
) -> "FastAPI":
    """
    Create a DET web application.

    Args:
        core: DETCore instance (creates one if not provided).
        harness: HarnessController instance.
        update_interval: State update interval in seconds.

    Returns:
        FastAPI application instance.
    """
    webapp = DETWebApp(core=core, harness=harness, update_interval=update_interval)
    webapp.add_event_callback()
    return webapp.app


def run_server(
    core: Optional[DETCore] = None,
    harness: Optional[HarnessController] = None,
    host: str = "127.0.0.1",
    port: int = 8420,
    update_interval: float = 0.1,
    start_paused: bool = True,
):
    """
    Run the DET web server.

    Args:
        core: DETCore instance.
        harness: HarnessController instance.
        host: Host to bind to.
        port: Port to bind to.
        update_interval: State update interval.
        start_paused: Whether to start simulation in paused state.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: pip install uvicorn")

    # Create harness if not provided, starting paused
    if harness is None and core is not None:
        from ..harness import create_harness
        harness = create_harness(core=core, start_paused=start_paused)
    elif harness is not None and start_paused:
        harness.pause()

    webapp = DETWebApp(core=core, harness=harness, update_interval=update_interval)
    webapp.add_event_callback()

    paused_msg = " (PAUSED)" if start_paused else ""
    print(f"\n{'='*60}")
    print(f"  DET Mind Viewer{paused_msg}")
    print(f"  Open http://{host}:{port} in your browser")
    print(f"{'='*60}\n")

    # Configure uvicorn with longer websocket timeouts
    # These are generous to handle slow LLM operations
    config = uvicorn.Config(
        webapp.app,
        host=host,
        port=port,
        log_level="warning",
        ws_ping_interval=30.0,   # Send ping every 30 seconds
        ws_ping_timeout=120.0,   # Wait 120 seconds for pong (long LLM ops)
        timeout_keep_alive=120,  # HTTP keep-alive timeout
    )
    server = uvicorn.Server(config)
    server.run()


def create_app(
    core: Optional[DETCore] = None,
    harness: Optional[HarnessController] = None,
    update_interval: float = 0.1,
):
    """
    Create a DET webapp without running it.

    Args:
        core: DETCore instance.
        harness: HarnessController instance.
        update_interval: State update interval.

    Returns:
        FastAPI app instance.
    """
    webapp = DETWebApp(core=core, harness=harness, update_interval=update_interval)
    webapp.add_event_callback()
    return webapp.app
