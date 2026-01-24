"""
Planner Creature
================

A DET-OS creature that decomposes tasks into executable steps.
Communicates with other creatures via bonds.

This Python wrapper interfaces with the PlannerCreature defined in creatures.ex.

Planner messages:
    PLAN: {"type": "plan", "task": str, "constraints": dict}
    STEPS: {"type": "steps", "plan": list, "dependencies": dict}
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Callable
from .base import CreatureWrapper
from ..existence.runtime import ExistenceKernelRuntime, CreatureState


@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: str
    description: str
    action_type: str  # 'think', 'execute', 'store', 'recall', 'ask'
    target: str = ""  # Target creature or resource
    depends_on: List[str] = field(default_factory=list)
    estimated_cost: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.step_id,
            "description": self.description,
            "action": self.action_type,
            "target": self.target,
            "depends_on": self.depends_on,
            "cost": self.estimated_cost
        }


@dataclass
class Plan:
    """A complete plan with steps and dependencies."""
    task: str
    steps: List[PlanStep]
    dependencies: Dict[str, List[str]]  # step_id -> [dependent_step_ids]
    total_cost: float
    elapsed_ms: float

    def get_ready_steps(self, completed: Set[str]) -> List[PlanStep]:
        """Get steps that are ready to execute (all deps satisfied)."""
        ready = []
        completed_set = set(completed)
        for step in self.steps:
            if step.step_id not in completed_set:
                if all(dep in completed_set for dep in step.depends_on):
                    ready.append(step)
        return ready

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "dependencies": self.dependencies,
            "step_count": len(self.steps),
            "total_cost": self.total_cost,
            "elapsed_ms": self.elapsed_ms
        }


class PlannerCreature(CreatureWrapper):
    """
    Planner Creature - decomposes tasks into executable steps.

    Protocol:
        Other creatures send PLAN messages via their bond.
        PlannerCreature generates steps and sends STEPS back.

    Planning costs F:
        - Base cost: 0.5 F
        - Per step: 0.15 F

    The planner considers available creatures and their capabilities
    when generating plans.

    Example usage:
        llm.send_to(planner.cid, {
            "type": "plan",
            "task": "Find all Python files and count lines of code",
            "constraints": {"max_steps": 10}
        })
        planner.process_messages()
        result = llm.receive_from(planner.cid)
    """

    # Cost constants
    BASE_COST = 0.5
    STEP_COST = 0.15

    # Action type costs (estimated F for execution)
    ACTION_COSTS = {
        'think': 0.5,
        'execute': 1.0,
        'store': 0.1,
        'recall': 0.1,
        'ask': 0.2,
        'reason': 0.8,
    }

    def __init__(self, runtime: ExistenceKernelRuntime, cid: int,
                 planning_fn: Optional[Callable] = None):
        super().__init__(runtime, cid)
        self.is_planning = False
        self.total_planned = 0
        self.total_steps = 0
        self.total_cost = 0.0

        # External planning function (e.g., LLM call)
        self.planning_fn = planning_fn

        # Known creature capabilities
        self.known_creatures: Dict[str, List[str]] = {
            'tool': ['execute', 'bash', 'file'],
            'memory': ['store', 'recall'],
            'reasoner': ['reason', 'analyze'],
            'llm': ['think', 'respond', 'generate']
        }

    def can_plan(self, estimated_steps: int = 5) -> tuple:
        """Check if we can afford to plan."""
        estimated_cost = self.BASE_COST + (estimated_steps * self.STEP_COST)
        if self.F < estimated_cost:
            return False, f"Insufficient F (need {estimated_cost:.2f}, have {self.F:.2f})"
        return True, "OK"

    def plan(self, task: str, constraints: Optional[Dict] = None) -> Plan:
        """
        Generate a plan for a task.
        """
        constraints = constraints or {}
        max_steps = constraints.get('max_steps', 10)

        # Check resources
        can, reason = self.can_plan(max_steps)
        if not can:
            return Plan(
                task=task,
                steps=[],
                dependencies={},
                total_cost=0,
                elapsed_ms=0
            )

        self.is_planning = True
        start_time = time.time()
        cost = self.BASE_COST
        self.F -= self.BASE_COST

        # Generate plan
        if self.planning_fn:
            steps = self._generate_plan_external(task, constraints, max_steps)
        else:
            steps = self._generate_plan_template(task, constraints, max_steps)

        # Add step costs
        for step in steps:
            if self.F < self.STEP_COST:
                break
            self.F -= self.STEP_COST
            cost += self.STEP_COST
            self.total_steps += 1

        # Build dependency graph
        dependencies = self._build_dependencies(steps)

        elapsed_ms = (time.time() - start_time) * 1000

        self.is_planning = False
        self.total_planned += 1
        self.total_cost += cost

        return Plan(
            task=task,
            steps=steps,
            dependencies=dependencies,
            total_cost=cost,
            elapsed_ms=elapsed_ms
        )

    def _generate_plan_template(self, task: str, constraints: Dict,
                                  max_steps: int) -> List[PlanStep]:
        """Generate a plan using templates."""
        steps = []
        task_lower = task.lower()

        # Analyze task keywords to determine plan structure
        step_id = 0

        # Always start with understanding
        steps.append(PlanStep(
            step_id=f"step_{step_id}",
            description=f"Analyze the task: {task[:100]}",
            action_type='think',
            estimated_cost=self.ACTION_COSTS['think']
        ))
        step_id += 1

        # If task involves files/commands
        if any(kw in task_lower for kw in ['file', 'find', 'search', 'list', 'run', 'execute']):
            steps.append(PlanStep(
                step_id=f"step_{step_id}",
                description="Execute command to gather information",
                action_type='execute',
                target='tool',
                depends_on=[f"step_{step_id-1}"],
                estimated_cost=self.ACTION_COSTS['execute']
            ))
            step_id += 1

        # If task involves memory
        if any(kw in task_lower for kw in ['remember', 'recall', 'store', 'save']):
            if 'recall' in task_lower or 'remember' in task_lower:
                steps.append(PlanStep(
                    step_id=f"step_{step_id}",
                    description="Recall relevant information from memory",
                    action_type='recall',
                    target='memory',
                    depends_on=[f"step_{step_id-1}"] if step_id > 0 else [],
                    estimated_cost=self.ACTION_COSTS['recall']
                ))
            else:
                steps.append(PlanStep(
                    step_id=f"step_{step_id}",
                    description="Store information in memory",
                    action_type='store',
                    target='memory',
                    depends_on=[f"step_{step_id-1}"] if step_id > 0 else [],
                    estimated_cost=self.ACTION_COSTS['store']
                ))
            step_id += 1

        # If task requires reasoning
        if any(kw in task_lower for kw in ['why', 'how', 'explain', 'analyze', 'reason']):
            steps.append(PlanStep(
                step_id=f"step_{step_id}",
                description="Apply chain-of-thought reasoning",
                action_type='reason',
                target='reasoner',
                depends_on=[f"step_{step_id-1}"] if step_id > 0 else [],
                estimated_cost=self.ACTION_COSTS['reason']
            ))
            step_id += 1

        # Always end with synthesis
        steps.append(PlanStep(
            step_id=f"step_{step_id}",
            description="Synthesize results and formulate response",
            action_type='think',
            depends_on=[f"step_{step_id-1}"] if step_id > 0 else [],
            estimated_cost=self.ACTION_COSTS['think']
        ))

        return steps[:max_steps]

    def _generate_plan_external(self, task: str, constraints: Dict,
                                  max_steps: int) -> List[PlanStep]:
        """Generate a plan using external function (e.g., LLM)."""
        if not self.planning_fn:
            return self._generate_plan_template(task, constraints, max_steps)

        prompt = f"""Task: {task}

Available capabilities:
- think: Generate thoughts/responses (LLM)
- execute: Run shell commands (Tool creature)
- store: Save to memory (Memory creature)
- recall: Retrieve from memory (Memory creature)
- reason: Chain-of-thought analysis (Reasoner creature)

Constraints: {constraints}

Generate a plan as a JSON array of steps. Each step should have:
- id: step_0, step_1, etc.
- description: What this step does
- action: One of think/execute/store/recall/reason
- target: Which creature handles this (optional)
- depends_on: List of step IDs this depends on

Maximum {max_steps} steps. Output JSON array only:"""

        try:
            response = self.planning_fn(prompt)

            # Parse JSON response
            import json
            steps_data = json.loads(response)

            steps = []
            for s in steps_data[:max_steps]:
                steps.append(PlanStep(
                    step_id=s.get('id', f'step_{len(steps)}'),
                    description=s.get('description', ''),
                    action_type=s.get('action', 'think'),
                    target=s.get('target', ''),
                    depends_on=s.get('depends_on', []),
                    estimated_cost=self.ACTION_COSTS.get(s.get('action', 'think'), 0.5)
                ))
            return steps

        except Exception:
            return self._generate_plan_template(task, constraints, max_steps)

    def _build_dependencies(self, steps: List[PlanStep]) -> Dict[str, List[str]]:
        """Build dependency graph from steps."""
        deps = {}
        for step in steps:
            deps[step.step_id] = step.depends_on
        return deps

    def process_messages(self):
        """Process incoming messages from all bonded creatures."""
        for peer_cid in list(self.bonds.keys()):
            messages = self.receive_all_from(peer_cid)

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                if msg_type == "plan":
                    task = msg.get("task", "")
                    constraints = msg.get("constraints", {})

                    plan = self.plan(task, constraints)

                    # Send result back
                    self.send_to(peer_cid, {
                        "type": "steps",
                        "task": plan.task,
                        "plan": [s.to_dict() for s in plan.steps],
                        "dependencies": plan.dependencies,
                        "step_count": len(plan.steps),
                        "total_cost": plan.total_cost
                    })

    def get_stats(self) -> Dict[str, Any]:
        """Get planner creature statistics."""
        base = self.get_state_dict()
        base.update({
            "total_planned": self.total_planned,
            "total_steps": self.total_steps,
            "total_cost": round(self.total_cost, 2),
            "is_planning": self.is_planning,
        })
        return base


def spawn_planner_creature(runtime: ExistenceKernelRuntime,
                           name: str = "planner",
                           initial_f: float = 35.0,
                           initial_a: float = 0.65,
                           planning_fn: Optional[Callable] = None) -> PlannerCreature:
    """
    Spawn a new planner creature.
    Returns the PlannerCreature wrapper.
    """
    cid = runtime.spawn(name, initial_f=initial_f, initial_a=initial_a)
    runtime.creatures[cid].state = CreatureState.RUNNING
    return PlannerCreature(runtime, cid, planning_fn)
