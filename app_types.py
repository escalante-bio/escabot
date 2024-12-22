from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from opentrons.hardware_control import HardwareControlAPI
from opentrons.types import DeckSlotName
from opentrons.protocol_engine.actions import ActionDispatcher
from opentrons.protocol_engine.commands import LoadPipetteResult
from opentrons.protocol_engine.types import (
    AddressableAreaLocation,
    DeckSlotLocation,
    ModuleLocation,
    OnLabwareLocation,
)
from opentrons.protocol_engine.execution import (
    CommandExecutor,
    EquipmentHandler,
    LabwareMovementHandler,
    LoadedLabwareData,
    LoadedModuleData,
    MovementHandler,
    PipettingHandler,
    RunControlHandler,
)
from opentrons.protocol_engine.execution.equipment import LoadedLabwareData
from opentrons.protocol_engine.execution.rail_lights import RailLightsHandler
from opentrons.protocol_engine.execution.status_bar import StatusBarHandler
from opentrons.protocol_engine.execution.tip_handler import TipHandler
from opentrons.protocol_engine.state.state import StateStore

from app_requests import InstructionRequest


class ExecutionState(Enum):
    QUEUED = 1
    EXECUTING = 2
    SUCCEEDED = 3
    FAILED = 4


@dataclass(kw_only=True)
class Instruction:
    id: str
    state: ExecutionState
    task: asyncio.Task
    execution_barrier: asyncio.Lock
    raw: InstructionRequest


@dataclass(kw_only=True)
class Stream:
    id: str
    active: bool
    failed: bool
    instructions: dict[str, Instruction]
    queue: list[Instruction]
    worker: asyncio.Task | None


@dataclass(frozen=True, kw_only=True)
class Robot:
    executor: CommandExecutor
    action_dispatcher: ActionDispatcher
    equipment: EquipmentHandler
    hardware: RobotHardware
    movement: MovementHandler
    labware_movement: LabwareMovementHandler
    pipetting: PipettingHandler
    tip: TipHandler
    rail_lights: RailLightsHandler
    state_store: StateStore
    status_bar: StatusBarHandler


@dataclass(frozen=True, kw_only=True)
class RobotDeckModule:
    model: str
    location: ModuleLocation


@dataclass(kw_only=True)
class RobotDeckSlot:

    @staticmethod
    def create(location: AddressableAreaLocation | DeckSlotLocation) -> RobotDeckSlot:
        return RobotDeckSlot(
            location=location,
            lock=asyncio.Lock(),
            module=None,
            stack=[location],
        )

    location: AddressableAreaLocation | DeckSlotLocation
    lock: asyncio.Lock
    module: RobotDeckModule | None
    stack: list[AddressableAreaLocation | DeckSlotLocation | ModuleLocation | OnLabwareLocation]

    def is_on_deck(self):
        return isinstance(self.location, DeckSlotLocation)

    def top(self):
        return self.stack[-1]

    def top_is_plate(self):
        return isinstance(self.stack[-1], OnLabwareLocation)


@dataclass(frozen=True, kw_only=True)
class RobotHardware:
    hardware_api: HardwareControlAPI
    simulate_hardware: bool


@dataclass(kw_only=True)
class RobotPipette:
    pipette: LoadPipetteResult
    channels: int
    has_tip: bool
    max_volume_nl: float


@dataclass(frozen=True, kw_only=True)
class RobotTip:
    labware: OnLabwareLocation
    well: str


@dataclass(kw_only=True)
class Run:

    @staticmethod
    def create(id: str, robot: Robot) -> Run:
        deck = {}
        for slot in [
            "A1",
            "A2",
            "A3",
            "B1",
            "B2",
            "B3",
            "C1",
            "C2",
            "C3",
            "D1",
            "D2",
            "D3",
        ]:
            deck[slot] = RobotDeckSlot.create(
                DeckSlotLocation(slotName=DeckSlotName.from_primitive(slot))
            )
        for slot in ["A4", "B4", "C4", "D4"]:
            deck[slot] = RobotDeckSlot.create(AddressableAreaLocation(addressableAreaName=slot))
        return Run(
            id=id,
            active=True,
            cleanup_event=asyncio.Event(),
            cleanup_task=None,  # type: ignore
            closed=False,
            deck=deck,
            failed=False,
            initialized=False,
            gantry_lock=asyncio.Lock(),
            pipettes={},
            robot=robot,
            streams={},
        )

    id: str
    active: bool
    cleanup_event: asyncio.Event
    cleanup_task: asyncio.Task
    closed: bool
    deck: dict[str, RobotDeckSlot]
    failed: bool
    # Also used while the thermocycler lid is moving
    gantry_lock: asyncio.Lock
    initialized: bool
    pipettes: dict[str, RobotPipette]
    robot: Robot
    streams: dict[str, Stream]
