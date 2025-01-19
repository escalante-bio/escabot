import asyncio
import logging
from dataclasses import dataclass
import os
import types
from typing import Literal, Never, Optional, cast

from fastapi import HTTPException
from opentrons.types import DeckSlotName, Mount, MountType
from opentrons.hardware_control import API as HardwareAPI, HardwareControlAPI
from opentrons.hardware_control.ot3api import OT3API
from opentrons.hardware_control.types import DoorState, HardwareFeatureFlags, StatusBarState
from opentrons.protocol_api.core.labware import LabwareLoadParams
from opentrons.protocol_engine import (
    AddressableOffsetVector,
    AllNozzleLayoutConfiguration,
    Command,
    CommandCreate,
    CommandIntent,
    CommandStatus,
    Config,
    DeckType,
    DropTipWellLocation,
    SingleNozzleLayoutConfiguration,
)
from opentrons.protocol_engine.actions import (
    ActionDispatcher,
    AddAddressableAreaAction,
    AddLabwareDefinitionAction,
    QueueCommandAction,
    RunCommandAction,
    SetDeckConfigurationAction,
    SucceedCommandAction,
)
from opentrons.protocol_engine.commands import (
    AspirateCreate,
    AspirateParams,
    AspirateResult,
    CommandDefinedErrorData,
    ConfigureNozzleLayoutCreate,
    ConfigureNozzleLayoutParams,
    ConfigureNozzleLayoutResult,
    DispenseCreate,
    DispenseParams,
    DispenseResult,
    DropTipCreate,
    DropTipParams,
    DropTipResult,
    DropTipInPlaceCreate,
    DropTipInPlaceParams,
    DropTipInPlaceResult,
    HomeCreate,
    HomeParams,
    HomeResult,
    LiquidProbeCreate,
    LiquidProbeParams,
    LiquidProbeResult,
    LoadLabwareCreate,
    LoadLabwareParams,
    LoadLabwareResult,
    LoadModuleCreate,
    LoadModuleParams,
    LoadModuleResult,
    LoadPipetteCreate,
    LoadPipetteParams,
    LoadPipetteResult,
    MoveLabwareCreate,
    MoveLabwareParams,
    MoveLabwareResult,
    MoveRelativeCreate,
    MoveRelativeParams,
    MoveRelativeResult,
    MoveToAddressableAreaForDropTipCreate,
    MoveToAddressableAreaForDropTipParams,
    MoveToAddressableAreaForDropTipResult,
    MoveToWellCreate,
    MoveToWellParams,
    MoveToWellResult,
    PickUpTipCreate,
    PickUpTipParams,
    PickUpTipResult,
)
from opentrons.protocol_engine.commands.temperature_module import (
    DeactivateTemperatureCreate,
    DeactivateTemperatureParams,
    DeactivateTemperatureResult,
    SetTargetTemperatureCreate,
    SetTargetTemperatureParams,
    SetTargetTemperatureResult,
    WaitForTemperatureCreate,
    WaitForTemperatureParams,
    WaitForTemperatureResult,
)
from opentrons.protocol_engine.commands.thermocycler import (
    CloseLidCreate,
    CloseLidParams,
    CloseLidResult,
    DeactivateBlockCreate,
    DeactivateBlockParams,
    DeactivateBlockResult,
    DeactivateLidCreate,
    DeactivateLidParams,
    DeactivateLidResult,
    OpenLidCreate,
    OpenLidParams,
    OpenLidResult,
    SetTargetBlockTemperatureCreate,
    SetTargetBlockTemperatureParams,
    SetTargetBlockTemperatureResult,
    SetTargetLidTemperatureCreate,
    SetTargetLidTemperatureParams,
    SetTargetLidTemperatureResult,
    WaitForBlockTemperatureCreate,
    WaitForBlockTemperatureParams,
    WaitForBlockTemperatureResult,
    WaitForLidTemperatureCreate,
    WaitForLidTemperatureParams,
    WaitForLidTemperatureResult,
)
from opentrons.protocol_engine.error_recovery_policy import (
    ErrorRecoveryPolicy,
    ErrorRecoveryType,
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
from opentrons.protocol_engine.execution.gantry_mover import create_gantry_mover
from opentrons.protocol_engine.execution.pipetting import create_pipetting_handler
from opentrons.protocol_engine.execution.rail_lights import RailLightsHandler
from opentrons.protocol_engine.execution.status_bar import StatusBarHandler
from opentrons.protocol_engine.execution.tip_handler import TipHandler, create_tip_handler
from opentrons.protocol_engine.resources import (
    DeckDataProvider,
    FileProvider,
    ModelUtils,
    ModuleDataProvider,
)
from opentrons.protocol_engine.state.command_history import CommandEntry
from opentrons.protocol_engine.state.state import StateStore
from opentrons.protocol_engine import (
    DropTipWellOrigin,
    WellLocation,
    WellOffset,
    WellOrigin,
)
from opentrons.protocol_engine.types import (
    AddressableAreaLocation,
    DeckSlotLocation,
    LabwareMovementOffsetData,
    LabwareMovementStrategy,
    LabwareOffsetVector,
    ModuleLocation,
    ModuleModel,
    MovementAxis,
    OnLabwareLocation,
    LiquidHandlingWellLocation,
)
from opentrons.protocols.api_support.deck_type import (
    guess_from_global_config as guess_deck_type_from_global_config,
)
from opentrons.protocols.api_support.definitions import MAX_SUPPORTED_VERSION
from opentrons.protocols.api_support.util import find_value_for_api_version
from opentrons.protocols.models import LabwareDefinition
from opentrons.util import entrypoint_util
from opentrons_shared_data.labware.types import LabwareUri
from opentrons_shared_data.pipette.types import PipetteNameType
from opentrons_shared_data.robot import load as load_robot

from app_requests import (
    AddLabwareDefinitionRequest,
    AspirateRequest,
    DispenseRequest,
    DropTipRequest,
    HomeRequest,
    InitializeRequest,
    InstructionRequest,
    LoadModuleRequest,
    LoadLabwareRequest,
    MoveLabwareRequest,
    MoveToWellRequest,
    PickUpTipRequest,
    TemperatureBlockTemperatureRequest,
    ThermocycleBlockDeactivateRequest,
    ThermocycleBlockTemperatureRequest,
    ThermocycleLidDeactivateRequest,
    ThermocycleLidHingeRequest,
    ThermocycleLidTemperatureRequest,
    WaitRequest,
)
from app_types import (
    Robot,
    RobotDeckModule,
    RobotDeckSlot,
    RobotHardware,
    RobotPipette,
    Run,
    TipSource,
)


# See the default aspirate and dispense offsets at
# https://github.com/Opentrons/opentrons/blob/aadc65ec79ebbf66acd9df08dcfb7910d34cdfb9/api/src/opentrons/protocol_api/instrument_context.py#L44
# See the mapping at
# https://github.com/Opentrons/opentrons/blob/aadc65ec79ebbf66acd9df08dcfb7910d34cdfb9/api/src/opentrons/protocol_api/validation.py#L63
PIPETTE_1_CHANNEL = PipetteNameType.P50_SINGLE_FLEX
PIPETTE_8_CHANNEL = PipetteNameType.P50_MULTI_FLEX
WASTE_CHUTE_AREA = "96ChannelWasteChute"  # originally was 1Channel. Was that important?

LABWARE_OFFSETS = {
}


async def create_hardware_control(simulate_hardware: bool) -> HardwareControlAPI:
    if simulate_hardware:
        return await HardwareAPI.build_hardware_simulator(
            attached_instruments={
                Mount.LEFT: {"id": "p1", "model": "p50_single_v3.6"},
                Mount.RIGHT: {"id": "p8", "model": "p50_multi_v3.5"},
            },
            feature_flags=HardwareFeatureFlags.build_from_ff(),
        )
    else:
        from opentrons.config import feature_flags as ff

        return await OT3API.build_hardware_controller(
            use_usb_bus=ff.rear_panel_integration(),
            status_bar_enabled=ff.status_bar_enabled(),
            feature_flags=HardwareFeatureFlags.build_from_ff(),
        )


def create_error_recovery_policy() -> ErrorRecoveryPolicy:
    def _policy(
        config: Config,
        failed_command: Command,
        defined_error_data: Optional[CommandDefinedErrorData],
    ) -> ErrorRecoveryType:
        return ErrorRecoveryType.FAIL_RUN

    return _policy


async def create_state_store(hardware_api: HardwareControlAPI, config: Config) -> StateStore:
    deck_data = DeckDataProvider(config.deck_type)
    deck_definition = await deck_data.get_deck_definition()
    deck_fixed_labware = await deck_data.get_deck_fixed_labware(
        load_fixed_trash=False,
        deck_definition=deck_definition,
        deck_configuration=None,
    )

    module_calibration_offsets = ModuleDataProvider.load_module_calibrations()
    robot_definition = load_robot(config.robot_type)
    state_store = StateStore(
        config=config,
        deck_definition=deck_definition,
        deck_fixed_labware=deck_fixed_labware,
        robot_definition=robot_definition,
        is_door_open=hardware_api.door_state is DoorState.OPEN,
        error_recovery_policy=create_error_recovery_policy(),
        module_calibration_offsets=module_calibration_offsets,
        deck_configuration=None,
        notify_publishers=None,
    )

    # Note to anyone who works at Opentrons: you hereby agree to not look at the following code and
    # to keep scrolling.
    #
    # * This place is a message... and part of a system of messages... pay attention to it!
    # * Sending this message was important to us. We considered ourselves to be a powerful culture.
    # * This place is not a place of honor... no highly esteemed deed is commemorated here...
    # * nothing valued is here.
    # * What is here was dangerous and repulsive to us. This message is a warning about danger.
    # * The danger is in a particular location... it increases towards a center... the center of
    #   danger is here... of a particular size and shape, and below us.
    # * The danger is still present, in your time, as it was in ours.

    command_history = state_store.commands._state.command_history

    # Allow commands to run concurrently
    def set_command_running_concurrent(command: Command) -> None:
        prev_entry = command_history.get(command.id)
        assert prev_entry.command.status == CommandStatus.QUEUED
        assert command.status == CommandStatus.RUNNING
        command_history._add(
            command.id,
            CommandEntry(
                index=prev_entry.index,
                command=command,
            ),
        )

    def set_command_succeeded_concurrent(command: Command) -> None:
        prev_entry = command_history.get(command.id)
        assert prev_entry.command.status == CommandStatus.RUNNING
        assert command.status == CommandStatus.SUCCEEDED
        command_history._add(
            command.id,
            CommandEntry(
                index=prev_entry.index,
                command=command,
            ),
        )

    def set_command_failed_concurrent(command: Command) -> None:
        prev_entry = command_history.get(command.id)
        assert prev_entry.command.status in (
            CommandStatus.RUNNING,
            CommandStatus.SUCCEEDED,
            CommandStatus.FAILED,
        )
        assert command.status == CommandStatus.FAILED
        command_history._add(
            command.id,
            CommandEntry(
                index=prev_entry.index,
                command=command,
            ),
        )

    command_history.set_command_running = set_command_running_concurrent
    command_history.set_command_succeeded = set_command_succeeded_concurrent
    command_history.set_command_failed = set_command_failed_concurrent

    geometry = state_store.geometry

    # Allow reracking multichannel tips while in partial tip configuration
    def get_unchecked_tip_drop_location(
        pipette_id: str,
        labware_id: str,
        well_location: DropTipWellLocation,
        partially_configured: bool = False,
    ) -> WellLocation:
        if well_location.origin != DropTipWellOrigin.DEFAULT:
            return WellLocation(
                origin=WellOrigin(well_location.origin.value),
                offset=well_location.offset,
            )

        if geometry._labware.get_definition(labware_id).parameters.isTiprack:
            z_offset = geometry._labware.get_tip_drop_z_offset(
                labware_id=labware_id,
                length_scale=geometry._pipettes.get_return_tip_scale(pipette_id),
                additional_offset=well_location.offset.z,
            )
        else:
            # return to top if labware is not tip rack
            z_offset = well_location.offset.z

        return WellLocation(
            origin=WellOrigin.TOP,
            offset=WellOffset(
                x=well_location.offset.x,
                y=well_location.offset.y,
                z=z_offset,
            ),
        )

    geometry.get_checked_tip_drop_location = get_unchecked_tip_drop_location

    # Opentrons people can resume looking past this point

    return state_store


async def create_opentrons_hardware(simulate_hardware: bool) -> RobotHardware:
    return RobotHardware(
        hardware_api=await create_hardware_control(simulate_hardware),
        simulate_hardware=simulate_hardware,
    )


async def create_opentrons_state(hardware: RobotHardware) -> Robot:
    config = Config(
        robot_type="OT-3 Standard",
        deck_type=DeckType.OT3_STANDARD,
        # We deliberately omit ignore_pause=True because, in the current implementation of
        # opentrons.protocol_api.core.engine, that would incorrectly make
        # ProtocolContext.is_simulating() return True.
        use_simulated_deck_config=hardware.simulate_hardware,
        use_virtual_pipettes=hardware.simulate_hardware,
        use_virtual_gripper=hardware.simulate_hardware,
        use_virtual_modules=hardware.simulate_hardware,
    )
    file_provider = FileProvider()
    state_store = await create_state_store(hardware.hardware_api, config)

    action_dispatcher = ActionDispatcher(sink=state_store)

    gantry_mover = create_gantry_mover(
        hardware_api=hardware.hardware_api,
        state_view=state_store,
    )

    equipment = EquipmentHandler(
        hardware_api=hardware.hardware_api,
        state_store=state_store,
    )

    movement = MovementHandler(
        hardware_api=hardware.hardware_api,
        state_store=state_store,
        gantry_mover=gantry_mover,
    )

    labware_movement = LabwareMovementHandler(
        hardware_api=hardware.hardware_api,
        state_store=state_store,
        equipment=equipment,
        movement=movement,
    )

    pipetting = create_pipetting_handler(
        hardware_api=hardware.hardware_api,
        state_view=state_store,
    )

    tip = create_tip_handler(
        hardware_api=hardware.hardware_api,
        state_view=state_store,
    )

    run_control = RunControlHandler(
        state_store=state_store,
        action_dispatcher=action_dispatcher,
    )

    rail_lights = RailLightsHandler(
        hardware_api=hardware.hardware_api,
    )

    status_bar = StatusBarHandler(hardware_api=hardware.hardware_api)

    return Robot(
        executor=CommandExecutor(
            action_dispatcher=action_dispatcher,
            hardware_api=hardware.hardware_api,
            equipment=equipment,
            movement=movement,
            file_provider=file_provider,
            gantry_mover=gantry_mover,
            labware_movement=labware_movement,
            pipetting=pipetting,
            tip_handler=tip,
            rail_lights=rail_lights,
            run_control=run_control,
            state_store=state_store,
            status_bar=status_bar,
        ),
        action_dispatcher=action_dispatcher,
        hardware=hardware,
        equipment=equipment,
        movement=movement,
        labware_movement=labware_movement,
        pipetting=pipetting,
        tip=tip,
        rail_lights=rail_lights,
        state_store=state_store,
        status_bar=status_bar,
    )


async def initialize_robot(run: Run):
    robot = run.robot
    await set_status_bar(StatusBarState.RUNNING, robot)
    await asyncio.get_running_loop().run_in_executor(None, lambda: (
        robot.action_dispatcher.dispatch(
            SetDeckConfigurationAction(entrypoint_util.get_deck_configuration())
        )
    ))
    await home(run, robot)
    await load_waste_chute(run, robot)
    await load_pipette(PIPETTE_1_CHANNEL, 1, 50000, MountType.LEFT, run, robot)
    await load_pipette(PIPETTE_8_CHANNEL, 8, 50000, MountType.RIGHT, run, robot)


async def execute_instruction(instruction: InstructionRequest, run: Run):
    robot = run.robot
    # Locking order:
    # * first the deck slots in alphabetical order
    # * then the gantry

    logging.warning("Executing instruction %s", instruction)
    if isinstance(instruction, InitializeRequest):
        if run.initialized:
            raise Exception(f"Run {run.id} has already been initialized")

        # We don't need to lock here because run.initialized basically is a lock
        await initialize_robot(run)
        run.initialized = True
        return
    elif not run.initialized:
        raise Exception(f"Run {run.id} hasn't been initialized yet")

    if isinstance(instruction, AddLabwareDefinitionRequest):
        add_labware_definition(instruction.definition, robot)
    elif isinstance(instruction, DropTipRequest):
        await run.gantry_lock.acquire()
        try:
            pipette = await get_pipette(instruction.channels, run, robot)
            if instruction.well:
                slot = run.deck[instruction.well.bay]
                well = instruction.well.well
                if not slot:
                    raise HTTPException(400, f"Bay {instruction.well.bay} is invalid")
            else:
                slot = None
                well = None
            await drop_tip(slot, well, pipette, robot, validate=False)
        finally:
            run.gantry_lock.release()
    elif isinstance(instruction, AspirateRequest):
        volume_nl = instruction.volume_nl
        if not volume_nl:
            raise HTTPException(400, f"volume_nl is {volume_nl}")

        bay = instruction.at.bay
        slot = run.deck.get(bay)
        if not slot:
            raise HTTPException(400, f"Bay {bay} is invalid")
        await slot.lock.acquire()

        if not slot.top_is_plate():
            raise HTTPException(400, f"Bay {bay} doesn't have a plate on top")

        await run.gantry_lock.acquire()
        pipette = await get_pipette(instruction.channels, run, robot)
        if volume_nl > pipette.max_volume_nl:
            raise HTTPException(
                400,
                (
                    f"Requested volume {volume_nl}nl exceeds pipette's limit of "
                    f"{pipette.max_volume_nl}nl"
                ),
            )

        await aspirate(
            slot,
            instruction.at.well,
            volume_nl,
            instruction.z_mm,
            pipette,
            run,
            robot,
        )

        run.gantry_lock.release()
        slot.lock.release()
    elif isinstance(instruction, DispenseRequest):
        volume_nl = instruction.volume_nl
        if not volume_nl:
            raise HTTPException(400, f"volume_nl is {volume_nl}")

        bay = instruction.at.bay
        slot = run.deck.get(bay)
        if not slot:
            raise HTTPException(400, f"Bay {bay} is invalid")
        await slot.lock.acquire()

        if not slot.top_is_plate():
            raise HTTPException(400, f"Bay {bay} doesn't have a plate on top")

        await run.gantry_lock.acquire()
        pipette = await get_pipette(instruction.channels, run, robot)
        if volume_nl > pipette.max_volume_nl:
            raise HTTPException(
                400,
                (
                    f"Requested volume {volume_nl}nl exceeds pipette's limit of "
                    f"{pipette.max_volume_nl}nl"
                ),
            )
        await dispense(slot, instruction.at.well, volume_nl, instruction.z_mm, pipette, run, robot)
        run.gantry_lock.release()
        slot.lock.release()
    elif isinstance(instruction, HomeRequest):
        await run.gantry_lock.acquire()
        await home(run, robot)
        run.gantry_lock.release()
    elif isinstance(instruction, LoadLabwareRequest):
        bay = instruction.into
        existing = run.deck[bay]
        if not existing:
            raise HTTPException(400, f"Bay {bay} is invalid")
        await existing.lock.acquire()
        # This is hard to check because the adapter to the temperature module is also labware...
        # Whatever.
        # if existing.top_is_plate():
        #    raise HTTPException(409, f"Bay {bay} already contains something")
        await load_labware(instruction.model, bay, run, robot)
        existing.lock.release()
    elif isinstance(instruction, LoadModuleRequest):
        bay = instruction.into
        existing = run.deck[bay]
        if not existing:
            raise HTTPException(400, f"Bay {bay} is invalid")
        await existing.lock.acquire()

        if instruction.module == "magnetic":
            await load_module(ModuleModel.MAGNETIC_BLOCK_V1, instruction.into, run, robot)
        elif instruction.module == "temperature":
            await load_module(ModuleModel.TEMPERATURE_MODULE_V2, instruction.into, run, robot)
        elif instruction.module == "thermocycling":
            await load_module(ModuleModel.THERMOCYCLER_MODULE_V2, instruction.into, run, robot)
        existing.lock.release()
    elif isinstance(instruction, MoveLabwareRequest):
        from_bay = instruction.from_
        from_slot = run.deck[from_bay]
        if not from_slot:
            raise HTTPException(400, f"Bay {from_bay} is invalid")
        to_bay = instruction.to
        to_slot = run.deck[to_bay]
        if not to_slot:
            raise HTTPException(400, f"Bay {to_bay} is invalid")

        deck_locks = order_locks([from_bay, to_bay], run)
        for lock in deck_locks:
            await lock.acquire()

        if not from_slot.top_is_plate():
            raise HTTPException(409, f"Bay {from_bay} doesn't have a plate on top")
        # This is hard to check because we place lids on top of plates
        # Whatever.
        # if to_slot.top_is_plate():
        #     raise HTTPException(409, f"Bay {to_bay} already has a plate on top")

        await run.gantry_lock.acquire()
        await move_labware(from_slot, to_slot, run, robot)
        run.gantry_lock.release()
        for lock in reversed(deck_locks):
            lock.release()
    elif isinstance(instruction, MoveToWellRequest):
        to_bay = instruction.to.bay
        to_slot = run.deck[to_bay]
        if not to_slot:
            raise HTTPException(400, f"Bay {to_bay} is invalid")

        await to_slot.lock.acquire()
        if to_slot.top_is_plate():
            raise HTTPException(409, f"Bay {to_bay} already has a plate on top")

        await run.gantry_lock.acquire()
        pipette = await get_pipette(instruction.channels, run, robot)
        await move_to_well(
            to_slot,
            instruction.to.well,
            pipette,
            robot,
            WellLocation(origin=WellOrigin.BOTTOM, offset=instruction.offset),
        )
        run.gantry_lock.release()
        to_slot.lock.release()
    elif isinstance(instruction, PickUpTipRequest):
        await run.gantry_lock.acquire()
        pipette = await get_pipette(instruction.channels, run, robot)
        slot = run.deck[instruction.well.bay]
        if not slot:
            raise HTTPException(400, f"Bay {instruction.well.bay} is invalid")
        await pick_up_tip(slot, instruction.well.well, pipette, robot)
        run.gantry_lock.release()
    elif isinstance(instruction, TemperatureBlockTemperatureRequest):
        at_bay = instruction.at
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")
        await at_slot.lock.acquire()
        await set_temperature_block(at_slot, instruction.temperature_c, run, robot)
        at_slot.lock.release()
    elif isinstance(instruction, ThermocycleBlockDeactivateRequest):
        at_bay = instruction.at
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")
        await at_slot.lock.acquire()
        await deactivate_thermocycler_block(
            at_slot=at_slot,
            run=run,
            robot=robot,
        )
        at_slot.lock.release()
    elif isinstance(instruction, ThermocycleBlockTemperatureRequest):
        at_bay = instruction.at
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")
        await at_slot.lock.acquire()
        await set_thermocycler_block(
            at_slot=at_slot,
            duration_us=instruction.duration_us,
            max_volume_nl=instruction.max_volume_nl,
            temperature_c=instruction.temperature_c,
            run=run,
            robot=robot,
        )
        at_slot.lock.release()
    elif isinstance(instruction, ThermocycleLidDeactivateRequest):
        at_bay = instruction.at
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")

        await at_slot.lock.acquire()
        await deactivate_thermocycler_lid(
            at_slot=at_slot,
            run=run,
            robot=robot,
        )
        at_slot.lock.release()
    elif isinstance(instruction, ThermocycleLidHingeRequest):
        at_bay = instruction.at
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")
        await at_slot.lock.acquire()
        await run.gantry_lock.acquire()
        await set_thermocycler_lid_hinge(
            at_slot=at_slot,
            closed=instruction.closed,
            run=run,
            robot=robot,
        )
        run.gantry_lock.release()
        at_slot.lock.release()
    elif isinstance(instruction, ThermocycleLidTemperatureRequest):
        at_bay = instruction.at
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")

        await at_slot.lock.acquire()
        await set_thermocycler_lid_temperature(
            at_slot=at_slot,
            temperature_c=instruction.temperature_c,
            run=run,
            robot=robot,
        )
        at_slot.lock.release()
    elif isinstance(instruction, WaitRequest):
        skip_waiting = os.environ.get("SKIP_WAITING", "false").lower() == "true"
        if not skip_waiting:
            await asyncio.sleep(instruction.duration_us / 1000 / 1000)
    else:
        # By typing this with Never we get a static exhaustive check too
        def assert_unexpected(instruction: Never) -> Never:
            raise HTTPException(400, f"Unknown command: {instruction}")

        assert_unexpected(instruction)
    logging.warning("Finished executing instruction %s", instruction)


async def get_pipette(
    channels: int,
    run: Run,
    robot: Robot,
) -> RobotPipette:
    pipette = run.pipettes[PIPETTE_1_CHANNEL if channels <= 1 else PIPETTE_8_CHANNEL]
    if pipette.channels == channels:
        return pipette

    request = ConfigureNozzleLayoutCreate(
        params=ConfigureNozzleLayoutParams(
            pipetteId=pipette.pipette.pipetteId,
            configurationParams=(
                AllNozzleLayoutConfiguration()
                if channels in [1, 8]
                else SingleNozzleLayoutConfiguration(primaryNozzle="A1")
            ),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    cast(ConfigureNozzleLayoutResult, await execute(request, robot))
    pipette.channels = channels
    return pipette


def add_labware_definition(definition: dict, robot: Robot) -> None:
    robot.state_store.handle_action(
        AddLabwareDefinitionAction(definition=LabwareDefinition.parse_obj(definition)))


async def aspirate(
    slot: RobotDeckSlot,
    well: str,
    volume_nl: int | float,
    z_mm: float,
    pipette: RobotPipette,
    run: Run,
    robot: Robot,
) -> MoveRelativeResult:
    if not pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} doesn't have a tip")
    pipetteId = pipette.pipette.pipetteId

    top = slot.stack[-1]
    if not isinstance(top, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    if isinstance(slot.location, DeckSlotLocation):
        offset = LABWARE_OFFSETS.get(slot.location.slotName.value) or WellOffset()
    else:
        offset = WellOffset()
    wellLocation = LiquidHandlingWellLocation(
        origin=WellOrigin.BOTTOM,
        offset=WellOffset(x=offset.x, y=offset.y, z=offset.z + z_mm),
    )
    aspirate = AspirateCreate(
        params=AspirateParams(
            labwareId=top.labwareId,
            pipetteId=pipetteId,
            volume=volume_nl / 1000,
            wellName=well,
            wellLocation=wellLocation,
            flowRate=find_value_for_api_version(
                MAX_SUPPORTED_VERSION,
                robot.state_store.pipettes.get_flow_rates(pipetteId).default_aspirate,
            ),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    cast(AspirateResult, await execute(aspirate, robot))

    # If we offset the aspirate and dispense to handle stupid OT bugs then OT may calculate that
    # it's clear of the well incorrectly and drag the pipette tip through wells. So just go up more.
    move = MoveRelativeCreate(
        params=MoveRelativeParams(
            pipetteId=pipetteId,
            axis=MovementAxis.Z,
            distance=20,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(MoveRelativeResult, await execute(move, robot))


async def home(run: Run, robot: Robot) -> HomeResult:
    request = HomeCreate(
        params=HomeParams(axes=None, skipIfMountPositionOk=None),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(HomeResult, await execute(request, robot))


async def dispense(
    slot: RobotDeckSlot,
    well: str,
    volume_nl: int | float,
    z_mm: float,
    pipette: RobotPipette,
    run: Run,
    robot: Robot,
) -> None:
    if not pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} doesn't have a tip")
    pipetteId = pipette.pipette.pipetteId

    top = slot.stack[-1]
    if not isinstance(top, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    if isinstance(slot.location, DeckSlotLocation):
        offset = LABWARE_OFFSETS.get(slot.location.slotName.value, WellOffset())
    else:
        offset = WellOffset()
    well_location = LiquidHandlingWellLocation(
        origin=WellOrigin.BOTTOM,
        offset=WellOffset(x=offset.x, y=offset.y, z=offset.z + z_mm),
    )
    dispense = DispenseCreate(
        params=DispenseParams(
            labwareId=top.labwareId,
            pipetteId=pipetteId,
            volume=volume_nl / 1000,
            wellName=well,
            wellLocation=well_location,
            pushOut=None,
            flowRate=find_value_for_api_version(
                MAX_SUPPORTED_VERSION,
                robot.state_store.pipettes.get_flow_rates(pipetteId).default_dispense,
            ),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    cast(DispenseResult, await execute(dispense, robot))

    move_to_well = MoveToWellCreate(
        params=MoveToWellParams(
            labwareId=top.labwareId,
            pipetteId=pipetteId,
            wellName=well,
            wellLocation=WellLocation(
                origin=well_location.origin,
                offset=WellOffset(
                    x=well_location.offset.x,
                    y=well_location.offset.y,
                    z=well_location.offset.z + 50,
                ),
            ),
            forceDirect=False,
            minimumZHeight=None,
            speed=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    cast(MoveToWellResult, await execute(move_to_well, robot))


async def drop_tip(
    slot: RobotDeckSlot | None,
    well: str | None,
    pipette: RobotPipette,
    robot: Robot,
    *,
    validate=True,
) -> None:
    if validate and not pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} doesn't have a tip")

    if slot and well:
        top = slot.stack[-1]
        if not isinstance(top, OnLabwareLocation):
            # We can still try to move the temperature adapter... oops
            raise Exception(f"Expected labware to be on top")

        drop = DropTipCreate(
            params=DropTipParams(
                labwareId=top.labwareId,
                wellName=well,
                # Set this high enough that it hopefully doesn't pick up other tips while dropping
                # these
                wellLocation=DropTipWellLocation(offset=WellOffset(z=20)),
                alternateDropLocation=True,
                pipetteId=pipette.pipette.pipetteId,
                homeAfter=None,
            ),
            intent=CommandIntent.PROTOCOL,
            key=None,
        )
        cast(DropTipResult, await execute(drop, robot))
    else:
        move = MoveToAddressableAreaForDropTipCreate(
            params=MoveToAddressableAreaForDropTipParams(
                addressableAreaName=WASTE_CHUTE_AREA,
                pipetteId=pipette.pipette.pipetteId,
                alternateDropLocation=None,
                forceDirect=False,
                ignoreTipConfiguration=None,
                minimumZHeight=None,
                # Two bugs in OT (2025-01-09):
                # 1) z=0 leads to tips being dragged through the trash chute
                # 2) y=0 leads to crashing into the front of the robot when dropping 4 tips from the
                #    8
                # TODO(april): it's possible this is caused by WASTE_CHUTE_AREA being set for the 96
                # well. Do we even need to call AddAddressableArea?
                offset=AddressableOffsetVector(x=0, y=20, z=20),
                speed=None,
            ),
            intent=CommandIntent.PROTOCOL,
            key=None,
        )
        await execute(move, robot)

        drop = DropTipInPlaceCreate(
            params=DropTipInPlaceParams(
                pipetteId=pipette.pipette.pipetteId,
                homeAfter=None,
            ),
            intent=CommandIntent.PROTOCOL,
            key=None,
        )
        cast(DropTipInPlaceResult, await execute(drop, robot))

    pipette.has_tip = False
    pipette.tip_source = None


async def liquid_probe(
    slot: RobotDeckSlot,
    well: str,
    pipette: RobotPipette,
    run: Run,
    robot: Robot,
) -> LiquidProbeResult:
    top = slot.stack[-1]
    if not isinstance(top, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    if isinstance(slot.location, DeckSlotLocation):
        offset = LABWARE_OFFSETS.get(slot.location.slotName.value) or WellOffset()
    else:
        offset = WellOffset()
    request = LiquidProbeCreate(
        params=LiquidProbeParams(
            pipetteId=pipette.pipette.pipetteId,
            labwareId=top.labwareId,
            wellName=well,
            wellLocation=WellLocation(
                origin=WellOrigin.TOP,
                offset=WellOffset(x=offset.x, y=offset.y, z=0),
            ),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(LiquidProbeResult, await execute(request, robot))


async def load_labware(model: str, location: str, run: Run, robot: Robot) -> OnLabwareLocation:
    actual = run.deck[location].stack[-1]
    details = LabwareLoadParams.from_uri(cast(LabwareUri, model))
    request = LoadLabwareCreate(
        params=LoadLabwareParams(
            location=actual,
            loadName=details.load_name,
            namespace=details.namespace,
            version=details.version,
            labwareId=None,
            displayName=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(LoadLabwareResult, await execute(request, robot))
    now = OnLabwareLocation(labwareId=result.labwareId)
    run.deck[location].stack.append(now)
    return now


async def load_module(
    model: ModuleModel, location: str, run: Run, robot: Robot
) -> LoadModuleResult:
    actual = run.deck[location].stack[-1]
    if not isinstance(actual, DeckSlotLocation):
        raise Exception("Expected deck slot")

    request = LoadModuleCreate(
        params=LoadModuleParams(model=model, location=actual, moduleId=None),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(LoadModuleResult, await execute(request, robot))
    module_location = ModuleLocation(moduleId=result.moduleId)
    slot = run.deck[location]
    slot.module = RobotDeckModule(model=model, location=module_location)
    slot.stack.append(module_location)
    return result


async def load_pipette(
    pipette_name: PipetteNameType,
    channels: int,
    max_volume_nl: float,
    mount: MountType,
    run: Run,
    robot: Robot,
) -> LoadPipetteResult:
    request = LoadPipetteCreate(
        params=LoadPipetteParams(
            pipetteName=pipette_name,
            mount=mount,
            pipetteId=None,
            tipOverlapNotAfterVersion=None,
            liquidPresenceDetection=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(LoadPipetteResult, await execute(request, robot))
    run.pipettes[pipette_name] = RobotPipette(
        pipette=result,
        channels=channels,
        has_tip=False,
        max_volume_nl=max_volume_nl,
        tip_source=None,
    )
    return result


async def load_waste_chute(run: Run, robot: Robot) -> None:
    robot.action_dispatcher.dispatch(
        AddAddressableAreaAction(
            addressable_area=AddressableAreaLocation(addressableAreaName=WASTE_CHUTE_AREA),
        )
    )


async def move_labware(
    before: RobotDeckSlot,
    after: RobotDeckSlot,
    run: Run,
    robot: Robot,
    *,
    pickUpOffset: LabwareOffsetVector | None = None,
    dropOffset: LabwareOffsetVector | None = None,
) -> MoveLabwareResult:
    was = before.stack[-1]
    if not isinstance(was, OnLabwareLocation):
        # This doesn't stop us from trying to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    if pickUpOffset:
        pass
    elif isinstance(before.location, AddressableAreaLocation):
        if before.location.addressableAreaName in ('A4', 'B4', 'C4', 'D4'):
            pickUpOffset = LabwareOffsetVector(x=0, y=0, z=5)
        else:
            pickUpOffset = LabwareOffsetVector(x=0, y=0, z=0)
    else:
        pickUpOffset = LabwareOffsetVector(x=0, y=0, z=0)

    if dropOffset:
        pass
    elif isinstance(after.location, AddressableAreaLocation):
        if after.location.addressableAreaName in ('A4', 'B4', 'C4', 'D4'):
            dropOffset = LabwareOffsetVector(x=0, y=0, z=5)
        else:
            dropOffset = LabwareOffsetVector(x=0, y=0, z=0)
    else:
        dropOffset = LabwareOffsetVector(x=0, y=0, z=0)

    request = MoveLabwareCreate(
        params=MoveLabwareParams(
            labwareId=was.labwareId,
            newLocation=after.stack[-1],
            strategy=LabwareMovementStrategy.USING_GRIPPER,
            pickUpOffset=pickUpOffset,
            dropOffset=dropOffset,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(MoveLabwareResult, await execute(request, robot))
    after.stack.append(before.stack.pop())
    return result


async def move_to_well(
    slot: RobotDeckSlot,
    well: str,
    pipette: RobotPipette,
    robot: Robot,
    wellLocation: WellLocation,
) -> MoveToWellResult:
    pipetteId = pipette.pipette.pipetteId

    top = slot.stack[-1]
    if not isinstance(top, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    request = MoveToWellCreate(
        params=MoveToWellParams(
            labwareId=top.labwareId,
            pipetteId=pipetteId,
            wellName=well,
            wellLocation=wellLocation,
            forceDirect=False,
            minimumZHeight=None,
            speed=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(MoveToWellResult, await execute(request, robot))
    return result


async def pick_up_tip(
    slot: RobotDeckSlot, well: str, pipette: RobotPipette, robot: Robot
) -> PickUpTipResult:
    if pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} already has a tip")

    top = slot.stack[-1]
    if not isinstance(top, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")
    request = PickUpTipCreate(
        params=PickUpTipParams(
            labwareId=top.labwareId,
            wellName=well,
            pipetteId=pipette.pipette.pipetteId,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(PickUpTipResult, await execute(request, robot))
    pipette.has_tip = True
    pipette.tip_source = TipSource(bay=slot, well=well)
    return result


async def set_status_bar(state: StatusBarState, robot: Robot) -> None:
    await robot.status_bar.set_status_bar(state)


async def set_temperature_block(
    at_slot: RobotDeckSlot, temperature_c: int | float, run: Run, robot: Robot
) -> WaitForTemperatureResult:
    module = at_slot.module
    if not module:
        raise HTTPException(400, f"Bay {at_slot.location} doesn't have a module")

    request = SetTargetTemperatureCreate(
        params=SetTargetTemperatureParams(
            moduleId=module.location.moduleId,
            celsius=float(temperature_c),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)

    request = WaitForTemperatureCreate(
        params=WaitForTemperatureParams(
            moduleId=module.location.moduleId,
            celsius=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(WaitForTemperatureResult, await execute(request, robot))


async def deactivate_thermocycler_block(
    *, at_slot: RobotDeckSlot, run: Run, robot: Robot
) -> None:
    module = at_slot.module
    if not module:
        raise HTTPException(400, f"Bay {at_slot.location} doesn't have a module")

    request = DeactivateBlockCreate(
        params=DeactivateBlockParams(moduleId=module.location.moduleId),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)


async def set_thermocycler_block(
    *,
    at_slot: RobotDeckSlot,
    duration_us: Optional[int | float],
    max_volume_nl: int | float,
    temperature_c: int | float,
    run: Run,
    robot: Robot,
) -> WaitForBlockTemperatureResult:
    module = at_slot.module
    if not module:
        raise HTTPException(400, f"Bay {at_slot.location} doesn't have a module")

    request = SetTargetBlockTemperatureCreate(
        params=SetTargetBlockTemperatureParams(
            moduleId=module.location.moduleId,
            celsius=float(temperature_c),
            blockMaxVolumeUl=float(max_volume_nl / 1000),
            holdTimeSeconds=float(duration_us / 1000_000) if duration_us else None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)

    request = WaitForBlockTemperatureCreate(
        params=WaitForBlockTemperatureParams(
            moduleId=module.location.moduleId,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(WaitForBlockTemperatureResult, await execute(request, robot))


async def deactivate_thermocycler_lid(
    *, at_slot: RobotDeckSlot, run: Run, robot: Robot
) -> None:
    module = at_slot.module
    if not module:
        raise HTTPException(400, f"Bay {at_slot.location} doesn't have a module")

    request = DeactivateLidCreate(
        params=DeactivateLidParams(moduleId=module.location.moduleId),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)


async def set_thermocycler_lid_hinge(
    *, at_slot: RobotDeckSlot, closed: bool, run: Run, robot: Robot
) -> None:
    module = at_slot.module
    if not module:
        raise HTTPException(400, f"Bay {at_slot.location} doesn't have a module")

    if closed:
        request = CloseLidCreate(
            params=CloseLidParams(moduleId=module.location.moduleId),
            intent=CommandIntent.PROTOCOL,
            key=None,
        )
    else:
        request = OpenLidCreate(
            params=OpenLidParams(moduleId=module.location.moduleId),
            intent=CommandIntent.PROTOCOL,
            key=None,
        )
    await execute(request, robot)


async def set_thermocycler_lid_temperature(
    *, at_slot: RobotDeckSlot, temperature_c: int | float, run: Run, robot: Robot
) -> WaitForLidTemperatureResult:
    module = at_slot.module
    if not module:
        raise HTTPException(400, f"Bay {at_slot.location} doesn't have a module")

    request = SetTargetLidTemperatureCreate(
        params=SetTargetLidTemperatureParams(
            moduleId=module.location.moduleId,
            celsius=float(temperature_c),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)

    request = WaitForLidTemperatureCreate(
        params=WaitForLidTemperatureParams(
            moduleId=module.location.moduleId,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(WaitForLidTemperatureResult, await execute(request, robot))


async def deactivate_robot(run: Run, *, success: bool) -> None:
    robot = run.robot

    await home(run, robot)  # if the robot is lost, must home prior to dropping
    for pipette in run.pipettes.values():
        if pipette.has_tip:
            # TODO(april): this reracks into the source *clean* tiprack, which is janky. It might be
            # better to just drop the tips somewhere on deck for analysis?
            s = pipette.tip_source
            await drop_tip(s.bay if s else None, s.well if s else None, pipette, robot)
            pipette.has_tip = False
            pipette.tip_source = None

    for slot in run.deck.values():
        if not slot.module:
            continue

        module = slot.module
        if module.model == ModuleModel.TEMPERATURE_MODULE_V2:
            request = DeactivateTemperatureCreate(
                params=DeactivateTemperatureParams(moduleId=module.location.moduleId),
                intent=CommandIntent.PROTOCOL,
                key=None,
            )
            await execute(request, robot)
        elif module.model == ModuleModel.THERMOCYCLER_MODULE_V2:
            await deactivate_thermocycler_block(at_slot=slot, run=run, robot=robot)
            await deactivate_thermocycler_lid(at_slot=slot, run=run, robot=robot)
            await set_thermocycler_lid_hinge(at_slot=slot, closed=False, run=run, robot=robot)

    await home(run, robot)

    if success:
        await set_status_bar(StatusBarState.IDLE, robot)
    else:
        await set_status_bar(StatusBarState.SOFTWARE_ERROR, robot)


async def execute(request: CommandCreate, robot: Robot):
    queued = QueueCommandAction(
        command_id=ModelUtils.generate_id(),
        created_at=ModelUtils.get_timestamp(),
        request=request,
        request_hash=None,
    )
    robot.state_store.handle_action(queued)
    await robot.executor.execute(queued.command_id)
    after = robot.state_store.commands.get(queued.command_id)
    if after.status == CommandStatus.FAILED:
        raise Exception(repr(after.error))
    return after.result


def order_locks(bays: list[str], run: Run) -> list[asyncio.Lock]:
    return list(map(lambda b: run.deck[b].lock, dict.fromkeys(sorted(bays)).keys()))


def make_spaces(*, rows, cols) -> list[str]:
    row_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    offset = 0
    spots = []
    for col in reversed(range(cols)):
        for row in reversed(row_names[0:rows]):
            spots.append(f"{row}{col + 1}")
    return spots
