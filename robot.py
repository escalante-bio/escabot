import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, Optional, cast

from fastapi import HTTPException
from opentrons.types import DeckSlotName, Mount, MountType
from opentrons.hardware_control import API as HardwareAPI, HardwareControlAPI
from opentrons.hardware_control.ot3api import OT3API
from opentrons.hardware_control.types import DoorState, HardwareFeatureFlags, StatusBarState
from opentrons.protocol_engine import (
    AddressableOffsetVector,
    Command,
    CommandCreate,
    CommandIntent,
    CommandStatus,
    Config,
    DeckType,
    DropTipWellLocation,
)
from opentrons.protocol_engine.actions import (
    ActionDispatcher,
    AddAddressableAreaAction,
    QueueCommandAction,
    RunCommandAction,
    # SetDeckConfigurationAction,
    SucceedCommandAction,
)
from opentrons.protocol_engine.commands import (
    AspirateCreate,
    AspirateParams,
    AspirateResult,
    DispenseCreate,
    DispenseParams,
    DispenseResult,
    DropTipInPlaceCreate,
    DropTipInPlaceParams,
    DropTipInPlaceResult,
    HomeCreate,
    HomeParams,
    HomeResult,
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
    MoveToAddressableAreaForDropTipCreate,
    MoveToAddressableAreaForDropTipParams,
    MoveToAddressableAreaForDropTipResult,
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
from opentrons.protocol_engine import WellLocation, WellOrigin, WellOffset
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
from opentrons.protocol_engine.resources import DeckDataProvider, ModelUtils, ModuleDataProvider
from opentrons.protocol_engine.state import CommandEntry, StateStore
from opentrons.protocol_engine.types import (
    AddressableAreaLocation,
    DeckSlotLocation,
    LabwareMovementOffsetData,
    LabwareMovementStrategy,
    LabwareOffsetVector,
    ModuleLocation,
    ModuleModel,
    OnLabwareLocation,
    WellLocation,
    WellOrigin,
)
from opentrons.protocols.api_support.deck_type import (
    guess_from_global_config as guess_deck_type_from_global_config,
)
from opentrons.protocols.api_support.definitions import MAX_SUPPORTED_VERSION
from opentrons.protocols.api_support.util import find_value_for_api_version
from opentrons.util import entrypoint_util
from opentrons_shared_data.pipette.dev_types import PipetteNameType

# from opentrons_shared_data.pipette.types import PipetteNameType
from opentrons_shared_data.robot import load as load_robot

from app_requests import (
    DropTipRequest,
    InitializeRequest,
    InstructionRequest,
    LoadPlateRequest,
    MovePlateRequest,
    PipetteRequest,
    TemperatureBlockRequest,
    ThermocyclerBlockRequest,
    ThermocyclerLidRequest,
)
from app_types import Robot, RobotDeckSlot, RobotHardware, RobotPipette, RobotTip, Run


# See the defaults at
# https://github.com/Opentrons/opentrons/blob/aadc65ec79ebbf66acd9df08dcfb7910d34cdfb9/api/src/opentrons/protocol_api/instrument_context.py#L44
ASPIRATE_DISPENSE_LOCATION = WellLocation(origin=WellOrigin.BOTTOM, offset=WellOffset(z=1))
# See the mapping at
# https://github.com/Opentrons/opentrons/blob/aadc65ec79ebbf66acd9df08dcfb7910d34cdfb9/api/src/opentrons/protocol_api/validation.py#L63
PIPETTE_1_CHANNEL = PipetteNameType.P50_SINGLE_FLEX
WASTE_CHUTE_AREA = "1ChannelWasteChute"


async def create_hardware_control(simulate_hardware: bool) -> HardwareControlAPI:
    if simulate_hardware:
        return await HardwareAPI.build_hardware_simulator(
            attached_instruments={
                Mount.LEFT: {"id": "p1", "model": "p50_single_v3.6"},
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
    # def _policy(
    #    config: Config,
    #    failed_command: Command,
    #    defined_error_data: Optional[CommandDefinedErrorData],
    # ) -> ErrorRecoveryType:
    def _policy(
        failed_command: Command,
        exception: Exception,
    ) -> ErrorRecoveryType:
        return ErrorRecoveryType.FAIL_RUN

    return _policy


async def create_state_store(hardware_api: HardwareControlAPI, config: Config) -> StateStore:
    deck_data = DeckDataProvider(config.deck_type)
    deck_definition = await deck_data.get_deck_definition()
    deck_fixed_labware = await deck_data.get_deck_fixed_labware(
        # TODO(april): not released yet
        # load_fixed_trash=False,
        deck_definition=deck_definition,
        # deck_configuration=None,
    )

    module_calibration_offsets = ModuleDataProvider.load_module_calibrations()
    robot_definition = load_robot(config.robot_type)
    state_store = StateStore(
        config=config,
        deck_definition=deck_definition,
        deck_fixed_labware=deck_fixed_labware,
        # TODO(april): not released yet
        # robot_definition=robot_definition,
        is_door_open=hardware_api.door_state is DoorState.OPEN,
        # TODO(april): not released yet
        # error_recovery_policy=create_error_recovery_policy(),
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
        assert prev_entry.command.status == CommandStatus.RUNNING
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
        # TODO(april): yikes! See https://github.com/Opentrons/opentrons/commit/b1b8361e279abe7e69c009c3749ec0bfd30a9bb4
        use_simulated_deck_config=True,
        # This seems expected
        use_virtual_pipettes=hardware.simulate_hardware,
        use_virtual_gripper=hardware.simulate_hardware,
        use_virtual_modules=hardware.simulate_hardware,
    )
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
            error_recovery_policy=create_error_recovery_policy(),
            movement=movement,
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
    # TODO(april): not released yet
    # robot.action_dispatcher.dispatch(
    #     SetDeckConfigurationAction(entrypoint_util.get_deck_configuration())
    # )
    await home(run, robot)
    await load_waste_chute(run, robot)
    await load_pipette(PIPETTE_1_CHANNEL, 50000, MountType.LEFT, run, robot)


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
        for temperature in instruction.temperatures:
            await load_module(ModuleModel.TEMPERATURE_MODULE_V2, temperature.bay, run, robot)
            # TODO(april): XXX bring this back when Opentrons fixes their garbage
            # await load_labware("opentrons_96_well_aluminum_block", temperature.bay, run, robot)
        for thermocycler in instruction.thermocyclers:
            await load_module(ModuleModel.THERMOCYCLER_MODULE_V2, thermocycler.bay, run, robot)
            await set_thermocycler_lid(
                at_slot=run.deck[thermocycler.bay],
                closed=False,
                temperature_c=50,
                run=run,
                robot=robot,
            )
        for tip_rack in instruction.tip_racks:
            loaded = await load_labware("opentrons_flex_96_tiprack_50ul", "C3", run, robot)
            run.pipettes[PIPETTE_1_CHANNEL].unused_tips.extend(
                map(lambda s: RobotTip(labware=loaded, well=s), make_spaces(rows=8, cols=12))
            )

        run.initialized = True
        return
    elif not run.initialized:
        raise Exception(f"Run {run.id} hasn't been initialized yet")

    if isinstance(instruction, DropTipRequest):
        await run.gantry_lock.acquire()
        await drop_tip(run, robot, validate=False)
        run.gantry_lock.release()
    elif isinstance(instruction, LoadPlateRequest):
        bay = instruction.into.bay
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
    elif isinstance(instruction, MovePlateRequest):
        from_bay = instruction.from_.bay
        from_slot = run.deck[from_bay]
        if not from_slot:
            raise HTTPException(400, f"Bay {from_bay} is invalid")
        to_bay = instruction.to.bay
        to_slot = run.deck[to_bay]
        if not to_slot:
            raise HTTPException(400, f"Bay {to_bay} is invalid")

        deck_locks = order_locks([from_bay, to_bay], run)
        for lock in deck_locks:
            await lock.acquire()

        if not from_slot.top_is_plate():
            raise HTTPException(409, f"Bay {from_bay} doesn't have a plate on top")
        if to_slot.top_is_plate():
            raise HTTPException(409, f"Bay {to_bay} already has a plate on top")

        await run.gantry_lock.acquire()
        await move_labware(from_slot, to_slot, run, robot)
        run.gantry_lock.release()
        for lock in reversed(deck_locks):
            lock.release()
    elif isinstance(instruction, PipetteRequest):
        volume_nl = instruction.volume_nl
        if not volume_nl:
            raise HTTPException(400, f"volume_nl is {volume_nl}")

        from_bay = instruction.from_.plate.bay
        from_slot = run.deck[from_bay]
        if not from_slot:
            raise HTTPException(400, f"Bay {from_bay} is invalid")
        to_bay = instruction.to.plate.bay
        to_slot = run.deck[to_bay]
        if not to_slot:
            raise HTTPException(400, f"Bay {to_bay} is invalid")

        deck_locks = order_locks([from_bay, to_bay], run)
        for lock in deck_locks:
            await lock.acquire()

        if not from_slot.top_is_plate():
            raise HTTPException(400, f"Bay {from_bay} doesn't have a plate on top")
        if not to_slot.top_is_plate():
            raise HTTPException(400, f"Bay {to_bay} doesn't have a plate on top")

        await run.gantry_lock.acquire()
        pipette = run.pipettes[PIPETTE_1_CHANNEL]
        remaining = volume_nl
        while remaining > 0:
            await pick_up_tip(run, robot)
            await aspirate(
                from_slot,
                instruction.from_.well,
                min(remaining, pipette.max_volume_nl),
                pipette,
                run,
                robot,
            )
            await dispense(
                to_slot,
                instruction.to.well,
                min(remaining, pipette.max_volume_nl),
                pipette,
                run,
                robot,
            )
            await drop_tip(run, robot)
            remaining -= pipette.max_volume_nl
        run.gantry_lock.release()
        for lock in reversed(deck_locks):
            lock.release()
    elif isinstance(instruction, TemperatureBlockRequest):
        at_bay = instruction.at.bay
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")
        await at_slot.lock.acquire()
        await set_temperature_block(at_slot, instruction.temperature_c, run, robot)
        at_slot.lock.release()
    elif isinstance(instruction, ThermocyclerBlockRequest):
        at_bay = instruction.at.bay
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")
        await at_slot.module_lock.acquire()
        await set_thermocycler_block(
            at_slot=at_slot,
            duration_us=instruction.duration_us,
            max_volume_nl=instruction.max_volume_nl,
            temperature_c=instruction.temperature_c,
            run=run,
            robot=robot,
        )
        at_slot.module_lock.release()
    elif isinstance(instruction, ThermocyclerLidRequest):
        at_bay = instruction.at.bay
        at_slot = run.deck[at_bay]
        if not at_slot:
            raise HTTPException(400, f"Bay {at_bay} is invalid")

        await at_slot.module_lock.acquire()
        if instruction.closed:
            await at_slot.lock.acquire()

        await run.gantry_lock.acquire()
        await set_thermocycler_lid(
            at_slot=at_slot,
            closed=instruction.closed,
            temperature_c=instruction.temperature_c,
            run=run,
            robot=robot,
        )
        run.gantry_lock.release()

        # It's possible the lid was already open and we got a request to open it, so we need to
        # check if we actually took the lock.
        if not instruction.closed and at_slot.lock.locked():
            at_slot.lock.release()
        at_slot.module_lock.release()
    else:
        raise HTTPException(400, f"Unknown command: {instruction}")
    logging.warning("Finished executing instruction %s", instruction)


async def aspirate(
    slot: RobotDeckSlot,
    well: str,
    volume_nl: int | float,
    pipette: RobotPipette,
    run: Run,
    robot: Robot,
) -> AspirateResult:
    if not pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} doesn't have a tip")
    pipetteId = pipette.pipette.pipetteId

    top = slot.stack[-1]
    if not isinstance(top, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    request = AspirateCreate(
        params=AspirateParams(
            labwareId=top.labwareId,
            pipetteId=pipetteId,
            volume=volume_nl / 1000,
            wellName=well,
            wellLocation=ASPIRATE_DISPENSE_LOCATION,
            flowRate=find_value_for_api_version(
                MAX_SUPPORTED_VERSION,
                robot.state_store.pipettes.get_flow_rates(pipetteId).default_aspirate,
            ),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(AspirateResult, await execute(request, robot))
    return result


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
    pipette: RobotPipette,
    run: Run,
    robot: Robot,
) -> DispenseResult:
    if not pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} doesn't have a tip")
    pipetteId = pipette.pipette.pipetteId

    top = slot.stack[-1]
    if not isinstance(top, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    request = DispenseCreate(
        params=DispenseParams(
            labwareId=top.labwareId,
            pipetteId=pipetteId,
            volume=volume_nl / 1000,
            wellName=well,
            wellLocation=ASPIRATE_DISPENSE_LOCATION,
            pushOut=None,
            flowRate=find_value_for_api_version(
                MAX_SUPPORTED_VERSION,
                robot.state_store.pipettes.get_flow_rates(pipetteId).default_dispense,
            ),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(DispenseResult, await execute(request, robot))
    return result


async def drop_tip(run: Run, robot: Robot, *, validate=True) -> DropTipInPlaceResult:
    pipette = run.pipettes[PIPETTE_1_CHANNEL]
    if validate and not pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} doesn't have a tip")

    move = MoveToAddressableAreaForDropTipCreate(
        params=MoveToAddressableAreaForDropTipParams(
            addressableAreaName=WASTE_CHUTE_AREA,
            pipetteId=pipette.pipette.pipetteId,
            alternateDropLocation=None,
            forceDirect=False,
            ignoreTipConfiguration=None,
            minimumZHeight=None,
            # Some bug in OT software means we need to set this to 20
            offset=AddressableOffsetVector(x=0, y=0, z=20),
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
    result = cast(DropTipInPlaceResult, await execute(drop, robot))

    pipette.has_tip = False
    return result


async def load_labware(model: str, location: str, run: Run, robot: Robot) -> OnLabwareLocation:
    actual = run.deck[location].stack[-1]
    request = LoadLabwareCreate(
        params=LoadLabwareParams(
            location=actual,
            loadName=model,
            namespace="opentrons",
            version=1,
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
    run.deck[location].stack.append(ModuleLocation(moduleId=result.moduleId))
    return result


async def load_pipette(
    pipette_name: PipetteNameType,
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
            # TODO(april): not ready yet
            # liquidPresenceDetection=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(LoadPipetteResult, await execute(request, robot))
    run.pipettes[pipette_name] = RobotPipette(
        pipette=result,
        has_tip=False,
        max_volume_nl=max_volume_nl,
        unused_tips=[],
    )
    return result


async def load_waste_chute(run: Run, robot: Robot) -> None:
    robot.action_dispatcher.dispatch(
        AddAddressableAreaAction(
            addressable_area=AddressableAreaLocation(addressableAreaName=WASTE_CHUTE_AREA),
        )
    )


async def move_labware(
    before: RobotDeckSlot, after: RobotDeckSlot, run: Run, robot: Robot
) -> MoveLabwareResult:
    was = before.stack[-1]
    if not isinstance(was, OnLabwareLocation):
        # We can still try to move the temperature adapter... oops
        raise Exception(f"Expected labware to be on top")

    request = MoveLabwareCreate(
        params=MoveLabwareParams(
            labwareId=was.labwareId,
            newLocation=after.stack[-1],
            strategy=LabwareMovementStrategy.USING_GRIPPER,
            pickUpOffset=None,
            dropOffset=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(MoveLabwareResult, await execute(request, robot))
    after.stack.append(before.stack.pop())
    return result


async def pick_up_tip(run: Run, robot: Robot) -> PickUpTipResult:
    pipette = run.pipettes[PIPETTE_1_CHANNEL]
    if pipette.has_tip:
        raise HTTPException(409, f"Pipette {PIPETTE_1_CHANNEL} already has a tip")

    tip = pipette.unused_tips.pop()
    request = PickUpTipCreate(
        params=PickUpTipParams(
            labwareId=tip.labware.labwareId,
            wellName=tip.well,
            pipetteId=pipette.pipette.pipetteId,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    result = cast(PickUpTipResult, await execute(request, robot))
    pipette.has_tip = True
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
            moduleId=module.moduleId,
            celsius=float(temperature_c),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)

    request = WaitForTemperatureCreate(
        params=WaitForTemperatureParams(
            moduleId=module.moduleId,
            celsius=None,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(WaitForTemperatureResult, await execute(request, robot))


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
            moduleId=module.moduleId,
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
            moduleId=module.moduleId,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    return cast(WaitForBlockTemperatureResult, await execute(request, robot))


async def set_thermocycler_lid(
    *, at_slot: RobotDeckSlot, closed: bool, temperature_c: int | float, run: Run, robot: Robot
) -> WaitForLidTemperatureResult:
    module = at_slot.module
    if not module:
        raise HTTPException(400, f"Bay {at_slot.location} doesn't have a module")

    if closed:
        request = CloseLidCreate(
            params=CloseLidParams(moduleId=module.moduleId),
            intent=CommandIntent.PROTOCOL,
            key=None,
        )
        await execute(request, robot)

    request = SetTargetLidTemperatureCreate(
        params=SetTargetLidTemperatureParams(
            moduleId=module.moduleId,
            celsius=float(temperature_c),
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)

    request = WaitForLidTemperatureCreate(
        params=WaitForLidTemperatureParams(
            moduleId=module.moduleId,
        ),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    response = cast(WaitForLidTemperatureResult, await execute(request, robot))

    if not closed:
        request = OpenLidCreate(
            params=OpenLidParams(moduleId=module.moduleId),
            intent=CommandIntent.PROTOCOL,
            key=None,
        )
        await execute(request, robot)
    return response


async def deactivate_robot(run: Run, *, success: bool) -> None:
    robot = run.robot

    pipette = run.pipettes[PIPETTE_1_CHANNEL]
    if pipette.has_tip:
        await drop_tip(run, robot)

    temperature = run.deck["C1"].module
    if not temperature:
        raise HTTPException(400, f"Bay C1 doesn't have a temperature")
    request = DeactivateTemperatureCreate(
        params=DeactivateTemperatureParams(moduleId=temperature.moduleId),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)

    thermocycler = run.deck["B1"].module
    if not thermocycler:
        raise HTTPException(400, f"Bay B1 doesn't have a thermocycler")
    request = DeactivateBlockCreate(
        params=DeactivateBlockParams(moduleId=thermocycler.moduleId),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)
    request = DeactivateLidCreate(
        params=DeactivateLidParams(moduleId=thermocycler.moduleId),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)
    request = OpenLidCreate(
        params=OpenLidParams(moduleId=thermocycler.moduleId),
        intent=CommandIntent.PROTOCOL,
        key=None,
    )
    await execute(request, robot)
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
