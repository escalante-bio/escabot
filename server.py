import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Path
from opentrons.hardware_control.types import StatusBarState
from opentrons.protocol_engine.state import TemperatureModuleId, ThermocyclerModuleId

from app_requests import (
    InstructionRequest,
    RunCreateRequest,
    StreamCreateRequest,
)
from app_types import ExecutionState, Instruction, RobotDeckSlot, Run, Stream
from robot import (
    RobotHardware,
    create_opentrons_hardware,
    create_opentrons_state,
    deactivate_robot,
    execute_instruction,
)


@dataclass(kw_only=True)
class AppState:
    create_run_lock: asyncio.Lock
    executing: Run | None
    robot_hardware: RobotHardware
    runs: dict[str, Run]


def async_cache(fn):
    state = {}

    @wraps(fn)
    async def invoke(*args, **kwargs):
        if "value" not in state:
            if asyncio.iscoroutinefunction(fn):
                state["value"] = await fn(*args, **kwargs)
            else:
                state["value"] = fn(*args, **kwargs)
        return state["value"]

    return invoke


@async_cache
async def create_state() -> AppState:
    simulate_hardware = os.environ.get("USE_VIRTUAL_HARDWARE", "false").lower() == "true"
    return AppState(
        create_run_lock=asyncio.Lock(),
        executing=None,
        robot_hardware=await create_opentrons_hardware(simulate_hardware),
        runs={},
    )


app = FastAPI()


@app.get("/robot/status")
async def robot_status(
    state: Annotated[AppState, Depends(create_state)],
):
    run = state.executing
    response = {}
    if run:
        robot = run.robot
        temperature = run.deck["C1"].module
        if temperature:
            hardware = robot.equipment.get_module_hardware_api(
                cast(TemperatureModuleId, temperature.moduleId)
            )
        else:
            hardware = None
        if hardware:
            response["C1"] = {
                "module": {
                    "kind": "temperature",
                    "temperature": hardware.temperature,
                    "target": hardware.target,
                }
            }

            # print(
            #    state.robot.state_store.modules.get_temperature_module_substate(
            #        temperature.moduleId
            #    )
            # )

        thermocycler = run.deck["B1"].module
        if thermocycler:
            hardware = robot.equipment.get_module_hardware_api(
                cast(ThermocyclerModuleId, thermocycler.moduleId)
            )
        else:
            hardware = None
        if hardware:
            response["B1"] = {
                "module": {
                    "kind": "thermocycler",
                    "block_status": hardware.status,
                    "block_target": hardware.target,
                    "block_temp": hardware.temperature,
                    "lid_status": hardware.lid_status,
                    "lid_target": hardware.lid_target,
                    "lid_temp": hardware.lid_temp,
                }
            }

            # print(
            #    state.robot.state_store.modules.get_thermocycler_module_substate(
            #        thermocycler.moduleId
            #    )
            # )

    return response


@app.put("/run/create")
async def run_create(
    request: RunCreateRequest,
    state: Annotated[AppState, Depends(create_state)],
):
    await state.create_run_lock.acquire()
    try:
        if state.executing and state.executing.active:
            if state.executing.id != request.id:
                raise HTTPException(409, "Another run is executing")
        else:
            if request.id in state.runs:
                raise HTTPException(409, f"Run {request.id} has already been archived")
            robot = await create_opentrons_state(state.robot_hardware)
            run = Run.create(request.id, robot)
            run.cleanup_task = asyncio.create_task(run_worker(run, state))
            state.runs[request.id] = run
            state.executing = run
    finally:
        state.create_run_lock.release()
    return "ok"


@app.put("/run/{run_id}/close")
async def run_close(
    run_id: Annotated[str, Path(title="The ID of the parent run")],
    state: Annotated[AppState, Depends(create_state)],
    force: bool = False,
):
    run = get_run(run_id, state)
    run.failed = force
    run.cleanup_event.set()
    while run.active:
        await asyncio.sleep(0.2)
    return "ok"


@app.put("/run/{run_id}/stream/create")
async def run_stream_create(
    request: StreamCreateRequest,
    run_id: Annotated[str, Path(title="The ID of the parent run")],
    state: Annotated[AppState, Depends(create_state)],
):
    run = get_run(run_id, state)
    stream_id = request.id
    if stream_id not in run.streams:
        if run.closed:
            raise HTTPException(400, "Run is closed")
        stream = Stream(
            id=stream_id, active=True, failed=False, instructions={}, queue=[], worker=None
        )
        instructions = request.instructions
        logging.warning("Creating stream %s with instructions:\n%s", stream_id, instructions)
        stream.worker = asyncio.create_task(stream_worker(instructions, stream, run, state))
        run.streams[stream_id] = stream

    return "ok"


@app.get("/run/{run_id}/stream/{stream_id}/wait")
async def run_stream_wait(
    run_id: Annotated[str, Path(title="The ID of the parent run")],
    stream_id: Annotated[str, Path(title="The ID of the stream")],
    state: Annotated[AppState, Depends(create_state)],
):
    run = get_run(run_id, state)
    stream = get_stream(stream_id, run)
    start = time.time()
    while stream.active:
        if time.time() - start > 30:
            return {
                "kind": "timeout",
                "status": run_stream_status(run_id, stream_id, state),
            }
        await asyncio.sleep(0.2)
    if stream.failed:
        return {
            "kind": "failed",
        }
    return {"kind": "finished"}


@app.get("/run/{run_id}/stream/{stream_id}/status")
def run_stream_status(
    run_id: Annotated[str, Path(title="The ID of the parent run")],
    stream_id: Annotated[str, Path(title="The ID of the stream")],
    state: Annotated[AppState, Depends(create_state)],
):
    run = get_historical_run(run_id, state)
    if not run:
        raise HTTPException(404, "No such run")
    stream = get_stream(stream_id, run)
    in_progress = 1 if stream.active else 0
    return {
        "state": "active" if stream.active else "failed" if stream.failed else "done",
    }


async def close_run(run: Run, state: AppState, *, force: bool):
    run.closed = False

    for stream in run.streams.values():
        if not force:
            while stream.active:
                await asyncio.sleep(0.2)
        else:
            for instruction in stream.queue:
                instruction.task.cancel()
        del stream.queue[:]
        stream.active = False

    await deactivate_robot(run, success=not run.failed)
    run.active = False


def get_run(run_id: str, state: AppState) -> Run:
    if not state.executing:
        raise HTTPException(404, f"No executing run")
    if state.executing.id != run_id:
        raise HTTPException(409, f"Run {state.executing.id} is executing, not {run_id}")
    return state.executing


def get_historical_run(run_id: str, state: AppState) -> Run | None:
    run = state.runs.get(run_id)
    if run:
        return run
    else:
        raise HTTPException(404, "No such run")


def get_stream(stream_id: str, run: Run) -> Stream:
    if stream := run.streams.get(stream_id):
        return stream
    else:
        raise HTTPException(404, "No such stream")


def get_instruction(instruction_id: str, stream: Stream) -> Instruction:
    if instruction := stream.instructions.get(instruction_id):
        return instruction
    else:
        raise HTTPException(404, "No such instruction")


async def queue_instruction(instruction: Instruction, run: Run):
    await instruction.execution_barrier.acquire()
    try:
        instruction.state = ExecutionState.EXECUTING
        logging.info("Executing instruction: %s", json.dumps(instruction.raw))
        await execute_instruction(instruction.raw, run)
        instruction.state = ExecutionState.SUCCEEDED
    except:
        instruction.state = ExecutionState.FAILED
        raise
    finally:
        instruction.execution_barrier.release()


async def run_worker(run: Run, state: AppState):
    try:
        await run.cleanup_event.wait()
    finally:
        await close_run(run, state, force=run.failed)


async def stream_worker(
    instructions: list[InstructionRequest], stream: Stream, run: Run, state: AppState
):
    for i in range(len(instructions)):
        raw = instructions[i]
        try:
            await execute_instruction(raw, run)
        except Exception as e:
            stream.failed = True
            run.failed = True
            if not isinstance(e, asyncio.CancelledError):
                logging.exception(f"Task {i} in stream {stream.id} failed, closing run {run.id}")
                run.cleanup_event.set()
            raise e

    stream.active = False
