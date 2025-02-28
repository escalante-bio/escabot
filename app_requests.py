from typing import Literal

from pydantic import BaseModel, Field, root_validator
from opentrons.protocol_engine import WellOffset


class WellLocationRequest(BaseModel):
    bay: str
    well: str


class WellOffsetRequest(BaseModel):
    x_mm: float = Field(alias="xMm")
    y_mm: float = Field(alias="yMm")
    z_mm: float = Field(alias="zMm")


class AddLabwareDefinitionRequest(BaseModel):
    kind: Literal["add_labware_definition"]
    definition: dict


class AspirateRequest(BaseModel):
    kind: Literal["aspirate"]
    at: WellLocationRequest
    channels: int
    offset: WellOffsetRequest
    volume_nl: float = Field(alias="volumeNl")


class DispenseRequest(BaseModel):
    kind: Literal["dispense"]
    at: WellLocationRequest
    channels: int
    offset: WellOffsetRequest
    volume_nl: float = Field(alias="volumeNl")


class DropTipRequest(BaseModel):
    kind: Literal["drop_tip"]
    channels: int
    well: WellLocationRequest | None = None


class HomeRequest(BaseModel):
    kind: Literal["home"]


class InitializeRequest(BaseModel):
    kind: Literal["initialize"]


class LoadLabwareRequest(BaseModel):
    kind: Literal["load_labware"]
    into: str
    model: str


class LoadModuleRequest(BaseModel):
    kind: Literal["load_module"]
    into: str
    module: Literal["magnetic"] | Literal["temperature"] | Literal["thermocycling"] | None = None


class MoveLabwareRequest(BaseModel):
    kind: Literal["move_labware"]
    from_: str = Field(alias="from")
    to: str

    @staticmethod
    @root_validator(pre=True)
    def alias_values(values):
        values["from_"] = values.pop("from")
        return values


class MoveToWellRequest(BaseModel):
    kind: Literal["move_to_well"]
    channels: int
    to: WellLocationRequest
    offset: WellOffset


class PickUpTipRequest(BaseModel):
    kind: Literal["pick_up_tip"]
    channels: int
    well: WellLocationRequest


class TemperatureBlockTemperatureRequest(BaseModel):
    kind: Literal["temperature_block_temperature"]
    at: str
    temperature_c: float = Field(alias="temperatureC")


class ThermocycleBlockDeactivateRequest(BaseModel):
    kind: Literal["thermocycle_block_deactivate"]
    at: str


class ThermocycleBlockTemperatureRequest(BaseModel):
    kind: Literal["thermocycle_block_temperature"]
    at: str
    duration_us: float | None = Field(alias="durationUs", default=None)
    max_volume_nl: float = Field(alias="maxVolumeNl")
    temperature_c: float = Field(alias="temperatureC")


class ThermocycleLidDeactivateRequest(BaseModel):
    kind: Literal["thermocycle_lid_deactivate"]
    at: str


class ThermocycleLidHingeRequest(BaseModel):
    kind: Literal["thermocycle_lid_hinge"]
    at: str
    closed: bool


class ThermocycleLidTemperatureRequest(BaseModel):
    kind: Literal["thermocycle_lid_temperature"]
    at: str
    temperature_c: float = Field(alias="temperatureC")


class WaitRequest(BaseModel):
    kind: Literal["wait"]
    duration_us: float = Field(alias="durationUs")


InstructionRequest = (
    AddLabwareDefinitionRequest
    | AspirateRequest
    | DispenseRequest
    | DropTipRequest
    | HomeRequest
    | InitializeRequest
    | LoadLabwareRequest
    | LoadModuleRequest
    | MoveLabwareRequest
    | MoveToWellRequest
    | PickUpTipRequest
    | TemperatureBlockTemperatureRequest
    | ThermocycleBlockDeactivateRequest
    | ThermocycleBlockTemperatureRequest
    | ThermocycleLidDeactivateRequest
    | ThermocycleLidHingeRequest
    | ThermocycleLidTemperatureRequest
    | WaitRequest
)


class RunCreateRequest(BaseModel):
    id: str


class StreamCreateRequest(BaseModel):
    id: str
    instructions: list[InstructionRequest]
