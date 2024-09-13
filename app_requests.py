from typing import Literal, Optional

from pydantic import BaseModel, Field, root_validator


class PlateLocationRequest(BaseModel):
    bay: str
    unit: str


class WellLocationRequest(BaseModel):
    plate: PlateLocationRequest
    well: str


class DropTipRequest(BaseModel):
    kind: Literal["drop_tip"]


class InitializeRequest(BaseModel):
    kind: Literal["initialize"]
    temperatures: list[PlateLocationRequest]
    thermocyclers: list[PlateLocationRequest]
    tip_racks: list[PlateLocationRequest]


class LoadPlateRequest(BaseModel):
    kind: Literal["load_plate"]
    into: PlateLocationRequest
    model: str


class MovePlateRequest(BaseModel):
    kind: Literal["load_plate"]
    from_: PlateLocationRequest = Field(alias="from")
    to: PlateLocationRequest

    @staticmethod
    @root_validator(pre=True)
    def alias_values(values):
        values["from_"] = values.pop("from")


class PipetteRequest(BaseModel):
    kind: Literal["pipette"]
    volume_nl: float = Field(alias="volumeNl")
    from_: WellLocationRequest = Field(alias="from")
    to: WellLocationRequest

    @staticmethod
    @root_validator(pre=True)
    def alias_values(values):
        values["from_"] = values.pop("from")


class TemperatureBlockRequest(BaseModel):
    kind: Literal["temperature_block"]
    at: PlateLocationRequest
    temperature_c: float = Field(alias="temperatureC")


class ThermocyclerBlockRequest(BaseModel):
    kind: Literal["thermocycler_block"]
    at: PlateLocationRequest
    duration_us: Optional[float] = Field(alias="durationUs")
    max_volume_nl: float = Field(alias="maxVolumeNl")
    temperature_c: float = Field(alias="temperatureC")


class ThermocyclerLidRequest(BaseModel):
    kind: Literal["thermocycler_lid"]
    at: PlateLocationRequest
    closed: bool
    temperature_c: float = Field(alias="temperatureC")


InstructionRequest = (
    DropTipRequest
    | InitializeRequest
    | LoadPlateRequest
    | MovePlateRequest
    | PipetteRequest
    | TemperatureBlockRequest
    | ThermocyclerBlockRequest
    | ThermocyclerLidRequest
)


class RunCreateRequest(BaseModel):
    id: str


class StreamCreateRequest(BaseModel):
    id: str
    instructions: list[InstructionRequest]

