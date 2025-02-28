from opentrons import protocol_api
from opentrons.execute import get_protocol_api


def run(protocol):
  protocol.home()

  temperature = protocol.load_module(module_name="temperature module gen2", location="C1")
  t_adapter = temperature.load_adapter("opentrons_96_well_aluminum_block")
  plate = t_adapter.load_labware("nest_96_wellplate_100ul_pcr_full_skirt")

  tip_racks = []
  tip_rack = protocol.load_labware(
    load_name="opentrons_flex_96_filtertiprack_50ul",
    location="B3"
  )
  tip_racks.append(tip_rack)

  pipette_1_50ul = protocol.load_instrument(
    instrument_name="flex_1channel_50",
    mount="left",
    tip_racks=tip_racks
  )

  pipette_8_50ul = protocol.load_instrument(
    instrument_name="flex_8channel_50",
    mount="right",
    tip_racks=tip_racks,
  )

  channels = 4
  pipette_8_50ul.configure_nozzle_layout(
      protocol_api.PARTIAL_COLUMN,
      start="A1",
      end=["B1", "C1", "D1", "E1", "F1", "G1"][channels - 2],
  )

  pipette_8_50ul.pick_up_tip(tip_rack.well("E1"))
  pipette_8_50ul.move_to(plate.well("E1").top(z=60))
  input("cow")
  pipette_8_50ul.drop_tip(tip_rack.well("E1").top(z=10))
  protocol.home()


def patch_drop_tips(protocol):
    from opentrons.protocol_engine import (
        DropTipWellLocation,
        DropTipWellOrigin,
        WellLocation,
        WellOffset,
        WellOrigin,
    )

    geometry = protocol._core._engine_client.state.geometry

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


def patch_pipette_loading():
    from opentrons_shared_data.pipette.pipette_definition import PipetteConfigurations

    original_model_validate = PipetteConfigurations.model_validate

    def evil_model_validate(obj, *args, **kwargs):
        valid_maps = obj["validNozzleMaps"]["maps"]
        configurations = obj["pickUpTipConfigurations"]["pressFit"]["configurationsByNozzleMap"]
        if obj["channels"] == 8:
            full = configurations["Full"]
            for start, ends in (
                ("A1", ["B1", "C1", "D1", "E1", "F1", "G1"]),
                ("H1", ["G1", "F1", "E1", "D1", "C1", "B1"]),
            ):
              for i, end in enumerate(ends):
                  key = f"{start}to{end}"
                  configurations[key] = full
                  valid_maps[key] = [start] + ends[:i + 1]
        return original_model_validate(obj, *args, **kwargs)

    PipetteConfigurations.model_validate = evil_model_validate

patch_pipette_loading()
protocol = get_protocol_api("2.22")
patch_drop_tips(protocol)
run(protocol)
