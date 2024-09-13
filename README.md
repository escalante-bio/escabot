# escabot

A FastAPI-based wrapper for an Opentrons robot that provides an asynchronous interface.

This means that you can (hopefully safely):
* thermocycle while pipetting somewhere else
* cool the temperature module and the thermocycler at the same time
* and perhaps more

## Should you use this?

Probably not right now! It is unstable, buggy, and subject to design change.

Future goals:
* Add magnet block and heater shaker support
* In an earlier version I supported appending instructions to the stream incrementally. I may revert
  back to that at some point. Still uncertain about which way is better.
* Add key support for pipette tips to enable reusing tips across successive pipetting operations
* Continue wringing out the bugs
* 8-channel pipette support? 96-channel?
* OT-2 support if someone wants to give me an OT-2
