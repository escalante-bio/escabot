# escabot

A FastAPI-based wrapper for an Opentrons robot that provides an asynchronous interface.

This means that you can (hopefully safely):
* thermocycle while pipetting somewhere else
* cool the temperature module and the thermocycler at the same time
* and perhaps more

## Should you use this?

Probably not right now! It is unstable, buggy, and subject to design change.

Future goals:
* Add heater shaker support
* Continue wringing out the bugs
* 96-channel support?
* OT-2 support if someone wants to give me an OT-2
