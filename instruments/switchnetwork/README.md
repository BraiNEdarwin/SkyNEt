# Switch network

This small readme covers the basic steps to get going with the
switch network pcb/dipstick. First, this gives an overview of how things
work. Then, it will give some installation instructions.

## Overview

The switch network dipstick has:

+ 8 connections (BNC) to the outside world, from now on called C1, C2
  etc.
+ 8 devices that can be bonded onto the PCB, from now on called D1, D2, 
  etc.
+ 8 electrodes per device, from now on called E1, E2, etc.
+ 8 switching ICs (MAX395), each containing eight individually targetable
  switches. Each IC controls the connections to one C. E.g., if the 
  state of the IC controlling C1 is 11010000, then E1 of device
  1, 2 and 4 are connected to C1 (see the connection matrix below).

![alt text](./pcb_schematic.svg "Connections schematic")

The most important thing to understand is the switching mechanism. On 
the pcb there are 8 switching ICs daisy chained. This means that any 
data that switching IC 1 receives is passed onto IC 2.
So suppose we want to apply one switching configuration to the pcb.
We will then send an 8 byte signal, where each byte has eight bits.
The first byte will control the switches connecting to C8 (since it is
sent first it will be pushed forward by all bytes sent after it).

### Example

Suppose we wish to connect the following:

+ Interconnect E3 of D1, D2 and D5
+ Interconnect E5 of D2, D3 and D8
+ Connect E8 of D7

Then we pass the following list of bytes:

`[00000010, 00000000, 00000000, 01100001, 00000000, 11001000, 00000000,
 00000000]`

## Installation

For the functions in `switch_utils.py` to work, it is important that
the Arduino sketch `switch_driver.ino` is uploaded to the Arduino.
This can be easily done from the Arduino IDE and should already be
fine. 
