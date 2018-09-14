'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 1000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 5.0.8
' Optimize                       = Yes
' Optimize_Level                 = 1
' Info_Last_Save                 = DARWIN-PC  Darwin-PC\PNPNteam
'<Header End>
#Include ADwinGoldII.inc
DIM DATA_1[20003] AS LONG AS FIFO  
DIM DATA_2[20003] AS LONG AS FIFO

INIT:
  FIFO_Clear(1)
  FIFO_Clear(2)
  Par_9 = 0
  
EVENT:
  Par_9 = DATA_2
  DAC(1, Par_9)
  DATA_1 = ADC(2)
