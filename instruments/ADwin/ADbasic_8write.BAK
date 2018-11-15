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
DIM DATA_1[40003] AS LONG AS FIFO  
DIM DATA_2[40003] AS LONG AS FIFO
DIM DATA_3[40003] AS LONG AS FIFO
DIM DATA_4[40003] AS LONG AS FIFO
DIM DATA_5[40003] AS LONG AS FIFO
DIM DATA_6[40003] AS LONG AS FIFO
DIM DATA_7[40003] AS LONG AS FIFO
DIM DATA_8[40003] AS LONG AS FIFO

INIT:
  FIFO_Clear(1)
  FIFO_Clear(2)
  FIFO_Clear(3)
  FIFO_Clear(4)
  FIFO_Clear(5)
  FIFO_Clear(6)
  FIFO_Clear(7)
  FIFO_Clear(8)
  
  Par_9 = 0
  Par_10 = 0
  Par_11 = 0
  Par_12 = 0
  Par_13 = 0
  Par_14 = 0
  Par_15 = 0
  Par_16 = 0
  
EVENT:
  Par_9 = DATA_1
  DAC(1, Par_9)
  Par_10 = DATA_2
  DAC(2, Par_10)
  Par_11 = DATA_3
  DAC(3, Par_11)
  Par_12 = DATA_4
  DAC(4, Par_12)
  Par_13 = DATA_5
  DAC(5, Par_13)
  Par_14 = DATA_6
  DAC(6, Par_14)
  Par_15 = DATA_7
  DAC(7, Par_15)
  Par_16 = DATA_8
  DAC(8, Par_16)
