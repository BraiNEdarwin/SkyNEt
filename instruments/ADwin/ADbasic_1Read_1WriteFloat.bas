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

INIT:
  FIFO_Clear(1)
  FIFO_Clear(2)
  Seq_Mode(2, 2)
  Seq_Set_Gain(2, 1) Rem sets range to -5, 5V
  Seq_Select(0FFFFh)
  Seq_Start(10b)
  Par_9 = 0
  
EVENT:
  Par_9 = DATA_2
  DAC(1, Par_9)
  DATA_1 = Seq_Read(2) 
  Rem DATA_1 = ADC(2)
