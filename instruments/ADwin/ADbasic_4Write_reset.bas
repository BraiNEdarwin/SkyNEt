'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 3000
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

INIT:
  
  Rem If Par is kept after stopping the process, we can just read Par 1 to 4 which contain the last written values on the 4 DACs
  
  
  Rem start flag
  Par_80 = 0
  

  
EVENT:
  Rem If Par_80 = 1 Then
    
    
