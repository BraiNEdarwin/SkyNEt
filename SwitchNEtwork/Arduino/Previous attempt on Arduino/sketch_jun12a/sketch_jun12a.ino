#include <SPI.h>
int SwitchArrayConfig[8];
const int ChipSelectPin = 10;
const int clockPin = 13;
const int dataPin = 11;
const int max395PowerPin = 12;
int i=0; // index of the active MAX395 chip (out of 8 connected serially in a row) 
int flag;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(clockPin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(ChipSelectPin, OUTPUT);
  pinMode(max395PowerPin, OUTPUT);
  digitalWrite(max395PowerPin, HIGH);
  //SPI.pinMode(dataPin, OUTPUT);
  SPI.setBitOrder(MSBFIRST);
  SPI.setDataMode(SPI_MODE0);
  SPI.setClockDivider(SPI_CLOCK_DIV2);
  Serial.setTimeout(5);
  SPI.begin();
  }

void loop() {
  // put your main code here, to run repeatedly:
  if ( Serial.available()){
    // cast the string read in an integer 
    SwitchArrayConfig[8] = Serial.read();
    flag= 1;
}
 if (flag == 1)
 { 
  //Write our Slave select low to enable the SHift register to begin listening for data
  digitalWrite(ChipSelectPin, LOW);
  //Transfer the 8-bit value of data to shift register, remembering that the least significant bit goes first
  SPI.transfer(SwitchArrayConfig[i]);
  i++;
  //Serial.write(SwitchArrayConfig);
    if(i==8)
    {
    digitalWrite(ChipSelectPin, HIGH);
    i=0;
    } 
 flag = 0;
 }
}
