#include <SPI.h>

const byte numBytes = 10;
byte receivedBytes[numBytes];
byte numReceived = 0;
const int CSPin = 10;
int i = 0;
boolean flag = false;
boolean newData = false;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(CSPin, OUTPUT);
  Serial.setTimeout(8);
  SPI.beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE0));
  SPI.begin();

}

void loop() {
  // put your main code here, to run repeatedly:
  recvBytesWithMarkers();
  
}

void recvBytesWithMarkers(){
  static boolean recvProgress = false;
  static byte ndx = 0;
  byte StartMark = 0x3C;
  byte EndMark = 0x3E;
  byte switcharray;
  while (Serial.available() >= 0 && flag == false){
    switcharray = Serial.read();
    if (switcharray == StartMark){
      recvProgress = true;
      }
    if(recvProgress == true){
      if(switcharray != EndMark){
        receivedBytes[ndx] = switcharray;
        ndx++;
        }
      else if(switcharray == EndMark){
        recvProgress = false;
        flag = true;
        }
      }
    }
    if(flag == true){
      digitalWrite(CSPin, LOW);
      delay(1);
      SPI.transfer(receivedBytes[i]);
      i++;
      if(i ==8){
        digitalWrite(CSPin, HIGH);
        i = 0;
        ndx = 0;
        flag == false;
        }
      }
  }
