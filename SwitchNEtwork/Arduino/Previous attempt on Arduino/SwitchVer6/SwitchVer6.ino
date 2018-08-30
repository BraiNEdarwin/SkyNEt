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
  Serial.setTimeout(200);
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
    switcharray = Serial.parseInt();
    if (switcharray == StartMark){
      recvProgress = true;
      }
    if(recvProgress == true){
        digitalWrite(CSPin, LOW);
        delay(10);
        for (i = 0; i < 8; i++){
          SPI.transfer(switcharray);
          delay(10);
          }
          digitalWrite(CSPin,HIGH);
          i = 0;
          delay(1000);
      }
     if(switcharray == EndMark){
      recvProgress =false;
      }

    }

  }
