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
  int counter = 0;

  if(Serial.available() >= 0 && flag==false){
    switcharray = Serial.parseInt();
    flag = true;
    }
  if(flag == true && switcharray ==StartMark){
    recvProgress = true;
    flag = false;
    if(counter == 0){
      digitalWrite(CSPin, LOW);
      }
    }
  if(flag == true && switcharray ==EndMark){
    recvProgress = false;
    flag = false;
    if(counter ==8){
      digitalWrite(CSPin, HIGH);
      counter = 0;
      }
    }
  if(flag == true && recvProgress == true){
    SPI.transfer(switcharray);
    flag = false;
    counter++;
    }
  }
