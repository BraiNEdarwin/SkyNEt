#include <SPI.h>

int switcharray[8];
const byte numBytes = 8;
byte receivedBytes[numBytes];
byte numReceived = 0;
const int CSPin = 10;
static byte i = 0;
boolean flag = false;
boolean newData = false;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(CSPin, OUTPUT);
  digitalWrite(CSPin, HIGH);
  Serial.setTimeout(200);
  SPI.beginTransaction(SPISettings(4000000, LSBFIRST, SPI_MODE0));
  SPI.begin();

}

void loop() {
  // put your main code here, to run repeatedly:
  recvBytesWithMarkers();
  SendSignals();
}



void recvBytesWithMarkers(){
  static boolean recvProgress = false;
  static byte ndx = 0;
  byte StartMark = 0x3C;
  byte EndMark = 0x3E;
  int counter = 0;

  while(Serial.available()>=0 && newData == false){
    char x = Serial.read();

    if (x == EndMark){
      recvProgress =false;
      newData = true;
      receivedBytes[ndx] = '\0';
      parseData();
      }
     if(recvProgress ==true){
      receivedBytes[ndx] = x;
      ndx++;
      if(ndx >=numBytes){
        ndx= numBytes - 1;
        }
      }
      if(x == StartMark){
        ndx = 0;
        recvProgress = true;
        }
    }
  }

void parseData(){
  char * strtokIndx;
  strtokIndx = strtok(receivedBytes, ",");
  switcharray[0] = atoi(strtokIndx);
  strtokIndx = strtok(NULL, ",");
  switcharray[1] = atoi(strtokIndx);
  strtokIndx = strtok(NULL, ",");
  switcharray[2] = atoi(strtokIndx);
  strtokIndx = strtok(NULL, ",");
  switcharray[3] = atoi(strtokIndx);
  strtokIndx = strtok(NULL, ",");
  switcharray[4] = atoi(strtokIndx);
    strtokIndx = strtok(NULL, ",");
  switcharray[5] = atoi(strtokIndx);
    strtokIndx = strtok(NULL, ",");
  switcharray[6] = atoi(strtokIndx);
    strtokIndx = strtok(NULL, ",");
  switcharray[7] = atoi(strtokIndx);

  
  
  }

void SendSignals(){
  if (newData ==true){
    digitalWrite(CSPin, LOW);
    for ( int k=0; k < 8; k++){
      SPI.transfer(switcharray[i]);
      i++;
      }
      Serial.println();
    if(i == 8){
      digitalWrite(CSPin, HIGH);
      i = 0;
      newData = false;
      }
    }
}

