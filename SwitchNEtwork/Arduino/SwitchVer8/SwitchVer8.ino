#include <SPI.h>

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
  byte switcharray;
  int counter = 0;

  while(Serial.available()>=0 && newData == false){
    switcharray = Serial.read();

    if (recvProgress ==true){
      if(switcharray !=EndMark){
        receivedBytes[ndx] = switcharray;
        ndx++;
        if(ndx >=numBytes){
          ndx = numBytes- 1;
          }
        }
      else{
        receivedBytes[ndx] = '\0'; // terminate the string
        recvProgress = false;
        numReceived = ndx;  // save the number for use when printing
        ndx = 0;
        newData = true;
        }
    }else if(switcharray ==StartMark){
      recvProgress =true;
      }
    
  }
  }

void SendSignals(){
  if (newData ==true){
    digitalWrite(CSPin, LOW);
    for ( int k=0; k < 8; k++){
      SPI.transfer(receivedBytes[i]);
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

