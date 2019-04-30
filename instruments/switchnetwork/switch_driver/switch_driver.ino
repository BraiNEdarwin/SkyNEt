#include <SPI.h>

const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing
int switcharray[8]; // variables to hold the parsed data
const int CSPin = 10;
int i = 0;
boolean flag = false;



boolean newData = false;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(CSPin, OUTPUT);
  digitalWrite(CSPin, HIGH);
  Serial.setTimeout(8);
  SPI.beginTransaction(SPISettings(4000000, LSBFIRST, SPI_MODE0));
  SPI.begin();

}

void loop() {
  // put your main code here, to run repeatedly:
  recvWithStartEndMarkers();
  if (newData == true) {
        strcpy(tempChars, receivedChars);
            // thist temporary copy is necessary to protect the original data
            //   because strtok() used in parseData() replaces the commas with \0
        parseData();
        showParsedData();
        scrambleData();
        SendSignal();
    }
}

void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

//============

void parseData() {      // split the data into its parts

    char * strtokIndx; // this is used by strtok() as an index
    strtokIndx = strtok(tempChars,",");
    for(int i=0; i<8; i++){
      switcharray[i]=atoi(strtokIndx);
      //Serial.print(strtokIndx);
      //Serial.print(atoi(strtokIndx));
      strtokIndx = strtok(NULL,",");
    }

    
    flag = true;
}

//============

void showParsedData() {
    Serial.print("switch configs matrix for each switch:\n");
    for (int j = 0; j<8; j++){
      //Serial.print(switcharray[j]);   
      for (int i = 7; i >= 0; i--)
      {
         bool b = bitRead(switcharray[j], i);
         Serial.print(b);
      }
      Serial.print("\n");
    }
}

//============
/*
 This is necessary because the device numbers are not connected as the figure shows 
 or as is seen on the board itself.
 The real order goes like this:
 D1 D3  D5  D7
 D2 D4  D6  D8
 this function scrambles the bits so that the original numbering can be followed
 */
//============
void scrambleData(){
  int temparray[8];
  for(int i=0; i<8;i++){
    temparray[i]=0;
    temparray[i]+=switcharray[i]&0b11000001;
    temparray[i]+=(switcharray[i]&0b00100000)>>2;
    temparray[i]+=(switcharray[i]&0b00010000)>>3;
    temparray[i]+=(switcharray[i]&0b00001000)<<2;
    temparray[i]+=(switcharray[i]&0b00000100)<<2;
    temparray[i]+=(switcharray[i]&0b00000010)<<1;
    switcharray[i]=temparray[i];   
  }
  Serial.print("scramble data\n");
      for (int j = 0; j<8; j++){
      //Serial.print(switcharray[j]);   
      for (int i = 7; i >= 0; i--)
      {
         bool b = bitRead(switcharray[j], i);
         Serial.print(b);
      }
      Serial.print("\n");
    }
}

//============
void SendSignal(){
  if(newData == true && flag == true){
    digitalWrite(CSPin, LOW);
    delay(10);
    //reverse order in which the bytes are sent, so that the last byte will be pushed to the last switch
    for (int k = 7; k >= 0; k--){
      SPI.transfer(switcharray[k]);
      i++;
      }
    if(i == 8){
      Serial.write("done");
      digitalWrite(CSPin, HIGH);
      i = 0;
      newData = false;
      flag = false;
      }
    }
  }
