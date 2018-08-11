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
  SPI.beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE0));
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

    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    switcharray[0] = atoi(strtokIndx);
 
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    switcharray[1] = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    switcharray[2] = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    switcharray[3] = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    switcharray[4] = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    switcharray[5] = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    switcharray[6] = atoi(strtokIndx);     // convert this part to an integer
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    switcharray[7] = atoi(strtokIndx);     // convert this part to an integer
    flag = true;
}

//============

void showParsedData() {
    Serial.print("Number:");
    Serial.print(String(switcharray[0]));
    Serial.print(" ");
    Serial.print("Number:");
    Serial.print(String(switcharray[1]));
    Serial.print(" ");
    Serial.print("Number:");
    Serial.print(String(switcharray[2]));
    Serial.print(" ");
    Serial.print("Number:");
    Serial.print(String(switcharray[3]));
    Serial.print(" ");
    Serial.print("Number:");
    Serial.print(String(switcharray[4]));
    Serial.print(" ");
    Serial.print("Number:");
    Serial.print(String(switcharray[5]));
    Serial.print(" ");
    Serial.print("Number:");
    Serial.print(String(switcharray[6]));
    Serial.print(" ");
    Serial.print("Number:");
    Serial.println(String(switcharray[7]));
}

void SendSignal(){
  if(newData == true && flag == true){
    digitalWrite(CSPin, LOW);
    delay(10);
    for (int k = 0; k < 8; k++){
      SPI.transfer(switcharray[i]);
      i++;
      }
    if(i == 8){
      digitalWrite(CSPin, HIGH);
      i = 0;
      newData = false;
      flag = false;
      }
    }
  }
