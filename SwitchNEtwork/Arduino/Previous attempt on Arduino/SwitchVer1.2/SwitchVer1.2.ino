boolean newData = false;
boolean flag = false;
int i = 0;
byte switcharray;
byte StartMark = 0x3C;
byte EndMark = 0x3E;
static boolean recvProgress = false;
static byte ndx = 0;
const byte numBytes = 32;
byte receivedBytes[numBytes];

void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);
Serial.setTimeout(200);
}

void loop(){
  ReceivedSig();
  SendSig();
  }

void ReceiveOne(){
  
while(Serial.available()>=0 && newData == false){
  switcharray = Serial.read();
  if (recvProgress ==true){
      if(switcharray !=EndMark){
        receivedBytes[0] = switcharray;
        ndx++;
        if(ndx >=numBytes){
          ndx = numBytes- 1;
          }
        }
      else{
        receivedBytes[ndx] = '\0'; // terminate the string
        recvProgress = false;
        ndx = 0;
        newData = true;
        }
    }else if(switcharray ==StartMark){
      recvProgress =true;
      }


  }
  
  }

void ReceivedSig() {
  // put your main code here, to run repeatedly:
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
        ndx = 0;
        newData = true;
        parseData();
        }
    }else if(switcharray ==StartMark){
      recvProgress =true;
      }


  }

}

void ParseData(){
  
  }

void SendSig(){

  if(newData == true){
    Serial.println(String(receivedBytes[0]));
    Serial.flush();
    }
  }
 
