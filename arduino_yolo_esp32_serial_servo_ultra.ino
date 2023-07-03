boolean flag1, flag2;
#define pingPin 4 // Trigger Pin of Ultrasonic Sensor
#define echoPin 15 // Echo Pin of Ultrasonic Sensor
const int ledPin = 2;
const int Buzzer_Pin = 5;
char userInput ;
float duration , distance ; 
float inches, cm;
#include <Servo.h>
#define SERVO_PIN 16
Servo servoMotor;
void setup() {
  // put your setup code here, to run once:
    Serial.begin(9600);
    pinMode(ledPin, OUTPUT);
    pinMode(Buzzer_Pin, OUTPUT);
    pinMode(pingPin, OUTPUT);
    pinMode(echoPin, INPUT);
    servoMotor.attach(SERVO_PIN);
}

void loop() {
   digitalWrite(pingPin, LOW);
   delayMicroseconds(2);
   digitalWrite(pingPin, HIGH);
   delayMicroseconds(10);
   digitalWrite(pingPin, LOW);
   duration = pulseIn(echoPin, HIGH);
   distance = (duration / 2) * 0.0343;
   //Serial.println("distance=");
   //Serial.println(distance);
   delay(100);
   // check if ledpin is low 
   if(flag1== true  and distance <= 20){
      digitalWrite(ledPin, LOW);
      digitalWrite(Buzzer_Pin, LOW);
      flag1=false;
      delay(1300);
        for (int pos = 0; pos <= 100; pos += 1) {
             servoMotor.write(pos);
             delay(10); }
        for (int pos = 100; pos >= 0; pos -= 1) {
             servoMotor.write(pos);
             delay(10);}
             digitalWrite(ledPin, LOW);
    }
  // put your main code here, to run repeatedly:
  if(Serial.available()>0){
    userInput = Serial.read();
    if (userInput == '1'){
      Serial.println("servo on");
      flag1=true;
      digitalWrite(ledPin, HIGH);
      digitalWrite(Buzzer_Pin, HIGH);


      
      }
    else if (userInput == '0'){
        Serial.println("servo off");

      }
    }

}
