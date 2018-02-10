#include <Servo.h>

Servo servo;
#define PIN_SERVO 10

#define PIN_MOTOR_RIGHT_DN 9
#define PIN_MOTOR_RIGHT_UP 8
#define PIN_MOTOR_RIGHT_SPEED 11

unsigned char RightMotor[3] = 
  {PIN_MOTOR_RIGHT_UP, PIN_MOTOR_RIGHT_DN, PIN_MOTOR_RIGHT_SPEED};

int agl = 125;
int spd = 0;
int whl = 0;

int index =0;

int servo_up = 160; //максимальный поворот влево
int servo_dn = 90;  //максимальный поворот вправо
int whl_up = 2;     //0-назад, 1- стоп, 2 - вперед

void setup() 
{
  pinMode (PIN_MOTOR_RIGHT_UP, OUTPUT);
  pinMode (PIN_MOTOR_RIGHT_DN, OUTPUT);
  servo.attach(PIN_SERVO);

  Serial.begin(9600);
  
  Serial.print("start;");
}

//https://www.arduino.cc/en/Reference/ASCIIchart
//33   space
//126  ~

void loop()
{
  if (Serial.available() > 0) 
  {  
    byte iByte = Serial.read();
    switch (index)
    {
      case 0:
        agl = iByte;
        break;
      case 1:
        spd = iByte;
        break;
      case 2:
        whl = iByte;
        break;
    }
    
    index++;
  }

  if (index > 2)
  {
    setUpMotors(agl,spd,whl);
    index = 0;
  }
     
}

void setUpMotors(int v_agl, int v_spd, int v_whl)
{
  bool w_one = false;
  bool w_two = false;

  if (v_agl>servo_up)
  {
    v_agl = servo_up;  
  }

  if (v_agl<servo_dn)
  {
    v_agl = servo_dn;  
  }

  if (v_whl > whl_up)
  {
    v_whl = whl_up;  
  }

  //задаём угол
  servo.write(v_agl);

  delay(100);
  
  //проверяем напр движения  
  switch (v_whl)
  {
    case 0:
      w_one = false;
      w_two = true;
    break;
    case 1:
      w_one = false;
      w_two = false;
      v_spd = 0;
    break;
    case 2:
      w_one = true;
      w_two = false;
    break;  
  }
  
  digitalWrite(RightMotor[0], w_one);
  digitalWrite(RightMotor[1], w_two);
  analogWrite(RightMotor[2], v_spd);

  //debug
  Serial.print("agl:");
  Serial.print(v_agl);
  Serial.print(";spd:");
  Serial.print(v_spd);
  Serial.print(";whl:");
  Serial.println(v_whl);
  
}

