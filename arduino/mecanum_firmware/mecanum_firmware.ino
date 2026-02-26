#include "commands.h"

// ===========================================================================
// Pin Definitions
// ===========================================================================

#define NUM_MOTORS 4

// Motor indices
#define FL 0  // Front Left
#define FR 1  // Front Right
#define RL 2  // Rear Left
#define RR 3  // Rear Right

struct MotorPins {
  uint8_t enable;  // PWM pin
  uint8_t in1;     // direction pin A
  uint8_t in2;     // direction pin B
};

// HW-095 #1: Front Left (ENA side), Front Right (ENB side)
// HW-095 #2: Rear Left  (ENA side), Rear Right  (ENB side)
static const MotorPins motorPins[NUM_MOTORS] = {
  {6,  26, 27},  // FL — HW-095 #1 channel A
  {7,  28, 29},  // FR — HW-095 #1 channel B
  {8,  30, 31},  // RL — HW-095 #2 channel A
  {9,  32, 33},  // RR — HW-095 #2 channel B
};

struct EncoderPins {
  uint8_t chA;  // interrupt pin
  uint8_t chB;  // direction-sense pin
};

static const EncoderPins encoderPins[NUM_MOTORS] = {
  {2,  22},  // FL — INT0
  {3,  23},  // FR — INT1
  {18, 24},  // RL — INT5
  {19, 25},  // RR — INT4
};

// ===========================================================================
// Encoder State (volatile — modified in ISRs)
// ===========================================================================

volatile long encoderTicks[NUM_MOTORS] = {0, 0, 0, 0};

static void isrFL() {
  encoderTicks[FL] += digitalRead(encoderPins[FL].chB) ? -1 : 1;
}
static void isrFR() {
  encoderTicks[FR] += digitalRead(encoderPins[FR].chB) ? -1 : 1;
}
static void isrRL() {
  encoderTicks[RL] += digitalRead(encoderPins[RL].chB) ? -1 : 1;
}
static void isrRR() {
  encoderTicks[RR] += digitalRead(encoderPins[RR].chB) ? -1 : 1;
}

// ===========================================================================
// Motor Control
// ===========================================================================

// pwmVal: -255..+255  (positive = forward, negative = reverse)
void setMotor(uint8_t index, int16_t pwmVal) {
  if (index >= NUM_MOTORS) return;

  const MotorPins &mp = motorPins[index];

  if (pwmVal > 0) {
    digitalWrite(mp.in1, HIGH);
    digitalWrite(mp.in2, LOW);
  } else if (pwmVal < 0) {
    digitalWrite(mp.in1, LOW);
    digitalWrite(mp.in2, HIGH);
  } else {
    digitalWrite(mp.in1, LOW);
    digitalWrite(mp.in2, LOW);
  }

  analogWrite(mp.enable, constrain(abs(pwmVal), 0, 255));
}

void stopAllMotors() {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    setMotor(i, 0);
  }
}

// ===========================================================================
// Pulse Execution
// ===========================================================================

int pulsePWM      = DEFAULT_PULSE_PWM;
int pulseDuration = DEFAULT_PULSE_DURATION;

void executePulse(Direction dir) {
  if (dir >= DIR_COUNT) return;

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    setMotor(i, KINEMATIC_TABLE[dir][i] * pulsePWM);
  }

  delay(pulseDuration);
  stopAllMotors();

  Serial.print(F("Pulse "));
  Serial.print(pulseDuration);
  Serial.println(F("ms done"));
}

// ===========================================================================
// Watchdog for raw motor commands
// ===========================================================================

#define WATCHDOG_TIMEOUT_MS 500

unsigned long lastCommandTime = 0;
bool rawMotorsActive = false;

// ===========================================================================
// Serial Command Parser
// ===========================================================================

#define CMD_BUF_SIZE 64
char cmdBuf[CMD_BUF_SIZE];
uint8_t cmdLen = 0;

void processCommand(const char *cmd) {
  lastCommandTime = millis();

  // --- Pulse movement commands ---
  if (strcmp(cmd, "F") == 0)        { executePulse(DIR_FORWARD);      return; }
  if (strcmp(cmd, "B") == 0)        { executePulse(DIR_BACKWARD);     return; }
  if (strcmp(cmd, "L") == 0)        { executePulse(DIR_STRAFE_LEFT);  return; }
  if (strcmp(cmd, "R") == 0)        { executePulse(DIR_STRAFE_RIGHT); return; }
  if (strcmp(cmd, "FL") == 0)       { executePulse(DIR_DIAG_FL);      return; }
  if (strcmp(cmd, "FR") == 0)       { executePulse(DIR_DIAG_FR);      return; }
  if (strcmp(cmd, "BL") == 0)       { executePulse(DIR_DIAG_BL);      return; }
  if (strcmp(cmd, "BR") == 0)       { executePulse(DIR_DIAG_BR);      return; }
  if (strcmp(cmd, "CW") == 0)       { executePulse(DIR_ROTATE_CW);    return; }
  if (strcmp(cmd, "CCW") == 0)      { executePulse(DIR_ROTATE_CCW);   return; }

  // --- Emergency stop ---
  if (strcmp(cmd, "s") == 0) {
    stopAllMotors();
    rawMotorsActive = false;
    Serial.println(F("STOP"));
    return;
  }

  // --- Read encoders ---
  if (strcmp(cmd, "e") == 0) {
    noInterrupts();
    long t0 = encoderTicks[FL];
    long t1 = encoderTicks[FR];
    long t2 = encoderTicks[RL];
    long t3 = encoderTicks[RR];
    interrupts();
    Serial.print(t0); Serial.print(' ');
    Serial.print(t1); Serial.print(' ');
    Serial.print(t2); Serial.print(' ');
    Serial.println(t3);
    return;
  }

  // --- Reset encoders ---
  if (strcmp(cmd, "r") == 0) {
    noInterrupts();
    for (uint8_t i = 0; i < NUM_MOTORS; i++) encoderTicks[i] = 0;
    interrupts();
    Serial.println(F("Encoders reset"));
    return;
  }

  // --- Set pulse PWM power: p <value> ---
  if (cmd[0] == 'p' && cmd[1] == ' ') {
    int val = atoi(cmd + 2);
    pulsePWM = constrain(val, 0, 255);
    Serial.print(F("Pulse PWM = "));
    Serial.println(pulsePWM);
    return;
  }

  // --- Set pulse duration: t <ms> ---
  if (cmd[0] == 't' && cmd[1] == ' ') {
    int val = atoi(cmd + 2);
    pulseDuration = constrain(val, 50, 5000);
    Serial.print(F("Pulse duration = "));
    Serial.print(pulseDuration);
    Serial.println(F("ms"));
    return;
  }

  // --- Raw motor command: m <FL> <FR> <RL> <RR> ---
  if (cmd[0] == 'm' && cmd[1] == ' ') {
    int vals[NUM_MOTORS] = {0};
    const char *p = cmd + 2;
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
      while (*p == ' ') p++;
      vals[i] = atoi(p);
      while (*p && *p != ' ') p++;
    }
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
      setMotor(i, constrain(vals[i], -255, 255));
    }
    rawMotorsActive = true;
    Serial.println(F("OK"));
    return;
  }

  Serial.print(F("Unknown: "));
  Serial.println(cmd);
}

// ===========================================================================
// Setup & Loop
// ===========================================================================

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    pinMode(motorPins[i].enable, OUTPUT);
    pinMode(motorPins[i].in1,    OUTPUT);
    pinMode(motorPins[i].in2,    OUTPUT);
  }

  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    pinMode(encoderPins[i].chA, INPUT_PULLUP);
    pinMode(encoderPins[i].chB, INPUT_PULLUP);
  }

  attachInterrupt(digitalPinToInterrupt(encoderPins[FL].chA), isrFL, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPins[FR].chA), isrFR, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPins[RL].chA), isrRL, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderPins[RR].chA), isrRR, RISING);

  stopAllMotors();

  Serial.println(F("Mecanum firmware ready"));
  Serial.println(F("Commands: F B L R FL FR BL BR CW CCW | m/e/r/s/p/t"));
}

void loop() {
  // Read serial input line-by-line
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen > 0) {
        cmdBuf[cmdLen] = '\0';
        processCommand(cmdBuf);
        cmdLen = 0;
      }
    } else if (cmdLen < CMD_BUF_SIZE - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }

  // Watchdog: auto-stop raw motor commands after timeout
  if (rawMotorsActive && (millis() - lastCommandTime > WATCHDOG_TIMEOUT_MS)) {
    stopAllMotors();
    rawMotorsActive = false;
    Serial.println(F("Watchdog stop"));
  }
}
