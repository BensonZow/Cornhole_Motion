#include <Arduino.h>
#include <math.h>

// ============================================================================
// Omni BLDC firmware (Arduino Mega 2560)
// - Receives DA commands: "DA <distance_in> <angle_rad>"
// - Computes basic body command and omni inverse kinematics on Arduino
// - Drives 4x BLD-515C using SV (PWM+RC), F/R (DIR), EN, BK
// - Reports PG pulse counts for telemetry (no closed loop in this phase)
// ============================================================================

#define NUM_WHEELS 4
#define FL 0
#define FR 1
#define RL 2
#define RR 3

struct DriverPins {
  uint8_t svPwm;    // SV input via external RC low-pass
  uint8_t dir;      // F/R
  uint8_t en;       // EN
  uint8_t bk;       // BK (held inactive during phase-1)
  uint8_t pg;       // PG pulse input
};

static const DriverPins kPins[NUM_WHEELS] = {
  {2, 22, 30, 34, 18},  // FL
  {3, 23, 31, 35, 19},  // FR
  {5, 24, 32, 36, 20},  // RL
  {6, 25, 33, 37, 21},  // RR
};

// ---- Geometry globals (parameterized for diagonal wheel layout) ----
// Board dimensions and wheel positions are body-frame meters:
//   +X forward, +Y left, +Z up
// Wheel coordinates are at board diagonals.
const float BOARD_HALF_LENGTH_M = 0.6096f;  // 24 in / 2
const float BOARD_HALF_WIDTH_M  = 0.3048f;  // 12 in / 2
const float WHEEL_RADIUS_M      = 0.0485f;  // example 97 mm diameter wheel

const float WHEEL_POS_X[NUM_WHEELS] = {
  +BOARD_HALF_LENGTH_M,  // FL
  +BOARD_HALF_LENGTH_M,  // FR
  -BOARD_HALF_LENGTH_M,  // RL
  -BOARD_HALF_LENGTH_M   // RR
};

const float WHEEL_POS_Y[NUM_WHEELS] = {
  +BOARD_HALF_WIDTH_M,   // FL
  -BOARD_HALF_WIDTH_M,   // FR
  +BOARD_HALF_WIDTH_M,   // RL
  -BOARD_HALF_WIDTH_M    // RR
};

// ---- Basic command and output limits ----
const int MAX_WHEEL_CMD = 255;
const int MIN_EFFECTIVE_CMD = 25;

const float MAX_VX_MPS = 1.2f;
const float MAX_VY_MPS = 1.2f;
const float MAX_OMEGA_RADPS = 2.5f;

// distance (inches) and angle (rad) are mapped to body commands
const float DIST_TO_SPEED_GAIN = 0.020f;    // m/s per inch
const float ANGLE_TO_OMEGA_GAIN = 1.20f;    // rad/s per rad

const unsigned long CONTROL_PERIOD_MS = 20;     // 50 Hz update
const unsigned long SERIAL_TIMEOUT_MS = 300;    // stop if DA stream drops
const unsigned long TELEMETRY_PERIOD_MS = 200;  // PG report rate

// If this is wrong for your driver polarity, flip it in one place.
const bool BK_INACTIVE_LEVEL = LOW;
const bool EN_ACTIVE_LEVEL = HIGH;

volatile unsigned long pgCounts[NUM_WHEELS] = {0, 0, 0, 0};

float targetDistanceIn = 0.0f;
float targetAngleRad = 0.0f;
bool commandActive = false;

unsigned long lastDaMillis = 0;
unsigned long lastControlMillis = 0;
unsigned long lastTelemetryMillis = 0;

#define CMD_BUF_SIZE 96
char cmdBuf[CMD_BUF_SIZE];
uint8_t cmdLen = 0;

static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline int clampi(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void isrPgFL() { pgCounts[FL]++; }
void isrPgFR() { pgCounts[FR]++; }
void isrPgRL() { pgCounts[RL]++; }
void isrPgRR() { pgCounts[RR]++; }

void stopAllMotors() {
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    digitalWrite(kPins[i].en, !EN_ACTIVE_LEVEL);
    analogWrite(kPins[i].svPwm, 0);
  }
}

void applyWheelCommand(uint8_t index, int signedCmd) {
  if (index >= NUM_WHEELS) return;

  int cmd = clampi(signedCmd, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);
  bool dirForward = (cmd >= 0);
  int magnitude = abs(cmd);

  if (magnitude > 0 && magnitude < MIN_EFFECTIVE_CMD) {
    magnitude = MIN_EFFECTIVE_CMD;
  }

  // DIR and EN control
  digitalWrite(kPins[index].dir, dirForward ? HIGH : LOW);
  digitalWrite(kPins[index].en, (magnitude > 0) ? EN_ACTIVE_LEVEL : !EN_ACTIVE_LEVEL);

  // SV setpoint over PWM+RC
  analogWrite(kPins[index].svPwm, magnitude);
}

void computeAndApplyKinematics(float distanceIn, float angleRad) {
  // Basic proportional body command from distance/angle
  float vx = DIST_TO_SPEED_GAIN * distanceIn * cosf(angleRad);
  float vy = DIST_TO_SPEED_GAIN * distanceIn * sinf(angleRad);
  float omega = ANGLE_TO_OMEGA_GAIN * angleRad;

  vx = clampf(vx, -MAX_VX_MPS, MAX_VX_MPS);
  vy = clampf(vy, -MAX_VY_MPS, MAX_VY_MPS);
  omega = clampf(omega, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS);

  // Basic omni inverse kinematics for diagonal wheel placement.
  // K term uses representative diagonal geometry distance.
  const float kGeom = BOARD_HALF_LENGTH_M + BOARD_HALF_WIDTH_M;

  float wheelRadPerSec[NUM_WHEELS];
  wheelRadPerSec[FL] = (vx - vy - kGeom * omega) / WHEEL_RADIUS_M;
  wheelRadPerSec[FR] = (vx + vy + kGeom * omega) / WHEEL_RADIUS_M;
  wheelRadPerSec[RL] = (vx + vy - kGeom * omega) / WHEEL_RADIUS_M;
  wheelRadPerSec[RR] = (vx - vy + kGeom * omega) / WHEEL_RADIUS_M;

  // Normalize to PWM command range.
  float maxAbs = 0.0f;
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    float a = fabsf(wheelRadPerSec[i]);
    if (a > maxAbs) maxAbs = a;
  }

  int wheelCmd[NUM_WHEELS] = {0, 0, 0, 0};
  if (maxAbs > 1e-4f) {
    const float scale = (maxAbs > (float)MAX_WHEEL_CMD)
                            ? ((float)MAX_WHEEL_CMD / maxAbs)
                            : 1.0f;
    for (uint8_t i = 0; i < NUM_WHEELS; i++) {
      wheelCmd[i] = (int)roundf(wheelRadPerSec[i] * scale);
    }
  }

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    applyWheelCommand(i, wheelCmd[i]);
  }
}

void processCommand(const char *cmd) {
  while (*cmd == ' ') cmd++;
  if (*cmd == '\0') return;

  if (strcmp(cmd, "STOP") == 0) {
    commandActive = false;
    stopAllMotors();
    Serial.println(F("ACK STOP"));
    return;
  }

  if (strcmp(cmd, "PING") == 0) {
    Serial.println(F("PONG"));
    return;
  }

  if (strcmp(cmd, "HELP") == 0) {
    Serial.println(F("Commands: DA <distance_in> <angle_rad> | STOP | PING"));
    return;
  }

  if (strncmp(cmd, "DA ", 3) == 0) {
    float d = 0.0f;
    float a = 0.0f;
    int parsed = sscanf(cmd + 3, "%f %f", &d, &a);
    if (parsed == 2 && isfinite(d) && isfinite(a)) {
      targetDistanceIn = d;
      targetAngleRad = a;
      lastDaMillis = millis();
      commandActive = true;
      Serial.println(F("ACK DA"));
    } else {
      Serial.println(F("ERR BAD_DA"));
    }
    return;
  }

  Serial.print(F("ERR UNKNOWN "));
  Serial.println(cmd);
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    pinMode(kPins[i].svPwm, OUTPUT);
    pinMode(kPins[i].dir, OUTPUT);
    pinMode(kPins[i].en, OUTPUT);
    pinMode(kPins[i].bk, OUTPUT);
    pinMode(kPins[i].pg, INPUT_PULLUP);

    digitalWrite(kPins[i].bk, BK_INACTIVE_LEVEL);
  }

  // PG interrupt setup (Mega supports external interrupts on these pins).
  attachInterrupt(digitalPinToInterrupt(kPins[FL].pg), isrPgFL, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[FR].pg), isrPgFR, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[RL].pg), isrPgRL, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[RR].pg), isrPgRR, RISING);

  stopAllMotors();

  Serial.println(F("Omni BLDC firmware ready"));
  Serial.println(F("Protocol: DA <distance_in> <angle_rad> | STOP | PING"));
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen > 0) {
        cmdBuf[cmdLen] = '\0';
        processCommand(cmdBuf);
        cmdLen = 0;
      }
    } else if (cmdLen < CMD_BUF_SIZE - 1) {
      cmdBuf[cmdLen++] = c;
    } else {
      cmdLen = 0;  // drop oversized line
      Serial.println(F("ERR CMD_OVERFLOW"));
    }
  }

  const unsigned long now = millis();

  if (commandActive && (now - lastControlMillis >= CONTROL_PERIOD_MS)) {
    lastControlMillis = now;
    computeAndApplyKinematics(targetDistanceIn, targetAngleRad);
  }

  if (commandActive && (now - lastDaMillis > SERIAL_TIMEOUT_MS)) {
    commandActive = false;
    stopAllMotors();
    Serial.println(F("WATCHDOG STOP"));
  }

  if (now - lastTelemetryMillis >= TELEMETRY_PERIOD_MS) {
    lastTelemetryMillis = now;

    noInterrupts();
    unsigned long fl = pgCounts[FL];
    unsigned long fr = pgCounts[FR];
    unsigned long rl = pgCounts[RL];
    unsigned long rr = pgCounts[RR];
    pgCounts[FL] = pgCounts[FR] = pgCounts[RL] = pgCounts[RR] = 0;
    interrupts();

    Serial.print(F("PG "));
    Serial.print(fl); Serial.print(' ');
    Serial.print(fr); Serial.print(' ');
    Serial.print(rl); Serial.print(' ');
    Serial.println(rr);
  }
}
