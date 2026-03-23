#include <Arduino.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Omni BLDC firmware (Arduino Mega 2560)
// - Serial (IDLE only): one line "distance_in,angle_rad" starts a move; lines
//   ignored while MOVING until complete or timeout.
// - angle_rad: direction of translation in the starting robot frame (+X forward,
//   +Y left, CCW positive). Omni inverse kinematics maps (vx, vy, 0) to wheels.
// - Scalar distance PID: error = distance_goal_m - (x_hat*ux + y_hat*uy) in
//   mission frame; PG + F/R -> signed wheel ds -> forward kinematics -> pose.
//
// BLD-515C control notes (vs literal datasheet wording):
// - F/R: reverse = tie to GND; forward = leave open (not driven). We emulate
//   "open" with pinMode INPUT_PULLUP (~20-50k to Vcc), not a true hi-Z—verify
//   on hardware. Never drive OUTPUT HIGH on F/R (would source 5V).
// - BK: brake asserted when line grounded; released when not. Same pattern as
//   F/R: INPUT_PULLUP = not asserting GND, OUTPUT LOW = grounded.
// - EN: manual says "EN grounded = run"; this sketch keeps EN_ACTIVE_LEVEL HIGH
//   when enabling (your wiring). Change EN_ACTIVE_LEVEL if your board inverts.
// - When switching F/R, EN is turned off first, short delay, then F/R updated
//   (per manual), then EN restored if still commanding motion.
// ============================================================================

#define NUM_WHEELS 4
#define FL 0
#define FR 1
#define RL 2
#define RR 3

// Set 1 to print periodic PG counts on Serial (may flood during MOVING).
#define DEBUG_TELEMETRY 0

struct DriverPins {
  uint8_t svPwm;
  uint8_t frPin;  // F/R: INPUT_PULLUP forward (open), OUTPUT LOW reverse (GND)
  uint8_t en;
  uint8_t bkPin;  // BK: same pin-mode convention as F/R
  uint8_t pg;
};

static const DriverPins kPins[NUM_WHEELS] = {
  {2, 22, 30, 34, 18},  // FL
  {3, 23, 31, 35, 19},  // FR
  {5, 24, 32, 36, 20},  // RL
  {6, 25, 33, 37, 21},  // RR
};

// Board frame: +X forward, +Y left, +Z up (wheel layout diagonals).
const float BOARD_HALF_LENGTH_M = 0.6096f;
const float BOARD_HALF_WIDTH_M = 0.3048f;
const float WHEEL_RADIUS_M = 0.0485f;

// Same kGeom as inverse kinematics (half diagonal span for yaw coupling).
const float kGeom = BOARD_HALF_LENGTH_M + BOARD_HALF_WIDTH_M;

// ---- Motor / encoder (PG: 3 * pole_pairs pulses per mechanical revolution) ----
// Set from your BLDC motor datasheet (pole pairs P).
const unsigned MOTOR_POLE_PAIRS = 7;

// ---- Motion & PID (tune on hardware) ----
const float INCH_TO_M = 0.0254f;
const float MAX_VX_MPS = 1.2f;
const float MAX_VY_MPS = 1.2f;
// Scalar speed cap for translation: v_cmd clamped to +/- this (m/s).
const float MAX_TRANSLATION_SPEED_MPS =
    (MAX_VX_MPS < MAX_VY_MPS) ? MAX_VX_MPS : MAX_VY_MPS;

const float DIST_TOL_M = 0.012f;     // ~1/2 inch
const float VEL_EPS_MPS = 0.04f;
const uint8_t DONE_HOLD_CYCLES = 5;
const unsigned long MOVE_TIMEOUT_MS = 60000UL;

// Distance PID (single axis: meters error -> m/s command).
const float PID_KP = 2.0f;
const float PID_KI = 0.15f;
const float PID_KD = 0.05f;
const float PID_I_CLAMP = 0.6f;  // m/s equivalent from integral

const int MAX_WHEEL_CMD = 255;
const int MIN_EFFECTIVE_CMD = 25;

const unsigned long CONTROL_PERIOD_MS = 20;
#if DEBUG_TELEMETRY
const unsigned long TELEMETRY_PERIOD_MS = 200;
#endif

const bool EN_ACTIVE_LEVEL = HIGH;

// When true, stopAllMotors() and zero-speed commands assert brake (BK grounded).
const bool BRAKE_WHEN_STOPPED = true;

// Delay after disabling EN before changing F/R (datasheet sequencing).
const uint16_t FR_EN_OFF_DELAY_US = 5000;

// ---------------------------------------------------------------------------
enum MoveState : uint8_t { STATE_IDLE = 0, STATE_MOVING = 1 };

volatile unsigned long pgPulseTotal[NUM_WHEELS] = {0, 0, 0, 0};
unsigned long pgSnapPrev[NUM_WHEELS] = {0, 0, 0, 0};

float missionX_m = 0.0f;
float missionY_m = 0.0f;
float missionTheta_rad = 0.0f;

float distanceGoal_m = 0.0f;
float ux = 1.0f;
float uy = 0.0f;

float pidIntegral = 0.0f;
float pidLastErr = 0.0f;
unsigned long pidLastUs = 0;

MoveState moveState = STATE_IDLE;
unsigned long lastControlMillis = 0;
unsigned long moveStartMillis = 0;
uint8_t doneStableCycles = 0;

#if DEBUG_TELEMETRY
unsigned long lastTelemetryMillis = 0;
#endif

#define CMD_BUF_SIZE 96
char cmdBuf[CMD_BUF_SIZE];
uint8_t cmdLen = 0;

// Last commanded F/R sense per wheel (forward = INPUT_PULLUP / reverse = GND).
static bool lastDirForward[NUM_WHEELS] = {true, true, true, true};

static inline void driverInputRelease(uint8_t pin) { pinMode(pin, INPUT_PULLUP); }

static inline void driverInputGround(uint8_t pin) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
}

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

void isrPgFL() { pgPulseTotal[FL]++; }
void isrPgFR() { pgPulseTotal[FR]++; }
void isrPgRL() { pgPulseTotal[RL]++; }
void isrPgRR() { pgPulseTotal[RR]++; }

void stopAllMotors() {
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    digitalWrite(kPins[i].en, !EN_ACTIVE_LEVEL);
    analogWrite(kPins[i].svPwm, 0);
    if (BRAKE_WHEN_STOPPED) {
      driverInputGround(kPins[i].bkPin);
    } else {
      driverInputRelease(kPins[i].bkPin);
    }
  }
}

// INPUT_PULLUP reads HIGH (forward); OUTPUT LOW reads LOW (reverse).
static inline bool wheelCommandedForward(uint8_t index) {
  return digitalRead(kPins[index].frPin) == HIGH;
}

// Meters of wheel centerline travel per PG pulse (signed by wheel rotation).
static inline float metersPerPulse() {
  const float pulsesPerRev = 3.0f * (float)MOTOR_POLE_PAIRS;
  return (2.0f * (float)M_PI * WHEEL_RADIUS_M) / pulsesPerRev;
}

void applyWheelCommand(uint8_t index, int signedCmd) {
  if (index >= NUM_WHEELS) return;

  int cmd = clampi(signedCmd, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);
  bool dirForward = (cmd >= 0);
  int magnitude = abs(cmd);

  if (magnitude > 0 && magnitude < MIN_EFFECTIVE_CMD) {
    magnitude = MIN_EFFECTIVE_CMD;
  }

  if (dirForward != lastDirForward[index]) {
    digitalWrite(kPins[index].en, !EN_ACTIVE_LEVEL);
    delayMicroseconds(FR_EN_OFF_DELAY_US);
    if (dirForward) {
      driverInputRelease(kPins[index].frPin);
    } else {
      driverInputGround(kPins[index].frPin);
    }
    lastDirForward[index] = dirForward;
  }

  if (magnitude > 0) {
    driverInputRelease(kPins[index].bkPin);
  } else {
    if (BRAKE_WHEN_STOPPED) {
      driverInputGround(kPins[index].bkPin);
    } else {
      driverInputRelease(kPins[index].bkPin);
    }
  }

  digitalWrite(kPins[index].en, (magnitude > 0) ? EN_ACTIVE_LEVEL : !EN_ACTIVE_LEVEL);
  analogWrite(kPins[index].svPwm, magnitude);
}

void applyInverseKinematics(float vx, float vy, float omega) {
  vx = clampf(vx, -MAX_VX_MPS, MAX_VX_MPS);
  vy = clampf(vy, -MAX_VY_MPS, MAX_VY_MPS);

  float wheelRadPerSec[NUM_WHEELS];
  wheelRadPerSec[FL] = (vx - vy - kGeom * omega) / WHEEL_RADIUS_M;
  wheelRadPerSec[FR] = (vx + vy + kGeom * omega) / WHEEL_RADIUS_M;
  wheelRadPerSec[RL] = (vx + vy - kGeom * omega) / WHEEL_RADIUS_M;
  wheelRadPerSec[RR] = (vx - vy + kGeom * omega) / WHEEL_RADIUS_M;

  float maxAbs = 0.0f;
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    float a = fabsf(wheelRadPerSec[i]);
    if (a > maxAbs) maxAbs = a;
  }

  int wheelCmd[NUM_WHEELS] = {0, 0, 0, 0};
  if (maxAbs > 1e-4f) {
    const float scale =
        (maxAbs > (float)MAX_WHEEL_CMD) ? ((float)MAX_WHEEL_CMD / maxAbs) : 1.0f;
    for (uint8_t i = 0; i < NUM_WHEELS; i++) {
      wheelCmd[i] = (int)roundf(wheelRadPerSec[i] * scale);
    }
  }

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    applyWheelCommand(i, wheelCmd[i]);
  }
}

// Forward kinematics: wheel linear displacements (m) -> body deltas -> mission frame.
void integrateOdometry(float dsWheel[NUM_WHEELS]) {
  const float dxBody =
      (dsWheel[FL] + dsWheel[FR] + dsWheel[RL] + dsWheel[RR]) * 0.25f;
  const float dyBody =
      (dsWheel[FR] + dsWheel[RL] - dsWheel[FL] - dsWheel[RR]) * 0.25f;
  const float dTheta = (dsWheel[FR] - dsWheel[RL]) / (2.0f * kGeom);

  const float c = cosf(missionTheta_rad);
  const float s = sinf(missionTheta_rad);
  missionX_m += c * dxBody - s * dyBody;
  missionY_m += s * dxBody + c * dyBody;
  missionTheta_rad += dTheta;
}

void snapshotPgBaseline() {
  noInterrupts();
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    pgSnapPrev[i] = pgPulseTotal[i];
  }
  interrupts();
}

// Returns false on timeout (caller should abort move).
bool runControlTick() {
  const unsigned long now = millis();
  if (now - moveStartMillis > MOVE_TIMEOUT_MS) {
    return false;
  }

  float dsWheel[NUM_WHEELS];
  const float mpp = metersPerPulse();

  noInterrupts();
  unsigned long totals[NUM_WHEELS];
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    totals[i] = pgPulseTotal[i];
  }
  interrupts();

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    unsigned long delta = totals[i] - pgSnapPrev[i];
    pgSnapPrev[i] = totals[i];
    const float sign = wheelCommandedForward(i) ? 1.0f : -1.0f;
    dsWheel[i] = sign * mpp * (float)delta;
  }

  integrateOdometry(dsWheel);

  const float distTraveled = missionX_m * ux + missionY_m * uy;
  const float err = distanceGoal_m - distTraveled;

  const unsigned long nowUs = micros();
  float dt = (pidLastUs == 0) ? ((float)CONTROL_PERIOD_MS * 0.001f)
                              : (float)(nowUs - pidLastUs) * 1e-6f;
  if (dt < 1e-4f) dt = (float)CONTROL_PERIOD_MS * 0.001f;
  pidLastUs = nowUs;

  const float dErr = (err - pidLastErr) / dt;
  pidLastErr = err;

  pidIntegral += PID_KI * err * dt;
  pidIntegral = clampf(pidIntegral, -PID_I_CLAMP, PID_I_CLAMP);

  float vCmd = PID_KP * err + pidIntegral + PID_KD * dErr;
  vCmd = clampf(vCmd, -MAX_TRANSLATION_SPEED_MPS, MAX_TRANSLATION_SPEED_MPS);

  const float vx = vCmd * ux;
  const float vy = vCmd * uy;
  applyInverseKinematics(vx, vy, 0.0f);

  if (fabsf(err) < DIST_TOL_M && fabsf(vCmd) < VEL_EPS_MPS) {
    if (doneStableCycles < 255) doneStableCycles++;
  } else {
    doneStableCycles = 0;
  }

  if (doneStableCycles >= DONE_HOLD_CYCLES) {
    stopAllMotors();
    moveState = STATE_IDLE;
    pidIntegral = 0.0f;
    pidLastErr = 0.0f;
    pidLastUs = 0;
    Serial.println(F("ACK DONE"));
  }

  return true;
}

static bool parseCommaPair(const char *line, float *outD, float *outA) {
  if (strchr(line, ',') == nullptr) return false;
  float d = 0.0f;
  float a = 0.0f;
  if (sscanf(line, "%f,%f", &d, &a) != 2) return false;
  if (!isfinite(d) || !isfinite(a)) return false;
  *outD = d;
  *outA = a;
  return true;
}

void tryProcessIdleLine() {
  if (cmdLen == 0) return;
  cmdBuf[cmdLen] = '\0';

  float dIn = 0.0f;
  float aRad = 0.0f;
  if (!parseCommaPair(cmdBuf, &dIn, &aRad)) {
    Serial.println(F("ERR BAD_LINE"));
    cmdLen = 0;
    return;
  }

  distanceGoal_m = dIn * INCH_TO_M;
  ux = cosf(aRad);
  uy = sinf(aRad);

  missionX_m = missionY_m = 0.0f;
  missionTheta_rad = 0.0f;

  pidIntegral = 0.0f;
  pidLastErr = 0.0f;
  pidLastUs = 0;
  doneStableCycles = 0;

  snapshotPgBaseline();
  moveStartMillis = millis();
  moveState = STATE_MOVING;
  // Run first control tick on the next loop iteration without waiting a full period.
  lastControlMillis = moveStartMillis - CONTROL_PERIOD_MS;

  Serial.println(F("ACK MOVE"));
  cmdLen = 0;
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    pinMode(kPins[i].svPwm, OUTPUT);
    analogWrite(kPins[i].svPwm, 0);
    pinMode(kPins[i].en, OUTPUT);
    digitalWrite(kPins[i].en, !EN_ACTIVE_LEVEL);
    driverInputRelease(kPins[i].frPin);
    if (BRAKE_WHEN_STOPPED) {
      driverInputGround(kPins[i].bkPin);
    } else {
      driverInputRelease(kPins[i].bkPin);
    }
    pinMode(kPins[i].pg, INPUT_PULLUP);
  }

  attachInterrupt(digitalPinToInterrupt(kPins[FL].pg), isrPgFL, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[FR].pg), isrPgFR, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[RL].pg), isrPgRL, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[RR].pg), isrPgRR, RISING);

  stopAllMotors();

  Serial.println(F("Omni BLDC: send distance_in,angle_rad then newline (IDLE only)"));
}

void loop() {
  // Serial: buffer full lines in IDLE; discard lines while MOVING.
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (moveState == STATE_MOVING) {
      if (c == '\n' || c == '\r') {
        /* discarded */
      }
      continue;
    }
    if (c == '\n' || c == '\r') {
      tryProcessIdleLine();
    } else if (cmdLen < CMD_BUF_SIZE - 1) {
      cmdBuf[cmdLen++] = c;
    } else {
      cmdLen = 0;
      Serial.println(F("ERR CMD_OVERFLOW"));
    }
  }

  const unsigned long now = millis();

  if (moveState == STATE_MOVING && (now - lastControlMillis >= CONTROL_PERIOD_MS)) {
    lastControlMillis = now;
    if (!runControlTick()) {
      stopAllMotors();
      moveState = STATE_IDLE;
      pidIntegral = 0.0f;
      pidLastErr = 0.0f;
      pidLastUs = 0;
      Serial.println(F("ERR TIMEOUT"));
    }
  }

#if DEBUG_TELEMETRY
  if (moveState == STATE_IDLE && (now - lastTelemetryMillis >= TELEMETRY_PERIOD_MS)) {
    lastTelemetryMillis = now;
    noInterrupts();
    unsigned long t0 = pgPulseTotal[FL];
    unsigned long t1 = pgPulseTotal[FR];
    unsigned long t2 = pgPulseTotal[RL];
    unsigned long t3 = pgPulseTotal[RR];
    interrupts();
    Serial.print(F("PG "));
    Serial.print(t0);
    Serial.print(' ');
    Serial.print(t1);
    Serial.print(' ');
    Serial.print(t2);
    Serial.print(' ');
    Serial.println(t3);
  }
#endif
}
