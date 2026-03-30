// Omni Mecanum / omni-wheel robot: four BLDC drivers (BLD-515C-style), PWM speed,
// direction (F/R), enable, brake, and PG (pulse) feedback per wheel.
#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Omni BLDC firmware (Arduino Mega 2560)
//
// Control flow (high level):
// 1. Host sends one line: "distance_in,angle_rad" then newline (only in IDLE).
// 2. Firmware resets mission pose, snapshots PG counters, runs closed-loop moves.
// 3. Every CONTROL_PERIOD_MS: read new PG pulses -> per-wheel ds -> forward
//    kinematics updates (missionX_m, missionY_m). Scalar PID on distance along
//    (ux,uy) commands body velocity (vx,vy); inverse kinematics -> wheel PWM.
// 4. When error and velocity are small for DONE_HOLD_CYCLES: stop, print ODO
//    line (inches per wheel + total along path), then "ACK DONE".
//
// Coordinate frames:
// - Body / board: +X forward, +Y left, +Z up (see wheel indices FL, FR, RL, RR).
// - Mission: fixed at move start; (ux, uy) is unit vector in body XY for the
//   commanded translation; angle_rad sets ux=cos(a), uy=sin(a) at command time.
//
// BLD-515C control notes (vs literal datasheet wording):
// - F/R and BK: push-pull OUTPUT. OUTPUT LOW = line grounded (reverse / brake
//   on). OUTPUT HIGH = line not at GND (forward / brake off).
// - Caveat: manual forward is "do not connect"; driving HIGH sources ~5V and is
//   not a true open circuit—only use if the driver inputs tolerate logic high.
// - BK: brake asserted when line grounded (OUTPUT LOW); released when HIGH.
// - EN: manual says "EN grounded = run"; this sketch keeps EN_ACTIVE_LEVEL HIGH
//   when enabling (your wiring). Change EN_ACTIVE_LEVEL if your board inverts.
// - When switching F/R, EN is turned off first, short delay, then F/R updated
//   (per manual), then EN restored if still commanding motion.
// ============================================================================

// Wheel indices: front-left, front-right, rear-left, rear-right (board frame).
#define NUM_WHEELS 4
#define FL 0
#define FR 1
#define RL 2
#define RR 3

// Status outputs (Mega digital pins; not used by motor wiring in this design)
const uint8_t STATUS_PIN_ARMED = 7;   // HIGH while closed-loop move is running (STATE_MOVING)
const uint8_t STATUS_PIN_DRIVE = 8;   // HIGH while any wheel has non-zero SV PWM duty

// 1 = periodic raw PG pulse totals on Serial when IDLE (fast rate; optional).
#define DEBUG_TELEMETRY 0

// PG totals + OUT (PWM/DIR/EN snapshot) on Serial at this interval (IDLE and MOVING).
const unsigned long OUT_TELEMETRY_PERIOD_MS = 2000;

// Pin bundle per wheel: must match your Mega wiring.
struct DriverPins {
  uint8_t svPwm;   // Speed PWM to driver (0..255 typical).
  uint8_t frPin;   // F/R: OUTPUT HIGH = forward, OUTPUT LOW = reverse (GND).
  uint8_t en;      // Enable: active level set by EN_ACTIVE_LEVEL.
  uint8_t bkPin;   // Brake: OUTPUT HIGH = brake off, OUTPUT LOW = brake on (GND).
  uint8_t pg;      // Pulse generator / Hall tach input (external interrupt).
};

// Order: FL, FR, RL, RR — pairs (svPwm, fr, en, bk, pg).
static const DriverPins kPins[NUM_WHEELS] = {
  {2, 22, 30, 34, 18},  // FL
  {3, 23, 31, 35, 19},  // FR
  {5, 24, 32, 36, 20},  // RL
  {6, 25, 33, 37, 21},  // RR
};

// ---- Geometry (meters): used in IK/FK. Half-length/width from wheel contact
// rectangle center to center along X / Y. kGeom couples yaw to diagonal wheels.
// Board frame: +X forward, +Y left, +Z up.
const float BOARD_HALF_LENGTH_M = 0.6096f;
const float BOARD_HALF_WIDTH_M = 0.3048f;
// Effective rolling radius for arc length = 2*pi*r per wheel revolution.
const float WHEEL_RADIUS_M = 0.0485f;

// Distance from center to wheel along diagonal used in omega terms (must match IK).
const float kGeom = BOARD_HALF_LENGTH_M + BOARD_HALF_WIDTH_M;

// ---- Motor / encoder --------------------------------------------------------
// PG edges per mechanical revolution: 3 * pole_pairs (common BLDC Hall pattern).
// Tune MOTOR_POLE_PAIRS and WHEEL_RADIUS_M against a measured roll distance.
const unsigned MOTOR_POLE_PAIRS = 7;

// ---- Motion & PID (tune on hardware) ---------------------------------------
const float INCH_TO_M = 0.0254f;  // Serial distance_in -> meters for goal.
const float MAX_VX_MPS = 1.2f;    // Body +X velocity clamp (m/s).
const float MAX_VY_MPS = 1.2f;    // Body +Y velocity clamp (m/s).
// PID outputs a scalar speed; this caps |vCmd| along (ux,uy) (m/s).
const float MAX_TRANSLATION_SPEED_MPS =
    (MAX_VX_MPS < MAX_VY_MPS) ? MAX_VX_MPS : MAX_VY_MPS;

const float DIST_TOL_M = 0.012f;  // |position error| below this counts as "at goal".
const float VEL_EPS_MPS = 0.04f;  // |vCmd| below this counts as "essentially stopped".
const uint8_t DONE_HOLD_CYCLES = 5;  // Consecutive OK ticks before declaring done.
const unsigned long MOVE_TIMEOUT_MS = 60000UL;  // Abort move if exceeded (ms).

// PID: one axis (meters error -> m/s). D term uses err derivative vs time.
const float PID_KP = 2.0f;
const float PID_KI = 0.15f;
const float PID_KD = 0.05f;
const float PID_I_CLAMP = 0.6f;  // Anti-windup: limit |integral contribution|.

// Wheel command magnitude 0..255 after IK scaling; floor for motion when nonzero.
const int MAX_WHEEL_CMD = 255;
const int MIN_EFFECTIVE_CMD = 25;

// How often runControlTick() runs while MOVING (also nominal PID dt fallback).
const unsigned long CONTROL_PERIOD_MS = 20;
#if DEBUG_TELEMETRY
const unsigned long TELEMETRY_PERIOD_MS = 200;
#endif

unsigned long lastOutTelemMillis = 0;

// HIGH here means "enable pin driven high to run" — flip if your driver is inverted.
const bool EN_ACTIVE_LEVEL = HIGH;

// If true, zero speed and stopAllMotors() ground BK (brake on).
const bool BRAKE_WHEN_STOPPED = true;

// Microseconds: after dropping EN, wait before toggling F/R (driver datasheet).
const uint16_t FR_EN_OFF_DELAY_US = 5000;

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

enum MoveState : uint8_t { STATE_IDLE = 0, STATE_MOVING = 1 };

// PG pulse counts (ISR increments; volatile for main/ISR visibility).
volatile unsigned long pgPulseTotal[NUM_WHEELS] = {0, 0, 0, 0};
// Last total seen at end of previous control tick (for delta pulses this tick).
unsigned long pgSnapPrev[NUM_WHEELS] = {0, 0, 0, 0};

// Sum of signed dsWheel[i] over the current move (meters); printed as inches at DONE.
float wheelTravelM[NUM_WHEELS] = {0.0f, 0.0f, 0.0f, 0.0f};

// Estimated pose in mission frame (origin and heading at move start). Units: m, rad.
float missionX_m = 0.0f;
float missionY_m = 0.0f;
float missionTheta_rad = 0.0f;

// Commanded straight-line distance along (ux,uy) from start (meters).
float distanceGoal_m = 0.0f;
// Unit direction in body XY at command time; projection (x*ux + y*uy) = scalar progress.
float ux = 1.0f;
float uy = 0.0f;

// PID accumulators and timing (micros for derivative dt).
float pidIntegral = 0.0f;
float pidLastErr = 0.0f;
unsigned long pidLastUs = 0;

MoveState moveState = STATE_IDLE;
unsigned long lastControlMillis = 0;  // Last runControlTick wall time.
unsigned long moveStartMillis = 0;    // For MOVE_TIMEOUT_MS.
uint8_t doneStableCycles = 0;         // Count of consecutive "at goal" ticks.

#if DEBUG_TELEMETRY
unsigned long lastTelemetryMillis = 0;
#endif

#define CMD_BUF_SIZE 96
char cmdBuf[CMD_BUF_SIZE];  // Incoming serial line buffer (no NUL until line end).
uint8_t cmdLen = 0;

// Tracks last F/R applied per wheel so we only sequence EN off when dir flips.
static bool lastDirForward[NUM_WHEELS] = {true, true, true, true};

// Last SV duty and EN state applied (for Serial OUT / STATUS); DIR uses lastDirForward.
uint8_t lastSvDuty[NUM_WHEELS] = {0, 0, 0, 0};
uint8_t lastEnOn[NUM_WHEELS] = {0, 0, 0, 0};

// Drive line high: used for "forward" / "brake released" on driver inputs that need it.
static inline void driverInputRelease(uint8_t pin) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, HIGH);
}

// Ground line: "reverse" / "brake on" per BLD-515C-style wiring notes in header.
static inline void driverInputGround(uint8_t pin) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
}

// Float / int saturate to [lo, hi] (PID clamps, PWM limits).
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

// External interrupt ISRs: count RISING edges on each PG line (keep minimal work here).
void isrPgFL() { pgPulseTotal[FL]++; }
void isrPgFR() { pgPulseTotal[FR]++; }
void isrPgRL() { pgPulseTotal[RL]++; }
void isrPgRR() { pgPulseTotal[RR]++; }

void updateStatusOutputs() {
  digitalWrite(STATUS_PIN_ARMED, (moveState == STATE_MOVING) ? HIGH : LOW);
  bool anyDrive = false;
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    if (lastSvDuty[i] > 0) {
      anyDrive = true;
      break;
    }
  }
  digitalWrite(STATUS_PIN_DRIVE, anyDrive ? HIGH : LOW);
}

void printWheelOutputLine() {
  Serial.print(F("OUT "));
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    if (i) Serial.print(' ');
    Serial.print(lastSvDuty[i]);
  }
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    Serial.print(' ');
    Serial.print(lastDirForward[i] ? 1 : 0);
  }
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    Serial.print(' ');
    Serial.print(lastEnOn[i]);
  }
  Serial.println();
}

void printStatusLine() {
  bool anyDrive = false;
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    if (lastSvDuty[i] > 0) {
      anyDrive = true;
      break;
    }
  }
  Serial.print(F("STATUS "));
  Serial.print((moveState == STATE_MOVING) ? 1 : 0);
  Serial.print(' ');
  Serial.print(anyDrive ? 1 : 0);
  Serial.print(' ');
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    if (i) Serial.print(' ');
    Serial.print(lastSvDuty[i]);
  }
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    Serial.print(' ');
    Serial.print(lastDirForward[i] ? 1 : 0);
  }
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    Serial.print(' ');
    Serial.print(lastEnOn[i]);
  }
  Serial.println();
}

// Safe stop: disable drivers, PWM zero, optional brake.
void stopAllMotors() {
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    digitalWrite(kPins[i].en, !EN_ACTIVE_LEVEL);
    analogWrite(kPins[i].svPwm, 0);
    lastSvDuty[i] = 0;
    lastEnOn[i] = 0;
    if (BRAKE_WHEN_STOPPED) {
      driverInputGround(kPins[i].bkPin);
    } else {
      driverInputRelease(kPins[i].bkPin);
    }
  }
  updateStatusOutputs();
}

// Read actual F/R pin state so odometry sign matches commanded wheel direction.
static inline bool wheelCommandedForward(uint8_t index) {
  return digitalRead(kPins[index].frPin) == HIGH;
}

// Circumference / pulses_per_rev -> meters advanced per PG pulse (sign from F/R).
static inline float metersPerPulse() {
  const float pulsesPerRev = 3.0f * (float)MOTOR_POLE_PAIRS;
  return (2.0f * (float)M_PI * WHEEL_RADIUS_M) / pulsesPerRev;
}

// signedCmd: sign = direction (F/R), magnitude = PWM. Handles EN/F/R sequencing.
void applyWheelCommand(uint8_t index, int signedCmd) {
  if (index >= NUM_WHEELS) return;

  int cmd = clampi(signedCmd, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);
  bool dirForward = (cmd >= 0);
  int magnitude = abs(cmd);

  // Avoid commanding a stall-speed PWM when nonzero command requested.
  if (magnitude > 0 && magnitude < MIN_EFFECTIVE_CMD) {
    magnitude = MIN_EFFECTIVE_CMD;
  }

  // Direction change: drop EN, wait, switch F/R, then EN can come back below.
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

  // Moving: release brake; stopped: optional brake on.
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

  lastSvDuty[index] = (uint8_t)magnitude;
  lastEnOn[index] = (magnitude > 0) ? 1 : 0;
}

// Body twist (vx, vy, omega) -> each wheel rad/s for this omni layout, then PWM.
// If any wheel exceeds MAX_WHEEL_CMD worth of demand, scale all down proportionally.
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

// Forward kinematics: small wheel arc lengths dsWheel[] (m) -> body motion, then
// rotate into mission frame by current missionTheta_rad (accumulated heading).
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

// At move start: latch PG totals so first tick deltas are from command time only.
void snapshotPgBaseline() {
  noInterrupts();
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    pgSnapPrev[i] = pgPulseTotal[i];
  }
  interrupts();
}

// One control iteration: odometry update, PID, IK, completion/timeout.
// Returns false if move exceeded MOVE_TIMEOUT_MS (caller stops and goes IDLE).
bool runControlTick() {
  const unsigned long now = millis();
  if (now - moveStartMillis > MOVE_TIMEOUT_MS) {
    return false;
  }

  float dsWheel[NUM_WHEELS];
  const float mpp = metersPerPulse();

  // Snapshot PG totals atomically so ISR cannot update mid-copy.
  noInterrupts();
  unsigned long totals[NUM_WHEELS];
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    totals[i] = pgPulseTotal[i];
  }
  interrupts();

  // Pulse delta since last tick -> signed wheel arc length (m) using F/R for sign.
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    unsigned long delta = totals[i] - pgSnapPrev[i];
    pgSnapPrev[i] = totals[i];
    const float sign = wheelCommandedForward(i) ? 1.0f : -1.0f;
    dsWheel[i] = sign * mpp * (float)delta;
    wheelTravelM[i] += dsWheel[i];  // Cumulative per wheel for ODO_IN at DONE.
  }

  integrateOdometry(dsWheel);

  // Scalar progress along commanded line vs start (mission frame, meters).
  const float distTraveled = missionX_m * ux + missionY_m * uy;
  const float err = distanceGoal_m - distTraveled;

  // PID derivative time step (micros); first tick assumes CONTROL_PERIOD_MS.
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

  // Translation only (omega=0): PID drives speed along (ux, uy) in body frame.
  const float vx = vCmd * ux;
  const float vy = vCmd * uy;
  applyInverseKinematics(vx, vy, 0.0f);

  // Require both small position error and small command to avoid stopping early.
  if (fabsf(err) < DIST_TOL_M && fabsf(vCmd) < VEL_EPS_MPS) {
    if (doneStableCycles < 255) doneStableCycles++;
  } else {
    doneStableCycles = 0;
  }

  // Stable at goal: stop, report odometry in inches, acknowledge.
  if (doneStableCycles >= DONE_HOLD_CYCLES) {
    stopAllMotors();
    moveState = STATE_IDLE;
    pidIntegral = 0.0f;
    pidLastErr = 0.0f;
    pidLastUs = 0;
    const float totalIn =
        (missionX_m * ux + missionY_m * uy) / INCH_TO_M;
    Serial.print(F("ODO_IN FL "));
    Serial.print(wheelTravelM[FL] / INCH_TO_M, 4);
    Serial.print(F(" FR "));
    Serial.print(wheelTravelM[FR] / INCH_TO_M, 4);
    Serial.print(F(" RL "));
    Serial.print(wheelTravelM[RL] / INCH_TO_M, 4);
    Serial.print(F(" RR "));
    Serial.print(wheelTravelM[RR] / INCH_TO_M, 4);
    Serial.print(F(" TOTAL "));
    Serial.println(totalIn, 4);
    Serial.println(F("ACK DONE"));
  }

  return true;
}

// Parse "float,float" from a single line (distance inches, angle radians).
// Uses strtod (AVR sscanf %f is unreliable).
static bool parseCommaPair(const char *line, float *outD, float *outA) {
  const char *p = line;
  char *end1 = NULL;
  double d = strtod(p, &end1);
  if (end1 == p) return false;
  p = end1;
  if (*p != ',') return false;
  p++;
  char *end2 = NULL;
  double a = strtod(p, &end2);
  if (end2 == p) return false;
  *outD = (float)d;
  *outA = (float)a;
  return isfinite(*outD) && isfinite(*outA);
}

// Called when a full line arrived in IDLE: validate, init mission, start MOVING.
void tryProcessIdleLine() {
  if (cmdLen == 0) return;
  cmdBuf[cmdLen] = '\0';

  if (strcmp(cmdBuf, "PING") == 0) {
    Serial.println(F("PONG"));
    cmdLen = 0;
    return;
  }
  if (strcmp(cmdBuf, "HELP") == 0) {
    Serial.println(F("distance_in,angle_rad | PING | STATUS | STOP"));
    cmdLen = 0;
    return;
  }
  if (strcmp(cmdBuf, "STATUS") == 0) {
    printStatusLine();
    cmdLen = 0;
    return;
  }
  if (strcmp(cmdBuf, "STOP") == 0) {
    Serial.println(F("ACK STOP"));
    cmdLen = 0;
    return;
  }

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

  // Mission pose origin at this command; heading starts 0 (accumulates from FK).
  missionX_m = missionY_m = 0.0f;
  missionTheta_rad = 0.0f;

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    wheelTravelM[i] = 0.0f;
  }

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
  updateStatusOutputs();
  cmdLen = 0;
}

void setup() {
  Serial.begin(115200);

  // Configure outputs and PG inputs; start disabled with brake policy.
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

  // RISING: one count per PG pulse edge (match driver PG output behavior).
  attachInterrupt(digitalPinToInterrupt(kPins[FL].pg), isrPgFL, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[FR].pg), isrPgFR, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[RL].pg), isrPgRL, RISING);
  attachInterrupt(digitalPinToInterrupt(kPins[RR].pg), isrPgRR, RISING);

  pinMode(STATUS_PIN_ARMED, OUTPUT);
  pinMode(STATUS_PIN_DRIVE, OUTPUT);
  digitalWrite(STATUS_PIN_ARMED, LOW);
  digitalWrite(STATUS_PIN_DRIVE, LOW);

  stopAllMotors();

  Serial.println(F("Omni BLDC: distance_in,angle_rad | PING | STATUS | STOP"));
  Serial.println(F("Telemetry: PG + OUT every 2s; STATUS on demand"));
}

void loop() {
  // Drain UART: in MOVING, only STOP is honored; other lines discarded.
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (moveState == STATE_MOVING) {
      if (c == '\n' || c == '\r') {
        if (cmdLen > 0) {
          cmdBuf[cmdLen] = '\0';
          if (strcmp(cmdBuf, "STOP") == 0) {
            stopAllMotors();
            moveState = STATE_IDLE;
            pidIntegral = 0.0f;
            pidLastErr = 0.0f;
            pidLastUs = 0;
            Serial.println(F("ACK STOP"));
          }
          cmdLen = 0;
        }
      } else if (cmdLen < CMD_BUF_SIZE - 1) {
        cmdBuf[cmdLen++] = c;
      } else {
        cmdLen = 0;
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

  // Fixed-rate control while MOVING; timeout returns false -> ERR TIMEOUT.
  if (moveState == STATE_MOVING && (now - lastControlMillis >= CONTROL_PERIOD_MS)) {
    lastControlMillis = now;
    if (!runControlTick()) {
      stopAllMotors();
      moveState = STATE_IDLE;
      pidIntegral = 0.0f;
      pidLastErr = 0.0f;
      pidLastUs = 0;
      Serial.println(F("ERR TIMEOUT"));
      updateStatusOutputs();
    }
  }

#if DEBUG_TELEMETRY
  // When IDLE, print raw cumulative PG counts (not reset per move).
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

  if (now - lastOutTelemMillis >= OUT_TELEMETRY_PERIOD_MS) {
    lastOutTelemMillis = now;
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
    printWheelOutputLine();
  }

  updateStatusOutputs();
}
