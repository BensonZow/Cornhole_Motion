// Mega 2560: all four motor channels — pins match omni_bldc_firmware kPins (FL, FR, RL, RR).
// Serial: signed_pwm,ignored — sign = direction, magnitude 0..255 (e.g. 80,0 or -80,0).
// Line "p" prints PG pulse totals for all wheels.

#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NUM_WHEELS 4
#define FL 0
#define FR 1
#define RL 2
#define RR 3

// Pin bundle per wheel: must match omni_bldc_firmware.ino wiring.
struct DriverPins {
  uint8_t svPwm;
  uint8_t frPin;
  uint8_t en;
  uint8_t bkPin;
  uint8_t pg;
};

// Order: FL, FR, RL, RR — (svPwm, fr, en, bk, pg).
static const DriverPins kPins[NUM_WHEELS] = {
    {2, 22, 30, 34, 18},   // FL
    {3, 23, 31, 35, 19},   // FR
    {5, 24, 32, 36, 20},   // RL
    {6, 25, 33, 37, 21},   // RR
};

const int MAX_WHEEL_CMD = 255;
const int MIN_EFFECTIVE_CMD = 25;
const uint16_t FR_EN_OFF_DELAY_US = 5000;
// Level on EN pin that means "driver run" (not "disabled"). Many BLD carriers invert Mega→driver;
// use serial E0 = run is LOW, E1 = run is HIGH to A/B test without reflash (default HIGH = omni_bldc_firmware).
static uint8_t enActiveLevel = HIGH;
const bool BRAKE_WHEN_STOPPED = true;

// Set 0 to silence debug lines (paste DBG12720 lines back for analysis).
#define DEBUG_SERIAL_INSTRUMENT 1

volatile unsigned long pgPulseTotal[NUM_WHEELS] = {0, 0, 0, 0};
static bool lastDirForward[NUM_WHEELS];

void isrPgFL() { pgPulseTotal[FL]++; }
void isrPgFR() { pgPulseTotal[FR]++; }
void isrPgRL() { pgPulseTotal[RL]++; }
void isrPgRR() { pgPulseTotal[RR]++; }

static inline void driverInputRelease(uint8_t pin) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, HIGH);
}

static inline void driverInputGround(uint8_t pin) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
}

static inline int clampi(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

#if DEBUG_SERIAL_INSTRUMENT
static uint16_t dbgSeq = 0;

// After analogWrite, digitalRead(svPwm) is not meaningful on timer pins; log EN/FR/BK only.
static void dbgPinSnapshot(const char *hypothesisId, const char *location, int magWritten) {
  Serial.print(F("DBG12720 {\"sessionId\":\"12720d\",\"seq\":"));
  Serial.print((unsigned int)++dbgSeq);
  Serial.print(F(",\"hypothesisId\":\""));
  Serial.print(hypothesisId);
  Serial.print(F("\",\"location\":\""));
  Serial.print(location);
  Serial.print(F("\",\"message\":\"pin_snapshot\",\"timestamp\":"));
  Serial.print(millis());
  Serial.print(F(",\"data\":{\"mag\":"));
  Serial.print(magWritten);
  Serial.print(F(",\"enRunIsHigh\":"));
  Serial.print(enActiveLevel == HIGH ? 1 : 0);
  Serial.print(F(",\"w\":["));
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    if (i) Serial.print(',');
    Serial.print(F("{\"i\":"));
    Serial.print(i);
    Serial.print(F(",\"en\":"));
    Serial.print(digitalRead(kPins[i].en));
    Serial.print(F(",\"fr\":"));
    Serial.print(digitalRead(kPins[i].frPin));
    Serial.print(F(",\"bk\":"));
    Serial.print(digitalRead(kPins[i].bkPin));
    Serial.print('}');
  }
  Serial.println(F("]}}"));
}
#endif

// Apply the same signed PWM command to every wheel (BLD-515C-style sequencing per channel).
void motorsAll(int16_t signedCmd) {
  int cmd = clampi((int)signedCmd, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);
  bool dirForward = (cmd >= 0);
  int magnitude = abs(cmd);

#if DEBUG_SERIAL_INSTRUMENT
  // #region agent log
  Serial.print(F("DBG12720 {\"sessionId\":\"12720d\",\"seq\":"));
  Serial.print((unsigned int)++dbgSeq);
  Serial.print(F(",\"hypothesisId\":\"H4\",\"location\":\"motorsAll:pre_min\",\"message\":\"cmd_shape\",\"timestamp\":"));
  Serial.print(millis());
  Serial.print(F(",\"data\":{\"signedIn\":"));
  Serial.print((int)signedCmd);
  Serial.print(F(",\"clamped\":"));
  Serial.print(cmd);
  Serial.print(F(",\"magBeforeMin\":"));
  Serial.print(magnitude);
  Serial.println(F("}}"));
  // #endregion
#endif

  if (magnitude > 0 && magnitude < MIN_EFFECTIVE_CMD) {
    magnitude = MIN_EFFECTIVE_CMD;
  }

  bool anyDirChange = false;
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    if (dirForward != lastDirForward[i]) {
      anyDirChange = true;
      break;
    }
  }
#if DEBUG_SERIAL_INSTRUMENT
  // #region agent log
  Serial.print(F("DBG12720 {\"sessionId\":\"12720d\",\"seq\":"));
  Serial.print((unsigned int)++dbgSeq);
  Serial.print(F(",\"hypothesisId\":\"H3\",\"location\":\"motorsAll:dir_gate\",\"message\":\"dir_sequence\",\"timestamp\":"));
  Serial.print(millis());
  Serial.print(F(",\"data\":{\"dirFwd\":"));
  Serial.print(dirForward ? 1 : 0);
  Serial.print(F(",\"anyDirChange\":"));
  Serial.print(anyDirChange ? 1 : 0);
  Serial.println(F("}}"));
  // #endregion
#endif
  if (anyDirChange) {
    for (uint8_t i = 0; i < NUM_WHEELS; i++) {
      digitalWrite(kPins[i].en, enActiveLevel == HIGH ? LOW : HIGH);
    }
    delayMicroseconds(FR_EN_OFF_DELAY_US);
    for (uint8_t i = 0; i < NUM_WHEELS; i++) {
      if (dirForward != lastDirForward[i]) {
        if (dirForward) {
          driverInputRelease(kPins[i].frPin);
        } else {
          driverInputGround(kPins[i].frPin);
        }
        lastDirForward[i] = dirForward;
      }
    }
  }

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    if (magnitude > 0) {
      driverInputRelease(kPins[i].bkPin);
    } else {
      if (BRAKE_WHEN_STOPPED) {
        driverInputGround(kPins[i].bkPin);
      } else {
        driverInputRelease(kPins[i].bkPin);
      }
    }

    digitalWrite(kPins[i].en,
                  (magnitude > 0) ? enActiveLevel : (uint8_t)(enActiveLevel == HIGH ? LOW : HIGH));
    analogWrite(kPins[i].svPwm, magnitude);
  }

#if DEBUG_SERIAL_INSTRUMENT
  // #region agent log
  dbgPinSnapshot("H1_H2_H3", "motorsAll:after_apply", magnitude);
  // H1: if mag>0, expect digitalRead(en)==enActiveLevel for each wheel
  // H2: if mag>0, expect bk HIGH (brake off) with our driverInputRelease = HIGH
  // H3: fr HIGH => forward per sketch convention
  // #endregion
#endif
}

#define LINE_BUF_SIZE 96
static char lineBuf[LINE_BUF_SIZE];
static uint8_t lineLen = 0;

// True when buffer is a complete two-field line "num,num" (no trailing junk). Lets Serial tools
// that send no LF still run motorsAll() after the last digit of the second field.
static bool commaLineComplete(uint8_t len) {
  if (len == 0 || len >= LINE_BUF_SIZE) {
    return false;
  }
  lineBuf[len] = '\0';
  if (strchr(lineBuf, ',') == NULL) {
    return false;
  }
  char *end1 = NULL;
  (void)strtod(lineBuf, &end1);
  if (end1 == lineBuf) {
    return false;
  }
  while (*end1 == ' ' || *end1 == '\t') {
    end1++;
  }
  if (*end1 != ',') {
    return false;
  }
  char *p2 = end1 + 1;
  while (*p2 == ' ' || *p2 == '\t') {
    p2++;
  }
  if (*p2 == '\0') {
    return false;
  }
  char *end2 = NULL;
  (void)strtod(p2, &end2);
  if (end2 == p2) {
    return false;
  }
  while (*end2 == ' ' || *end2 == '\t') {
    end2++;
  }
  return *end2 == '\0';
}

void processSerialLineBuffer() {
  if (lineLen == 0) {
    return;
  }
  lineBuf[lineLen] = '\0';
  lineLen = 0;

  char *p = lineBuf;
  while (*p == ' ' || *p == '\t') {
    p++;
  }

  if ((p[0] == 'e' || p[0] == 'E') && (p[1] == '0' || p[1] == '1')) {
    char *er = p + 2;
    while (*er == ' ' || *er == '\t') {
      er++;
    }
    if (*er == '\0') {
      enActiveLevel = (p[1] == '0') ? LOW : HIGH;
      motorsAll(0);
      Serial.print(F("OK enRunLevel="));
      Serial.println(enActiveLevel == HIGH ? F("HIGH") : F("LOW"));
#if DEBUG_SERIAL_INSTRUMENT
      // #region agent log
      dbgPinSnapshot("H1_toggle", "serial:Ecmd", 0);
      // #endregion
#endif
      return;
    }
  }

  if (*p == 'p' && (p[1] == '\0' || p[1] == ' ' || p[1] == '\t')) {
    noInterrupts();
    unsigned long pg0 = pgPulseTotal[FL];
    unsigned long pg1 = pgPulseTotal[FR];
    unsigned long pg2 = pgPulseTotal[RL];
    unsigned long pg3 = pgPulseTotal[RR];
    interrupts();
    Serial.print(F("PG FL="));
    Serial.print(pg0);
    Serial.print(F(" FR="));
    Serial.print(pg1);
    Serial.print(F(" RL="));
    Serial.print(pg2);
    Serial.print(F(" RR="));
    Serial.println(pg3);
    return;
  }

  char *comma = strchr(p, ',');
  if (comma != NULL) {
    *comma = '\0';
  }

  char *endPtr = NULL;
  double spd = strtod(p, &endPtr);
  if (endPtr == p || !isfinite(spd)) {
    Serial.println(F("ERR: bad number (use e.g. -80 or -80,0 then newline)"));
    return;
  }

  if (comma == NULL) {
    while (*endPtr == ' ' || *endPtr == '\t') {
      endPtr++;
    }
    if (*endPtr != '\0') {
      Serial.println(F("ERR: one number per line, or signed_pwm,ignored"));
      return;
    }
  }

  int rounded = (int)lround(spd);
  int16_t cmd = (int16_t)clampi(rounded, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);

  motorsAll(cmd);

  noInterrupts();
  unsigned long pg0 = pgPulseTotal[FL];
  unsigned long pg1 = pgPulseTotal[FR];
  unsigned long pg2 = pgPulseTotal[RL];
  unsigned long pg3 = pgPulseTotal[RR];
  interrupts();

  Serial.print(F("OK cmd="));
  Serial.print(cmd);
  Serial.print(F(" PG FL="));
  Serial.print(pg0);
  Serial.print(F(" FR="));
  Serial.print(pg1);
  Serial.print(F(" RL="));
  Serial.print(pg2);
  Serial.print(F(" RR="));
  Serial.println(pg3);
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    pinMode(kPins[i].svPwm, OUTPUT);
    analogWrite(kPins[i].svPwm, 0);
    pinMode(kPins[i].en, OUTPUT);
    digitalWrite(kPins[i].en, enActiveLevel == HIGH ? LOW : HIGH);
    driverInputRelease(kPins[i].frPin);
    lastDirForward[i] = true;
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

  const unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 3000UL)) {
    ;
  }

  Serial.println(F("All 4 wheels: pins = omni_bldc_firmware kPins (FL..RR)"));
  Serial.print(F("EN_run (default HIGH, omni match)="));
  Serial.print(enActiveLevel == HIGH ? F("HIGH") : F("LOW"));
  Serial.println(
      F(" | FL SV2 FR3 RL5 RR6 | F/R 22-25 EN 30-33 BK 34-37 PG 18-21"));
  Serial.println(
      F("Send: E0 = run level LOW on EN, E1 = run HIGH (A/B polarity); then send speed"));
  Serial.println(
      F("Send: -80 + Newline, OR -80,0 + Newline; two-field num,num also accepted without LF"));
  Serial.println(F("Send: p — PG counts (all wheels)"));
  Serial.println(F("Serial Monitor: 115200, line ending Newline or Both NL & CR (CR ends line)."));
#if DEBUG_SERIAL_INSTRUMENT
  // #region agent log
  Serial.print(F("DBG12720 {\"sessionId\":\"12720d\",\"seq\":"));
  Serial.print((unsigned int)++dbgSeq);
  Serial.print(F(",\"hypothesisId\":\"H1\",\"location\":\"setup:end\",\"message\":\"constants\",\"timestamp\":"));
  Serial.print(millis());
  Serial.print(F(",\"data\":{\"enRunIsHigh\":"));
  Serial.print(enActiveLevel == HIGH ? 1 : 0);
  Serial.print(F(",\"BRAKE_WHEN_STOPPED\":"));
  Serial.print(BRAKE_WHEN_STOPPED ? 1 : 0);
  Serial.println(F("}}"));
  dbgPinSnapshot("H1_H2", "setup:idle_pins", 0);
  // #endregion
#endif
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r' || c == '\n') {
      processSerialLineBuffer();
      continue;
    }
    if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuf[lineLen++] = c;
      if (strchr(lineBuf, ',') != NULL && commaLineComplete(lineLen)) {
        processSerialLineBuffer();
      }
    } else {
      lineLen = 0;
      Serial.println(F("ERR: line too long"));
    }
  }
}
