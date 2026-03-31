// Mega 2560: single FR motor channel — pins match omni_bldc_firmware kPins[FR].
// Serial: signed_pwm,ignored — sign = direction, magnitude 0..255 (e.g. 80,0 or -80,0).
// Line "p" prints PG pulse total.

#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// SV pin must match your wiring (omni FR row uses D3 per README).
const uint8_t PIN_SV = 3;
const uint8_t PIN_FR = 23;
const uint8_t PIN_EN = 31;
const uint8_t PIN_BK = 35;
const uint8_t PIN_PG = 19;

const int MAX_WHEEL_CMD = 255;
const int MIN_EFFECTIVE_CMD = 25;
const uint16_t FR_EN_OFF_DELAY_US = 5000;
// BLD-515C manual: EN grounded = run. Use HIGH only if your board inverts EN between Mega and driver.
const bool EN_ACTIVE_LEVEL = LOW;
const bool BRAKE_WHEN_STOPPED = true;

volatile unsigned long pgCount = 0;
static bool lastDirForward = true;

void isrPgFr() { pgCount++; }

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

// signedCmd: sign = direction (F/R), magnitude = SV PWM 0..255 (BLD-515C-style sequencing).
void motor(int16_t signedCmd) {
  int cmd = clampi((int)signedCmd, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);
  bool dirForward = (cmd >= 0);
  int magnitude = abs(cmd);

  if (magnitude > 0 && magnitude < MIN_EFFECTIVE_CMD) {
    magnitude = MIN_EFFECTIVE_CMD;
  }

  if (dirForward != lastDirForward) {
    digitalWrite(PIN_EN, !EN_ACTIVE_LEVEL);
    delayMicroseconds(FR_EN_OFF_DELAY_US);
    if (dirForward) {
      driverInputRelease(PIN_FR);
    } else {
      driverInputGround(PIN_FR);
    }
    lastDirForward = dirForward;
  }

  if (magnitude > 0) {
    driverInputRelease(PIN_BK);
  } else {
    if (BRAKE_WHEN_STOPPED) {
      driverInputGround(PIN_BK);
    } else {
      driverInputRelease(PIN_BK);
    }
  }

  digitalWrite(PIN_EN, (magnitude > 0) ? EN_ACTIVE_LEVEL : !EN_ACTIVE_LEVEL);
  analogWrite(PIN_SV, magnitude);
}

#define LINE_BUF_SIZE 96
static char lineBuf[LINE_BUF_SIZE];
static uint8_t lineLen = 0;

// True when buffer is a complete two-field line "num,num" (no trailing junk). Lets Serial tools
// that send no LF still run motor() after the last digit of the second field.
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

  if (*p == 'p' && (p[1] == '\0' || p[1] == ' ' || p[1] == '\t')) {
    noInterrupts();
    unsigned long snap = pgCount;
    interrupts();
    Serial.print(F("PG "));
    Serial.println(snap);
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

  motor(cmd);

  noInterrupts();
  unsigned long pgSnap = pgCount;
  interrupts();

  Serial.print(F("OK cmd="));
  Serial.print(cmd);
  Serial.print(F(" PG="));
  Serial.println(pgSnap);
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_SV, OUTPUT);
  analogWrite(PIN_SV, 0);
  pinMode(PIN_EN, OUTPUT);
  digitalWrite(PIN_EN, !EN_ACTIVE_LEVEL);
  driverInputRelease(PIN_FR);
  lastDirForward = true;
  if (BRAKE_WHEN_STOPPED) {
    driverInputGround(PIN_BK);
  } else {
    driverInputRelease(PIN_BK);
  }

  pinMode(PIN_PG, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_PG), isrPgFr, RISING);

  const unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 3000UL)) {
    ;
  }

  Serial.print(F("FR motor SV=D"));
  Serial.print(PIN_SV);
  Serial.print(F(" EN_run="));
  Serial.print(EN_ACTIVE_LEVEL ? F("HIGH") : F("LOW"));
  Serial.println(F(" F/R=D23 EN=D31 BK=D35 PG=D19"));
  Serial.println(
      F("Send: -80 + Newline, OR -80,0 + Newline; two-field num,num also accepted without LF"));
  Serial.println(F("Send: p — PG count"));
  Serial.println(F("Serial Monitor: 115200, line ending Newline or Both NL & CR (CR ends line)."));
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
