// Mega 2560: four motor channels (FR, FL, RR, RL style but generic).
// Serial format: pwm0,pwm1,pwm2,pwm3   (signed -255..255, comma separated)
// Line "p" prints PG pulse totals for all motors.
// Each motor uses the same sequencing as the original single‑motor code.

#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ------------------------- Pin Definitions -------------------------
const int NUM_MOTORS = 4;

// SV pins (PWM)
const uint8_t svPins[NUM_MOTORS] = {2, 3, 5, 6};
// FR pins (direction)
const uint8_t frPins[NUM_MOTORS] = {22, 23, 24, 25};
// EN pins (enable)
const uint8_t enPins[NUM_MOTORS] = {30, 31, 32, 33};
// BK pins (brake)
const uint8_t bkPins[NUM_MOTORS] = {34, 35, 36, 37};
// PG pins (quadrature / pulse input)
const uint8_t pgPins[NUM_MOTORS] = {18, 19, 20, 21};

// ------------------------- Motor Constants -------------------------
const int MAX_WHEEL_CMD = 255;
const int MIN_EFFECTIVE_CMD = 25;
const uint16_t EN_OFF_DELAY_US = 5000;      // delay when changing direction
const bool EN_ACTIVE_LEVEL = LOW;           // LOW = run (BLD‑515C: EN grounded = run)
const bool BRAKE_WHEN_STOPPED = true;

// ------------------------- Per‑motor State -------------------------
volatile unsigned long pgCount[NUM_MOTORS] = {0};
static bool lastDirForward[NUM_MOTORS] = {true, true, true, true};

// ------------------------- Helper Functions -------------------------
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

// ------------------------- Individual ISRs for PG pins -------------------------
void isrPg0() { pgCount[0]++; }
void isrPg1() { pgCount[1]++; }
void isrPg2() { pgCount[2]++; }
void isrPg3() { pgCount[3]++; }

// ------------------------- Motor Control Function -------------------------
// Controls a single motor according to the signed command (-255..255)
void motor(int idx, int16_t signedCmd) {
    int cmd = clampi((int)signedCmd, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);
    bool dirForward = (cmd >= 0);
    int magnitude = abs(cmd);

    // Apply minimum effective command if non‑zero but too small
    if (magnitude > 0 && magnitude < MIN_EFFECTIVE_CMD) {
        magnitude = MIN_EFFECTIVE_CMD;
    }

    // Direction change: disable EN, delay, set FR pin
    if (dirForward != lastDirForward[idx]) {
        digitalWrite(enPins[idx], !EN_ACTIVE_LEVEL);   // disable
        delayMicroseconds(EN_OFF_DELAY_US);
        if (dirForward) {
            driverInputRelease(frPins[idx]);           // forward = HIGH
        } else {
            driverInputGround(frPins[idx]);            // reverse = LOW
        }
        lastDirForward[idx] = dirForward;
    }

    // Brake pin: release (HIGH) when moving, else optional brake
    if (magnitude > 0) {
        driverInputRelease(bkPins[idx]);               // brake off
    } else {
        if (BRAKE_WHEN_STOPPED) {
            driverInputGround(bkPins[idx]);            // brake on
        } else {
            driverInputRelease(bkPins[idx]);
        }
    }

    // Enable pin and PWM output
    digitalWrite(enPins[idx], (magnitude > 0) ? EN_ACTIVE_LEVEL : !EN_ACTIVE_LEVEL);
    analogWrite(svPins[idx], magnitude);
}

// ------------------------- Serial Command Parsing -------------------------
#define LINE_BUF_SIZE 96
static char lineBuf[LINE_BUF_SIZE];
static uint8_t lineLen = 0;

// Process a complete line (terminated by CR/LF)
void processSerialLine() {
    if (lineLen == 0) return;
    lineBuf[lineLen] = '\0';
    lineLen = 0;

    // Trim leading spaces
    char *p = lineBuf;
    while (*p == ' ' || *p == '\t') p++;

    // Command 'p' : print PG counts
    if (*p == 'p' && (p[1] == '\0' || p[1] == ' ' || p[1] == '\t')) {
        noInterrupts();
        unsigned long snap0 = pgCount[0];
        unsigned long snap1 = pgCount[1];
        unsigned long snap2 = pgCount[2];
        unsigned long snap3 = pgCount[3];
        interrupts();
        Serial.print(F("PG0=")); Serial.print(snap0);
        Serial.print(F(" PG1=")); Serial.print(snap1);
        Serial.print(F(" PG2=")); Serial.print(snap2);
        Serial.print(F(" PG3=")); Serial.println(snap3);
        return;
    }

    // Parse four comma‑separated integers
    int values[NUM_MOTORS];
    int parsedCount = 0;
    char *token = strtok(p, ",");
    while (token != NULL && parsedCount < NUM_MOTORS) {
        char *end;
        long v = strtol(token, &end, 10);
        if (end == token) {
            Serial.println(F("ERR: invalid number"));
            return;
        }
        values[parsedCount] = (int)clampi((int)v, -MAX_WHEEL_CMD, MAX_WHEEL_CMD);
        parsedCount++;
        token = strtok(NULL, ",");
    }

    // Check for extra tokens or missing values
    if (parsedCount != NUM_MOTORS) {
        Serial.print(F("ERR: need exactly "));
        Serial.print(NUM_MOTORS);
        Serial.println(F(" comma‑separated values"));
        return;
    }

    // Apply commands to all motors
    for (int i = 0; i < NUM_MOTORS; i++) {
        motor(i, (int16_t)values[i]);
    }

    // Report new PG counts after command
    noInterrupts();
    unsigned long snap0 = pgCount[0];
    unsigned long snap1 = pgCount[1];
    unsigned long snap2 = pgCount[2];
    unsigned long snap3 = pgCount[3];
    interrupts();

    Serial.print(F("OK CMD="));
    Serial.print(values[0]); Serial.print(F(","));
    Serial.print(values[1]); Serial.print(F(","));
    Serial.print(values[2]); Serial.print(F(","));
    Serial.print(values[3]);
    Serial.print(F(" PG="));
    Serial.print(snap0); Serial.print(F(","));
    Serial.print(snap1); Serial.print(F(","));
    Serial.print(snap2); Serial.print(F(","));
    Serial.println(snap3);
}

// ------------------------- Setup -------------------------
void setup() {
    Serial.begin(115200);

    // Initialize pins for each motor
    for (int i = 0; i < NUM_MOTORS; i++) {
        pinMode(svPins[i], OUTPUT);
        analogWrite(svPins[i], 0);

        pinMode(enPins[i], OUTPUT);
        digitalWrite(enPins[i], !EN_ACTIVE_LEVEL);   // initially disabled

        driverInputRelease(frPins[i]);               // default direction forward
        lastDirForward[i] = true;

        if (BRAKE_WHEN_STOPPED) {
            driverInputGround(bkPins[i]);            // brake on when stopped
        } else {
            driverInputRelease(bkPins[i]);
        }

        pinMode(pgPins[i], INPUT_PULLUP);
    }

    // Attach interrupts for PG pins
    attachInterrupt(digitalPinToInterrupt(pgPins[0]), isrPg0, RISING);
    attachInterrupt(digitalPinToInterrupt(pgPins[1]), isrPg1, RISING);
    attachInterrupt(digitalPinToInterrupt(pgPins[2]), isrPg2, RISING);
    attachInterrupt(digitalPinToInterrupt(pgPins[3]), isrPg3, RISING);

    // Wait for Serial Monitor (optional)
    const unsigned long t0 = millis();
    while (!Serial && (millis() - t0 < 3000UL)) { }

    // Print startup info
    Serial.println(F("4‑Motor Controller ready"));
    Serial.print(F("SV pins: "));
    for (int i = 0; i < NUM_MOTORS; i++) Serial.print(svPins[i]), Serial.print(i < 3 ? "," : "");
    Serial.print(F("  FR pins: "));
    for (int i = 0; i < NUM_MOTORS; i++) Serial.print(frPins[i]), Serial.print(i < 3 ? "," : "");
    Serial.print(F("  EN pins: "));
    for (int i = 0; i < NUM_MOTORS; i++) Serial.print(enPins[i]), Serial.print(i < 3 ? "," : "");
    Serial.print(F("  BK pins: "));
    for (int i = 0; i < NUM_MOTORS; i++) Serial.print(bkPins[i]), Serial.print(i < 3 ? "," : "");
    Serial.print(F("  PG pins: "));
    for (int i = 0; i < NUM_MOTORS; i++) Serial.print(pgPins[i]), Serial.print(i < 3 ? "," : "");
    Serial.println();

    Serial.println(F("Send four comma‑separated PWM values, e.g.  -80,120,0,200"));
    Serial.println(F("Send p  to print PG counts"));
}

// ------------------------- Main Loop -------------------------
void loop() {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r' || c == '\n') {
            processSerialLine();
            continue;
        }
        if (lineLen < LINE_BUF_SIZE - 1) {
            lineBuf[lineLen++] = c;
        } else {
            lineLen = 0;
            Serial.println(F("ERR: line too long"));
        }
    }
}