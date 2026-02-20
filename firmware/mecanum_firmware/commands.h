#ifndef COMMANDS_H
#define COMMANDS_H

// ---------------------------------------------------------------------------
// Robot physical constants (for future velocity-based control)
// ---------------------------------------------------------------------------
#define WHEEL_DIAMETER_MM  97.0
#define WHEEL_RADIUS_MM    (WHEEL_DIAMETER_MM / 2.0)
#define WHEEL_RADIUS_M     (WHEEL_RADIUS_MM / 1000.0)

#define WHEELBASE_MM       133.35   // front-to-rear axle center-to-center
#define TRACK_WIDTH_MM     304.8    // left-to-right wheel center-to-center

#define LX_MM              (WHEELBASE_MM / 2.0)   // 66.675
#define LY_MM              (TRACK_WIDTH_MM / 2.0)  // 152.4
#define K_MM               (LX_MM + LY_MM)         // 219.075
#define K_M                (K_MM / 1000.0)          // 0.219075

// ---------------------------------------------------------------------------
// Pulse defaults
// ---------------------------------------------------------------------------
#define DEFAULT_PULSE_PWM       150   // 0-255
#define DEFAULT_PULSE_DURATION  300   // milliseconds

// ---------------------------------------------------------------------------
// Movement directions
// ---------------------------------------------------------------------------
enum Direction {
  DIR_FORWARD = 0,
  DIR_BACKWARD,
  DIR_STRAFE_LEFT,
  DIR_STRAFE_RIGHT,
  DIR_DIAG_FL,
  DIR_DIAG_FR,
  DIR_DIAG_BL,
  DIR_DIAG_BR,
  DIR_ROTATE_CW,
  DIR_ROTATE_CCW,
  DIR_COUNT
};

// Kinematic sign table: {FL, FR, RL, RR}
// +1 = forward, -1 = reverse, 0 = stopped
// Derived from inverted-V (/ \) roller pattern:
//   FL = Vx - Vy - K*omega
//   FR = Vx + Vy + K*omega
//   RL = Vx + Vy - K*omega
//   RR = Vx - Vy + K*omega
static const int8_t KINEMATIC_TABLE[DIR_COUNT][4] = {
  //              FL  FR  RL  RR
  /* FORWARD  */ { 1,  1,  1,  1},
  /* BACKWARD */ {-1, -1, -1, -1},
  /* STR LEFT */ {-1,  1,  1, -1},
  /* STR RIGHT*/ { 1, -1, -1,  1},
  /* DIAG FL  */ { 0,  1,  1,  0},
  /* DIAG FR  */ { 1,  0,  0,  1},
  /* DIAG BL  */ {-1,  0,  0, -1},
  /* DIAG BR  */ { 0, -1, -1,  0},
  /* ROT CW   */ { 1, -1,  1, -1},
  /* ROT CCW  */ {-1,  1, -1,  1},
};

#endif
