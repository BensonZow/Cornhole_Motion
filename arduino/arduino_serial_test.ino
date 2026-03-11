void setup() {
  Serial.begin(9600); // Must match Python node baud rate
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read(); // Read one character
    
    if (data == '1') {
      digitalWrite(LED_BUILTIN, HIGH); // Turn LED ON
    } else if (data == '0') {
      digitalWrite(LED_BUILTIN, LOW);  // Turn LED OFF
    }
  }
}
