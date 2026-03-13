// Arduino code to read two comma-separated float values from Serial,
// parse them, store in variables, and print them back.
int PWM_pin = 2;
void setup() {
  Serial.begin(115200); // Start serial communication at 9600 baud
  pinMode(PWM_pin, OUTPUT);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for some boards)
  }
  Serial.println("Ready. Please enter two numbers separated by a comma (e.g., 12.34,56.78):");
}

void loop() {
  // Check if data is available
  if (Serial.available() > 0) {
    // Read the incoming string until newline character
    String inputString = Serial.readStringUntil('\n');
    inputString.trim(); // Remove any leading/trailing whitespace

    // Find the position of the comma
    int commaIndex = inputString.indexOf(',');
    
    if (commaIndex == -1) {
      Serial.println("Error: No comma found. Please use format: number1,number2");
      return;
    }

    // Extract substrings for the two numbers
    String firstPart = inputString.substring(0, commaIndex);
    String secondPart = inputString.substring(commaIndex + 1);

    // Trim whitespace from each part (optional but recommended)
    firstPart.trim();
    secondPart.trim();

    // Convert to float
    float num1 = firstPart.toFloat();
    float num2 = secondPart.toFloat();

    // Check if conversion was valid (toFloat returns 0.0 if conversion fails)
    // A more robust check could involve checking if the string actually contained numbers.
    if (num1 == 0.0 && firstPart != "0.0" && firstPart != "0") {
      Serial.println("Error: First number is not a valid float.");
      return;
    }
    if (num2 == 0.0 && secondPart != "0.0" && secondPart != "0") {
      Serial.println("Error: Second number is not a valid float.");
      return;
    }

    // Display the extracted numbers
    Serial.print("First number: ");
    Serial.println(num1, 4); // Print with 4 decimal places
    Serial.print("Second number: ");
    Serial.println(num2, 4);
    Serial.println("Enter another pair:");

    analogWrite(PWM_pin, num1);
  }
}