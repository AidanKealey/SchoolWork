const int numPins = 23; // Number of output pins
// int outputPins[numPins] = {26, 22, 24, 7, 6, 25, 23, 5, 4, 52, 53, 3, 33, 13, 49, 51, 12, 11, 40, 38, 10, 9, 8};
            // bitstring  {00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

// NEW MAPPINGS HERE
int outputPins[numPins] = {44, 42, 40, 2, 3, 38, 36, 4, 5, 34, 32, 6, 7, 8, 30, 28, 9, 10, 26, 24, 11, 12, 13};
            // bitstring  {00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


void setup() {
  Serial.begin(9600); // Initialize serial communication
  for (int i = 0; i < numPins; i++) {
    pinMode(outputPins[i], OUTPUT); // Set pins as outputs
  }
}

void loop() {
  if (Serial.available() >= 23) { // Wait until all 23 characters are available
    String inputString = Serial.readString(); // Read the string from serial port
    Serial.print(inputString);
    for (int i = 0; i < numPins; i++) {
      if (inputString.charAt(i) == '1') {
        digitalWrite(outputPins[i], HIGH); // Set pin to HIGH if corresponding character is '1'
      } else {
        digitalWrite(outputPins[i], LOW); // Set pin to LOW if corresponding character is '0'
      }
    }
  }
}
