#include <SPI.h>
#include <MFRC522.h>

// Hardware Pin Configurations for ESP32 DevKit V1 and RC522 Reader
#define SS_PIN    5   // VSPI Chip Select (CS / SDA)
#define RST_PIN   22  // Reset Pin
#define BUZZER_PIN 12 // Piezo Buzzer Digital Output Pin
#define GREEN_LED  14 // Pass Indicator LED Pin
#define RED_LED    27 // Fail Indicator LED Pin

MFRC522 rfrc522(SS_PIN, RST_PIN); // Instantiate MFRC522 Reader Object

void setup() {
  // Initialize Serial Monitor interface pipeline targeting our Python Terminal Core
  Serial.begin(115200);
  while (!Serial); // Wait for connection stream to establish
  
  // Set peripheral hardware pins to output mode
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  // Default turn off all output peripherals
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  // Initialize SPI bus communication protocol channels
  SPI.begin();
  // Initialize MFRC522 hardware unit
  rfrc522.PCD_Init();

  // Chirp buzzer once to signal successful boot up initialization
  digitalWrite(BUZZER_PIN, HIGH);
  delay(150);
  digitalWrite(BUZZER_PIN, LOW);
}

void loop() {
  // Check down-stream Serial pipeline for incoming acknowledgment execution feedbacks from python script
  if (Serial.available() > 0) {
    char access_signal = Serial.read();
    if (access_signal == '1') {
      // Access Granted: Flash green light and beep twice
      digitalWrite(GREEN_LED, HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
      delay(100);
      digitalWrite(BUZZER_PIN, LOW);
      delay(100);
      digitalWrite(BUZZER_PIN, HIGH);
      delay(100);
      digitalWrite(BUZZER_PIN, LOW);
      delay(400);
      digitalWrite(GREEN_LED, LOW);
    } 
    else if (access_signal == '0') {
      // Access Denied: Flash red light and hold a solid error tone
      digitalWrite(RED_LED, HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
      delay(600);
      digitalWrite(BUZZER_PIN, LOW);
      digitalWrite(RED_LED, LOW);
    }
  }

  // Look for new physical transponder cards present on the receiver sensor antenna coil
  if (!rfrc522.PICC_IsNewCardPresent()) {
    return;
  }

  // Read the unique identifier sector block values of the target card
  if (!rfrc522.PICC_ReadCardSerial()) {
    return;
  }

  // Parse the byte elements of the card array into an organized uppercase string format
  String scanned_uid = "";
  for (byte i = 0; i < rfrc522.uid.size; i++) {
    scanned_uid += String(rfrc522.uid.uidByte[i] < 0x10 ? "0" : "");
    scanned_uid += String(rfrc522.uid.uidByte[i], HEX);
  }
  scanned_uid.toUpperCase();

  // Print raw structured text token downstream to our Serial connection orchestrator parsing pipeline
  Serial.print("UID:");
  Serial.println(scanned_uid);

  // Halt PICC transponder transmission to avoid duplicated continuous polling loops
  rfrc522.PICC_HaltA();
  // Stop encryption communication cycles on the master PCD unit module
  rfrc522.PCD_StopCrypto1();

  // Native hardware debounce protection block delay to prevent rapid duplicate scan streams
  delay(1500);
}