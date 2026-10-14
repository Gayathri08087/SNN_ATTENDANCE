#include "esp_camera.h"
#include <WiFi.h>
#include "secrets.h"

// Select the specific ESP32 Camera Hardware module configuration variant
#define CAMERA_MODEL_AI_THINKER

// AI-Thinker Pin Mappings Configuration
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Network Settings configuration parameters
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// Local Web Server instance running on default HTTP standard Port 80
WiFiServer server(80);

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG; // Output format optimized for streaming

  // Memory buffer optimization configuration depending on PSRAM access availability
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA; // Optimal dimension ratio balance for facial metrics recognition (640x480)
    config.jpeg_quality = 10;          // Range 0-63 (Lower value means higher visual clarity details)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Execute camera device initialization sequence
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera initialization failed with status error structure code: 0x%x", err);
    return;
  }
}

void handleCaptureRequest(WiFiClient client) {
  // Clear camera internal capture ring buffers to drop stale leftover frames
  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  if (fb) {
    esp_camera_fb_return(fb);
    fb = NULL;
  }
  
  // Capture fresh live snapshot view frame matrix
  fb = esp_camera_fb_get();
  if (!fb) {
    client.println("HTTP/1.1 500 Internal Server Error");
    client.println("Content-Type: text/plain");
    client.println();
    client.println("Camera capture failure event occurred inside hardware layer.");
    return;
  }

  // Write typical HTTP payload frame headers targeting our python client receiver
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: image/jpeg");
  client.print("Content-Length: ");
  client.println(fb->len);
  client.println("Connection: close");
  client.println();

  // Send binary image buffer raw stream contents safely downstream over network socket
  uint8_t *fb_buf = fb->buf;
  size_t fb_len = fb->len;
  client.write(fb_buf, fb_len);

  // Return the buffer resource structure container back onto the camera device pool memory segment
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);
  initCamera();

  // Initialize network connection interface
  WiFi.begin(ssid, password);
  Serial.print("Connecting to network station gateway topology...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi Link stabilized.");

  // Launch Local HTTP API Listener Core
  server.begin();
  Serial.print("Snapshot Capture API active. Target Endpoint URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/capture");
}

void loop() {
  // Listen for incoming transaction connections from our Python API service layer framework
  WiFiClient client = server.available();
  if (client) {
    String current_line = "";
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (current_line.length() == 0) {
            // Once HTTP structural request headers boundary trace terminates, deploy frame payload
            handleCaptureRequest(client);
            break;
          } else {
            current_line = "";
          }
        } else if (c != '\r') {
          current_line += c;
        }
      }
    }
    // Safely terminate client transaction channel allocations
    client.stop();
  }
}