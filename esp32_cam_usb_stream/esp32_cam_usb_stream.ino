/*
  ESP32-CAM -> USB serial JPEG streamer
  --------------------------------------
  Sends JPEG frames over the USB cable. Each frame is:
      0xAA 0x55 0xAA 0x55 | length (4 bytes, little-endian) | JPEG bytes

  Setup (Arduino IDE):
   1. Open Elegoo's own "CameraWebServer" example once and note:
        - the Board you must select in Tools > Board
        - the CAMERA_MODEL_xxx line that matches your board
   2. Copy "camera_pins.h" from that example folder into THIS sketch folder.
   3. Set the CAMERA_MODEL_xxx define below to the same model.
   4. Tools: PSRAM = Enabled (if the board has PSRAM), upload.
   5. DO NOT use Serial.print() anywhere else - it would corrupt the frames.

  Speed: QVGA (320x240) at 921600 baud ~ 6-10 fps. If your USB-serial chip
  supports it (CH340 = 2000000) you can raise BAUD here AND in the Python side
  (--camera serial:COM5@2000000).
*/
#include "esp_camera.h"

#define CAMERA_MODEL_AI_THINKER   // <-- CHANGE to match your Elegoo board
#include "camera_pins.h"

#define BAUD 921600

void setup() {
  Serial.begin(BAUD);

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
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;     // 320x240
  config.jpeg_quality = 12;               // 10 = best, 63 = worst
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 2;
  } else {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
  }

  if (esp_camera_init(&config) != ESP_OK) {
    while (true) delay(1000);             // camera init failed (wrong CAMERA_MODEL?)
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_whitebal(s, 1);                  // auto white balance
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  const uint8_t hdr[4] = {0xAA, 0x55, 0xAA, 0x55};
  uint32_t len = fb->len;
  Serial.write(hdr, 4);
  Serial.write((uint8_t *)&len, 4);
  Serial.write(fb->buf, len);

  esp_camera_fb_return(fb);
}
