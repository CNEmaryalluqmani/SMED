# MEDthon - Diabetic Foot Monitor

## Run
    pip install -r requirements.txt
    python main.py                      # everything simulated, no hardware needed

Options: `--windowed` (maximised window instead of full screen), `--scenario normal|hotspot|overload|ulcer`, `--camera demo|serial:PORT|webcam:0|file:x.jpg|http://...`,
`--thermal sim|pseudo|mlx90640`, `--model foot_model.pt`, `--lang en|ar`
You can also change the simulated situation live from the box at the top of the window.

## Layout
* **Sensors panel** (left side, independent - F3 hides / shows it): ONE tile per sensor - big value, a coloured range bar with a
  marker, a short history line and a badge showing where the value comes from (SIMULATED / LIVE / FILE).
  Tiles: pressure (FSR), weight (load cell), pulse (PPG), ambient (DHT11), thermal camera, camera analysis, vibration motor, UV disinfection.
  When you connect the Arduino: make `SensorHub.read()` in sensors.py return the real value under the same key
  (`fsr`, `bpm`/`spo2`/`pi`/`ppg_samples`, `dht_t`/`dht_h`) and set the badge to LIVE in `SensorHub.sources` -
  the tile follows automatically. (The maps and 3D feet still use the simulated 12-zone array view.)
* **Dashboard**: top row = thermal analysis and AI analysis (same size); bottom row (same height) = pressure map, the **pulse column**
  (live waveform on top, pulse / SpO2 / perfusion values directly under it) and the **"Weight, UV and LRA" column** (weight with a scale picture,
  UV disinfection with a lamp picture, and a simple vibration-motor reading: level %, state, bar, and simulated frequency / acceleration / current
  while it buzzes). While a foot stays at WARNING or worse and a person is on the mat, the motor repeats its buzz every `LRA_COOLDOWN` seconds.
  Then 3D feet and services.
* **Vibration motor (LRA)**: its simulated value (vibration level 0-100 %) is shown as a small card in the "Weight, UV and LRA" column
  and as a tile in the side Sensors panel. An alert still makes the motor buzz by itself (settings: `strength`, `pattern`, `duration_ms` of `LRAPanel`,
  which now only holds the motor logic). Its model is `LRAActuator` in sensors.py (`telemetry()["level"]` = the simulated value).
* **Ambient (DHT11)**: shown ONLY in the header (thermometer + droplet) and in the side Sensors panel - no separate panel in the
  middle of the dashboard any more (the pulse column sits beside the pressure map).
* 3D feet: solid light-beige feet standing on a grid (the sensor mat). The colours on the mat under each foot are the
  pressure readings (blue = low, red = high) and show as a coloured rim around each foot.
  "Feet: See-through" or the "Under" view shows the whole map under the foot.
* **Weight (load cell + HX711)**: tile with a scale picture (dial + needle, foot prints when someone stands on it). Weight >= `WEIGHT_PRESENT_KG`
  (config.py) means a person is on the mat.
* **UV disinfection (UV-C lamp)**: an output with a SAFETY LOCK - the lamp is only allowed on while the mat is EMPTY. States: LOCKED (person on the mat,
  red padlock) -> WAITING (mat empty, `UV_START_DELAY`) -> UV ON (purple glow and rays, `UV_CYCLE_SECONDS`) -> CLEAN (green tick).
  Try it with the "Person on the mat" tick box in the header (it also sets the simulated weight and pressure). On real hardware add a hardware
  interlock too; never rely on software alone for UV-C.
* The pulse rate and SpO2 are also shown in the header, so you always see them.
* Pictures and text scale with the screen. On a very small screen the dashboard gets scroll bars.

## Full screen
The window opens full screen and scales its panels to the screen size.
F11 = switch full screen on/off, Esc = leave full screen, Ctrl+Q = quit, F3 = sensors panel.

## Files
| File | Job |
|---|---|
| main.py | builds the window from the modules below and runs it |
| config.py | sizes, thresholds, zones - tune everything here |
| camera.py | camera input (demo / sample image or video / your own file / USB ESP32-CAM / webcam / stream) + the panel with two slots: thermal analysis and AI analysis, with a source selector |
| make_samples.py | recreates the files in samples/ (optional) |
| samples/ | demo_feet.mp4 (15 s story: normal -> inflamed -> wound) and 3 sample images |
| thermal_camera.py | thermal camera code (simulated + MLX90640 skeleton) and the left-vs-right diabetic-foot analysis |
| ai_analysis.py | AI / image analysis of the foot (redness, dark areas, optional trained YOLO model) |
| services.py | status of both feet, risk stage 0-3, findings and advice (English / Arabic) |
| foot_3d.py | interactive 3D feet only (solid feet on a grid mat, colour map under them) |
| sensors.py | models of each sensor with simulated values (FSR, load cell, PPG, DHT11, LRA, UV) |
| sensor_fields.py | the on-screen field for each sensor (maps, pulse waveform, vibration panel, sensor tiles) |
| foot_geometry.py | shared foot shape and colour maps |
| widgets.py | the picture widget that fills the space it is given |
| esp32_cam_usb_stream/ | Arduino sketch for the ESP32 camera (later stage) |

## Choosing what the camera shows (inside the window)
Above the two camera slots:
* **Camera source**: Demo | Sample video | Sample image (normal / inflamed / wound) | My image file... | My video file... |
  Live USB camera (ESP32-CAM) | Live webcam | Live network stream.
  For the live options type the port / number / URL in the box and press Connect.
* **Thermal slot**: Simulated thermal camera | Redness map from the camera | MLX90640 thermal camera.
  When you play an image, video or live camera, choose "Redness map from the camera" so the thermal slot matches it.
* "Simulated situation" (top of the window) drives the simulated sensors, the demo camera and the simulated thermal camera.

## Going from simulation to real hardware (later)
* Arduino Nano sensors: replace the classes in `sensors.py` (one sensor = one class) and `LRAActuator`.
* Real thermal camera: `--thermal mlx90640` (skeleton, untested) or write your own class in `thermal_camera.py`.
* ESP32 camera over USB: upload the sketch, then `--camera serial:/dev/ttyUSB0`.

Screening aid / decision support only - not a medical diagnosis.p