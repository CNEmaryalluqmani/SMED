# SMED – Smart Mat for Early Detection

A smart-mat dashboard that helps catch diabetic foot problems early. It combines pressure, weight, pulse, temperature and camera analysis for **both feet** and turns them into a simple risk stage (0–3) with clear findings and advice, in English and Arabic.

Built for the **MEDthon** hackathon. Runs on a Raspberry Pi (Arduino Nano for the sensors). All sensors have been tested on real hardware, and the dashboard also runs with **simulated sensors by default**, so you can try everything without any hardware.

## The problem

Diabetic foot is a serious complication of diabetes that can progress silently to ulcers, severe infections, lower-limb amputation and death.

Pain is our body's alarm system, but for diabetic patients this alarm is broken.

Delayed detection of early warning signs can allow preventable foot damage to progress to life-threatening complications. SMED watches the feet continuously and warns early, before the damage is serious.

## What it does

- **Pressure map** of both feet (12-zone FSR array) and an interactive **3D view** of the feet standing on the sensor mat.
- **Thermal analysis** comparing the left foot against the right to spot hot areas.
- **AI image analysis** of the foot camera (redness, dark areas, optional trained YOLO model).
- **Pulse, SpO₂ and perfusion** from a PPG sensor, with a live waveform.
- **Weight** from a load cell (HX711); the system knows when a person is standing on the mat.
- **Ambient** temperature and humidity (DHT11).
- **Vibration alert (LRA motor)** that buzzes while a foot stays at WARNING or worse.
- **UV-C disinfection** of the mat with a safety lock: the lamp can only turn on while the mat is empty.
- **Risk stage 0–3** per foot with findings and advice in **English / Arabic**.

## Quick start

```bash
pip install -r requirements.txt
python main.py          # everything simulated, no hardware needed
```

Useful options:

| Option | Values |
|---|---|
| `--windowed` | maximised window instead of full screen |
| `--scenario` | `normal` \| `hotspot` \| `overload` \| `ulcer` |
| `--camera` | `demo` \| `serial:PORT` \| `webcam:0` \| `file:x.jpg` \| `http://...` |
| `--thermal` | `sim` \| `pseudo` \| `mlx90640` |
| `--model` | path to a trained model, e.g. `foot_model.pt` |

You can also change the simulated situation live from the box at the top of the window.

**Keys:** `F11` full screen on/off · `Esc` leave full screen · `F3` show/hide the sensors panel · `Ctrl+Q` quit.

## Hardware

| Function | Component | Status |
|---|---|---|
| Foot pressure | FSR array (12 zones) | tested |
| Weight | Load cell + HX711 | tested |
| Pulse / SpO₂ | PPG sensor | tested |
| Ambient | DHT11 | tested |
| Thermal imaging | MLX90640 | tested (`--thermal mlx90640`) |
| Foot camera | ESP32-CAM over USB | tested (sketch included) |
| Vibration alert | LRA motor | tested |
| Disinfection | UV-C lamp | tested, with software safety lock |

Controller: **Raspberry Pi** (dashboard) + **Arduino Nano** (sensors). The dashboard shows live readings from the sensors (marked `LIVE`) and uses simulated values when no hardware is connected. Each sensor is one class in `sensors.py`.

> **UV-C safety:** the software lock is not enough on real hardware – always add a hardware interlock too.

## Using real hardware

- **Arduino sensors:** make `SensorHub.read()` in `sensors.py` return the real value under the same key (`fsr`, `bpm` / `spo2` / `pi` / `ppg_samples`, `dht_t` / `dht_h`) and set the badge to `LIVE` in `SensorHub.sources`. The tiles follow automatically.
- **Thermal camera:** `--thermal mlx90640`, or write your own class in `thermal_camera.py`.
- **ESP32-CAM:** upload `esp32_cam_usb_stream/esp32_cam_usb_stream.ino`, then run with `--camera serial:/dev/ttyUSB0`.

## Camera source (inside the window)

Above the two camera slots you can choose what the camera shows: demo, sample video, sample images (normal / inflamed / wound), your own image or video file, live USB camera (ESP32-CAM), live webcam, or a network stream. For live sources type the port / number / URL and press **Connect**.

In the thermal slot choose *Simulated thermal camera*, *Redness map from the camera* or *MLX90640*. When you play an image, video or live camera, pick *Redness map from the camera* so the thermal slot matches it.

## Project files

| File | Job |
|---|---|
| `main.py` | builds the window from the modules below and runs it |
| `config.py` | sizes, thresholds, zones – tune everything here |
| `camera.py` | camera input and the thermal / AI analysis panel with a source selector |
| `thermal_camera.py` | thermal camera (simulated + MLX90640) and left-vs-right analysis |
| `ai_analysis.py` | image analysis of the foot (redness, dark areas, optional YOLO model) |
| `services.py` | status of both feet, risk stage 0–3, findings and advice (EN / AR) |
| `sensors.py` | one model per sensor with simulated values (FSR, load cell, PPG, DHT11, LRA, UV) |
| `sensor_fields.py` | on-screen fields for each sensor (maps, pulse waveform, vibration panel, tiles) |
| `foot_3d.py` | interactive 3D feet on a grid mat with the colour map under them |
| `foot_geometry.py` | shared foot shape and colour maps |
| `widgets.py` | picture widget that fills the space it is given |
| `make_samples.py` | recreates the files in `samples/` (optional) |
| `samples/` | `demo_feet.mp4` (15 s story: normal → inflamed → wound) and 3 sample images |
| `esp32_cam_usb_stream/` | Arduino sketch for the ESP32 camera |

## Team

| Name | Role |
|---|---|
| Maryam Saleh Alluqmani | Prototype Design & Sensor Integration |
| Hala Mohammed Abousamak | Market Research & Project Feasibility |
| Lina Agil Alattas | Project Lead & Coordination |
| Lama Sultan Alghamdi | Medical Research & Clinical Applications |
| Amjad Dhaifalluh Alotaibi | AI & Application Design |
