"""
config.py - MEDthon shared settings.
Every size, threshold and zone definition lives here, so you tune the project in one place.
"""

LANGUAGE = "en"                 # "en" or "ar" (can also be switched live inside the window)

# ---- internal picture resolution (the window scales the pictures to fit the screen) ----
CAM_W, CAM_H = 640, 480         # camera / thermal analysis pictures
MAP_W, MAP_H = 240, 480         # each foot heat map

# ---- foot geometry ------------------------------------------------------------
FOOT_LEN = 26.0                 # cm

# Zones: name, (x_cm, y_cm). x: medial(-) -> lateral(+), y: heel(0) -> toe(26). Right-foot frame.
ZONES = [
    ("Heel",             (0.0, 3.5)),
    ("Lateral midfoot",  (1.3, 11.0)),
    ("Metatarsal 1",     (-1.6, 19.0)),
    ("Metatarsal 3",     (0.0, 19.5)),
    ("Metatarsal 5",     (2.3, 18.0)),
    ("Hallux (big toe)", (-0.9, 24.2)),
]
ZONE_NAMES_AR = ["الكعب", "منتصف القدم الجانبي", "مشط 1", "مشط 3", "مشط 5", "الإبهام"]

# ---- temperature --------------------------------------------------------------
TMIN, TMAX = 22.0, 38.0                 # thermal camera colour range (C)
WATCH_DELTA = 1.5                       # left-right difference (C) -> watch
HOT_DELTA = 2.2                         # left-right difference (C) -> warning (common clinical threshold)
CRITICAL_DELTA = 4.0                    # left-right difference (C) -> critical

# ---- pressure -----------------------------------------------------------------
PRESSURE_MAX = 300.0                    # kPa (colour scale)
PRESSURE_ALERT = 200.0                  # kPa
SUSTAIN_SECONDS = 3.0                   # a finding must last this long to count as sustained

# ---- vitals (PPG) -------------------------------------------------------------
PI_LOW = 0.5                            # perfusion index (%) below this = poor perfusion
SPO2_LOW = 92.0                         # % below this = low saturation
PPG_WINDOW = 300                        # samples shown (6 s at 50 Hz)

# ---- weight (load cell + HX711) and UV disinfection --------------------------
WEIGHT_MAX = 150.0                      # kg (end of the dial / range bar)
WEIGHT_PRESENT_KG = 5.0                 # at or above this weight: a person is on the mat -> UV is locked OFF
SIM_PERSON_KG = 78.0                    # simulated person
UV_START_DELAY = 2.0                    # seconds the mat must stay empty before the UV lamp may switch on
UV_CYCLE_SECONDS = 15.0                 # length of one disinfection cycle (short for the demo)

# ---- LRA vibration ------------------------------------------------------------
LRA_COOLDOWN = 10.0                     # seconds between automatic vibrations

# ---- AI / image analysis (camera) ---------------------------------------------
MIN_FOOT_FRAC = 0.03                    # a skin blob must cover >= 3 % of the frame
RED_DELTA = 10.0                        # Lab a* above the foot median that counts as "redder"
DARK_RATIO = 0.55                       # darker than 55 % of the median brightness
MIN_REGION_FRAC = 0.004                 # ignore regions smaller than 0.4 % of the foot
RED_WATCH_PCT, RED_ALERT_PCT = 0.8, 3.0
DARK_WATCH_PCT, DARK_ALERT_PCT = 0.3, 1.0
AI_LEFT_IS_IMAGE_LEFT = True            # camera looks at the feet with toes up -> patient's left foot is on the left
AI_ALERT_LABELS = ("ulcer", "wound", "gangrene", "necrosis")   # YOLO class names treated as serious
AI_MIN_CONF = 0.5


# ---- text / widget scale (set by main.py from the screen size; 1.0 = design size 1920 x 1080) ----
UI_SCALE = 1.0


def px(n):
    """Scale a size in pixels (for text in the rich-text panels)."""
    return max(1, int(round(n * UI_SCALE)))