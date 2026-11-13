"""
services.py ("الخدمات") - shows the state of both feet, how serious it is, and what to do.

RiskEngine combines every source (pressure, thermal camera, AI camera analysis, PPG) into a
RISK STAGE per foot:
    0 Normal | 1 Watch | 2 Warning | 3 Critical
Rules (all thresholds are in config.py):
    pressure      > PRESSURE_ALERT        stage 1, stage 2 if it lasts SUSTAIN_SECONDS
    warmth        (thermal camera) left-right difference   >= 1.5 C stage 1, >= 2.2 C stage 2, >= 4 C stage 3
    camera (AI)   redness >= 3 % -> 2, dark area >= 1 % -> 3, trained-model "ulcer"/"wound" -> 3
    combination   high pressure AND warmth/redness on the same foot -> stage 3
    PPG           low perfusion index / low SpO2 are shown as general findings (do not change a foot's stage)
The stage is the HIGHEST stage found. This is decision support - NOT a medical diagnosis.
"""
from PySide6 import QtCore, QtWidgets

import config as C

STAGE_COLORS = ["#3fb950", "#d29922", "#f0883e", "#f85149"]
STAGE_NAMES = {"en": ["Normal", "Watch", "Warning", "Critical"],
               "ar": ["طبيعي", "مراقبة", "تحذير", "حرج"]}
SIDE_NAMES = {"en": {"L": "Left foot", "R": "Right foot"}, "ar": {"L": "القدم اليسرى", "R": "القدم اليمنى"}}
REGION_AR = {"Toes": "الأصابع", "Forefoot": "مقدمة القدم", "Midfoot": "منتصف القدم", "Heel": "الكعب"}

STAGE_ADVICE = {
    "en": ["No concerning signs. Keep up the daily foot check and wear suitable footwear.",
           "Small change detected. Re-check in a few hours, reduce time on your feet and look at the foot.",
           "Reduce weight on this foot, inspect it for redness, swelling, blisters or wounds, and contact your care team within 24 hours.",
           "Stop walking on this foot, keep it off the ground and contact your doctor or clinic today."],
    "ar": ["لا توجد علامات مقلقة. استمر بفحص قدميك يومياً وارتدِ حذاءً مناسباً.",
           "تم رصد تغيّر بسيط. أعد الفحص بعد ساعات، وقلّل الوقوف والمشي، وانظر إلى القدم.",
           "خفّف الوزن عن هذه القدم، وافحصها بحثاً عن احمرار أو تورم أو فقاعات أو جروح، وتواصل مع فريقك الطبي خلال 24 ساعة.",
           "توقف عن المشي على هذه القدم، ولا تضع وزنك عليها، وتواصل مع طبيبك أو العيادة اليوم."],
}

FINDING_TEXT = {
    "en": {
        "pressure_high": "High pressure at {zone}: {kpa:.0f} kPa",
        "temp_diff": "{zone} is {dt:.1f} °C warmer than the other foot",
        "thermal_cam": "Thermal camera: {dt:.1f} °C warmer than the other foot ({region})",
        "ai_red": "Camera: red area {pct:.1f}% of the foot (possible inflammation)",
        "ai_dark": "Camera: dark area {pct:.1f}% of the foot (possible wound / tissue damage)",
        "ai_detect": "AI model detected {label} ({conf:.0%})",
        "combined": "Pressure and warmth/redness together on the same foot - high risk of skin breakdown",
        "low_perfusion": "Low perfusion index ({pi:.1f}%)",
        "low_spo2": "Low oxygen saturation ({spo2:.0f}%)",
    },
    "ar": {
        "pressure_high": "ضغط مرتفع عند {zone}: {kpa:.0f} كيلوباسكال",
        "temp_diff": "{zone} أدفأ بمقدار {dt:.1f} °م من القدم الأخرى",
        "thermal_cam": "الكاميرا الحرارية: أدفأ بمقدار {dt:.1f} °م من القدم الأخرى ({region})",
        "ai_red": "الكاميرا: منطقة حمراء {pct:.1f}% من القدم (احتمال التهاب)",
        "ai_dark": "الكاميرا: منطقة داكنة {pct:.1f}% من القدم (احتمال جرح أو تلف أنسجة)",
        "ai_detect": "نموذج الذكاء الاصطناعي رصد {label} ({conf:.0%})",
        "combined": "ضغط مع دفء/احمرار في نفس القدم - خطر مرتفع لتقرّح الجلد",
        "low_perfusion": "مؤشر التروية منخفض ({pi:.1f}%)",
        "low_spo2": "تشبّع الأكسجين منخفض ({spo2:.0f}%)",
    },
}

FINDING_ADVICE = {
    "en": {
        "pressure_high": "Shift your weight, sit or lie down, and check the cushioning / insole under that area.",
        "temp_diff": "Rest the warmer foot and measure again later; a lasting difference of 2.2 °C or more can mean inflammation.",
        "thermal_cam": "Rest the warmer foot and measure again later; a lasting difference of 2.2 °C or more can mean inflammation.",
        "ai_red": "Look for redness, swelling or blisters, and avoid tight shoes.",
        "ai_dark": "Do not treat it yourself; show it to a clinician as soon as possible.",
        "ai_detect": "Do not treat it yourself; show it to a clinician as soon as possible.",
        "low_perfusion": "Weak circulation signal: keep the foot warm, avoid tight socks, and tell your doctor if it continues.",
        "low_spo2": "Check the sensor fit; if the reading stays low, seek medical advice.",
    },
    "ar": {
        "pressure_high": "غيّر توزيع وزنك، واجلس أو استلقِ، وتأكد من الحشوة / النعل تحت هذه المنطقة.",
        "temp_diff": "أرح القدم الأدفأ وأعد القياس لاحقاً؛ استمرار فرق 2.2 °م أو أكثر قد يعني التهاباً.",
        "thermal_cam": "أرح القدم الأدفأ وأعد القياس لاحقاً؛ استمرار فرق 2.2 °م أو أكثر قد يعني التهاباً.",
        "ai_red": "افحص القدم بحثاً عن احمرار أو تورم أو فقاعات، وتجنب الأحذية الضيقة.",
        "ai_dark": "لا تعالجها بنفسك؛ اعرضها على طبيب في أقرب وقت.",
        "ai_detect": "لا تعالجها بنفسك؛ اعرضها على طبيب في أقرب وقت.",
        "low_perfusion": "إشارة تروية ضعيفة: حافظ على دفء القدم، وتجنب الجوارب الضيقة، وأخبر طبيبك إذا استمر ذلك.",
        "low_spo2": "تأكد من تثبيت الحساس؛ وإذا بقيت القراءة منخفضة فاستشر طبيباً.",
    },
}
DISCLAIMER = {"en": "Decision support only - not a medical diagnosis.",
              "ar": "أداة مساندة للقرار فقط وليست تشخيصاً طبياً."}


def _fmt(lang, finding):
    p = dict(finding["params"])
    if "zi" in p:
        p["zone"] = C.ZONE_NAMES_AR[p["zi"]] if lang == "ar" else C.ZONES[p["zi"]][0]
    if "region" in p and lang == "ar":
        p["region"] = REGION_AR.get(p["region"], p["region"])
    return FINDING_TEXT[lang][finding["key"]].format(**p)


class FootAssessment:
    def __init__(self, side):
        self.side, self.findings = side, []

    def add(self, key, stage, **params):
        self.findings.append(dict(key=key, stage=stage, params=params))

    @property
    def stage(self):
        return max([f["stage"] for f in self.findings], default=0)


class RiskEngine:
    def __init__(self):
        self.p_cnt = {s: [0] * len(C.ZONES) for s in "LR"}
        self.prev = {"L": 0, "R": 0}

    @staticmethod
    def _delta_stage(d):
        if d >= C.CRITICAL_DELTA:
            return 3
        if d >= C.HOT_DELTA:
            return 2
        return 1 if d >= C.WATCH_DELTA else 0

    def update(self, data, thermal=None, ai=None, dt=0.1):
        """dt = seconds since the previous call. Returns (assessments {'L','R'}, general findings, overall stage, new events)."""
        feet = {s: FootAssessment(s) for s in "LR"}

        # pressure (FSR array)
        for s in "LR":
            p = data[s]["pressure"]
            for i, v in enumerate(p):
                self.p_cnt[s][i] = self.p_cnt[s][i] + dt if v > C.PRESSURE_ALERT else 0
            i = max(range(len(p)), key=p.__getitem__)
            if p[i] > C.PRESSURE_ALERT:
                stage = 2 if self.p_cnt[s][i] >= C.SUSTAIN_SECONDS else 1
                feet[s].add("pressure_high", stage, zi=i, kpa=p[i])

        # thermal camera
        if thermal and thermal.get("found"):
            stage = self._delta_stage(thermal["asym"])
            if stage:
                feet["L" if thermal["side"] == "Left" else "R"].add(
                    "thermal_cam", stage, dt=thermal["asym"], region=thermal["region"])

        # AI camera analysis
        if ai and ai.get("found"):
            for f in ai["feet"]:
                for s in (["L", "R"] if f["side"] == "?" else [f["side"]]):
                    if f["dark_pct"] >= C.DARK_ALERT_PCT:
                        feet[s].add("ai_dark", 3, pct=f["dark_pct"])
                    elif f["dark_pct"] >= C.DARK_WATCH_PCT:
                        feet[s].add("ai_dark", 1, pct=f["dark_pct"])
                    if f["red_pct"] >= C.RED_ALERT_PCT:
                        feet[s].add("ai_red", 2, pct=f["red_pct"])
                    elif f["red_pct"] >= C.RED_WATCH_PCT:
                        feet[s].add("ai_red", 1, pct=f["red_pct"])
            for dtc in ai.get("detections", []):
                if dtc["conf"] >= C.AI_MIN_CONF and any(k in dtc["label"].lower() for k in C.AI_ALERT_LABELS):
                    for s in (["L", "R"] if dtc["side"] == "?" else [dtc["side"]]):
                        feet[s].add("ai_detect", 3, label=dtc["label"], conf=dtc["conf"])

        # combination: pressure + warmth/redness on the same foot
        for s in "LR":
            keys = {f["key"] for f in feet[s].findings if f["stage"] >= 2}
            if "pressure_high" in keys and keys & {"thermal_cam", "ai_red"}:
                feet[s].add("combined", 3)

        # general findings (PPG)
        general = []
        if data["pi"] < C.PI_LOW:
            general.append(dict(key="low_perfusion", stage=1, params=dict(pi=data["pi"])))
        if data["spo2"] < C.SPO2_LOW:
            general.append(dict(key="low_spo2", stage=2, params=dict(spo2=data["spo2"])))

        overall = max([feet["L"].stage, feet["R"].stage] + [g["stage"] for g in general])

        events = []
        for s in "LR":
            if feet[s].stage >= 2 and feet[s].stage > self.prev[s]:
                top = max(feet[s].findings, key=lambda f: f["stage"])
                events.append(dict(side=s, stage=feet[s].stage,
                                   detail={lg: _fmt(lg, top) for lg in ("en", "ar")}))
            self.prev[s] = feet[s].stage
        return feet, general, overall, events


class ServicesPanel(QtWidgets.QWidget):
    """Both feet: stage badge, findings, advice. Language can be switched live."""
    lang_changed = QtCore.Signal(str)

    def __init__(self, lang="en"):
        super().__init__()
        self.lang = lang if lang in ("en", "ar") else "en"
        self.last = None
        self.cache = {}

        self.banner = QtWidgets.QLabel("--")
        self.banner.setTextFormat(QtCore.Qt.RichText)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["English", "العربية"])
        self.combo.setCurrentIndex(0 if self.lang == "en" else 1)
        self.combo.currentIndexChanged.connect(self._lang_changed)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.banner, 1)
        top.addWidget(self.combo)

        self.view = {}
        row = QtWidgets.QHBoxLayout()
        for s in "LR":
            tb = QtWidgets.QTextBrowser()
            tb.setStyleSheet("background:#161b22;border:1px solid #30363d;border-radius:6px;")
            tb.document().setDefaultStyleSheet("ol{margin-left:0px;-qt-list-indent:1;}")
            self.view[s] = tb
            row.addWidget(tb)

        self.foot = QtWidgets.QLabel()
        self.foot.setStyleSheet(f"color:#8b949e;font-size:{C.px(13)}px;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addLayout(row, 1)
        lay.addWidget(self.foot)

    def _lang_changed(self, idx):
        self.lang = "en" if idx == 0 else "ar"
        self.lang_changed.emit(self.lang)
        self.cache = {}
        if self.last:
            self.show(*self.last)

    def show(self, feet, general, overall):
        self.last = (feet, general, overall)
        lang = self.lang
        rtl = lang == "ar"
        P = C.px
        wrap = (lambda h: f"<div dir='rtl' style='text-align:right'>{h}</div>") if rtl else (lambda h: h)

        col = STAGE_COLORS[overall]
        word = "الحالة العامة" if rtl else "Overall"
        self.banner.setText(wrap(f"<span style='font-size:{P(24)}px;font-weight:bold;'>{word}: "
                                 f"<span style='color:{col}'>{overall} - {STAGE_NAMES[lang][overall]}</span></span>"))
        self.foot.setText(DISCLAIMER[lang])
        self.foot.setAlignment(QtCore.Qt.AlignRight if rtl else QtCore.Qt.AlignLeft)

        h_found = "ما تم رصده" if rtl else "What was found"
        h_todo = "ماذا أفعل؟" if rtl else "What to do"
        more_txt = "ملاحظات أخرى: {n}" if rtl else "+ {n} more finding(s)"
        none_txt = "لا توجد ملاحظات - كل القراءات ضمن الطبيعي" if rtl else "No findings - all readings look normal"
        head = f"font-size:{P(18)}px;font-weight:bold;color:#58a6ff;margin-top:{P(12)}px;"
        MAX_ITEMS = 5

        for s in "LR":
            st = feet[s].stage
            items = sorted(feet[s].findings, key=lambda f: -f["stage"])
            advice, seen = [STAGE_ADVICE[lang][st]], set()
            for f in items:
                a_ = FINDING_ADVICE[lang].get(f["key"])
                if a_ and a_ not in seen:
                    seen.add(a_)
                    advice.append(a_)

            # 1) who + stage   2) WHAT TO DO (most important, first)   3) what was found
            html = (f"<div style='font-size:{P(24)}px;font-weight:bold;'>{SIDE_NAMES[lang][s]}</div>"
                    f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
                    f"<td style='background:{STAGE_COLORS[st]};color:#000;font-size:{P(20)}px;font-weight:bold;"
                    f"padding:{P(7)}px {P(12)}px;'>{st} - {STAGE_NAMES[lang][st]}</td></tr></table>"
                    f"<div style='{head}'>{h_todo}</div><ol style='margin-top:4px;'>")
            html += "".join(f"<li style='font-size:{P(17)}px;margin-bottom:{P(6)}px;'>{a_}</li>" for a_ in advice) + "</ol>"
            html += f"<div style='{head}'>{h_found}</div>"
            if items:
                html += "<table cellspacing='0' cellpadding='3'>"
                html += "".join(
                    f"<tr><td width='{P(26)}' style='color:{STAGE_COLORS[f['stage']]};font-size:{P(21)}px;'>&#9679;</td>"
                    f"<td style='font-size:{P(17)}px;'>{_fmt(lang, f)}</td></tr>" for f in items[:MAX_ITEMS])
                html += "</table>"
                if len(items) > MAX_ITEMS:
                    html += f"<div style='color:#8b949e;font-size:{P(15)}px;'>{more_txt.format(n=len(items) - MAX_ITEMS)}</div>"
            else:
                html += f"<div style='color:#3fb950;font-size:{P(17)}px;margin-top:4px;'>&#10003; {none_txt}</div>"
            html = wrap(html)
            if self.cache.get(s) != html:                  # only redraw on change (keeps the scroll position)
                self.cache[s] = html
                self.view[s].setHtml(html)
        if general:
            extra = " | ".join(_fmt(lang, g) for g in general)
            self.foot.setText(f"{extra}   -   {DISCLAIMER[lang]}")