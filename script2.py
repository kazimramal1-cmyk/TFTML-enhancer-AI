# script2.py
# ════════════════════════════════════════════════════════════
#  TORCHVISION YAMAQ — HƏR ŞEYDƏN ƏVVƏL (ayrıca patch.py lazım deyil)
# ════════════════════════════════════════════════════════════
import sys
import types
import torchvision.transforms.functional as _F

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
for _attr in dir(_F):
    setattr(_mod, _attr, getattr(_F, _attr))
sys.modules["torchvision.transforms.functional_tensor"] = _mod

import torchvision.transforms as _transforms
_transforms.functional_tensor = _mod
# ════════════════════════════════════════════════════════════

# ── Standart kitabxanalar ────────────────────────────────────
import os
from pathlib import Path

# ── Üçüncü tərəf kitabxanalar ───────────────────────────────
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ════════════════════════════════════════════════════════════
#  QOVLUQ YOLLARI
# ════════════════════════════════════════════════════════════
BASE_DIR      = Path(__file__).resolve().parent   # bu faylın olduğu yer
TEMPLATE_DIR  = BASE_DIR / "templates"            # templates/index.html
UPLOAD_DIR    = BASE_DIR / "uploads"              # yüklənən şəkillər
OUTPUT_DIR    = BASE_DIR / "outputs"              # böyüdülmüş nəticələr
MODEL_PATH    = BASE_DIR / "RealESRGAN_x4.pb"    # model faylı

# Qovluqlar yoxdursa yarat
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════
#  FLASK TƏTBİQİ
#  template_folder ilə mütləq yol verilir → index.html həmişə tapılır
# ════════════════════════════════════════════════════════════
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


# ════════════════════════════════════════════════════════════
#  MODELİ YÜKLƏ
# ════════════════════════════════════════════════════════════
def load_model() -> RealESRGANer:
    """RealESRGAN_x4.pb modelini yükləyir və RealESRGANer qaytarır."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model faylı tapılmadı: {MODEL_PATH}\n"
            "RealESRGAN_x4.pb faylını script2.py ilə eyni qovluğa qoy."
        )

    backbone = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=str(MODEL_PATH),
        model=backbone,
        tile=0,          # böyük şəkillər üçün 256 və ya 512 et
        tile_pad=10,
        pre_pad=0,
        half=False,      # CPU istifadə edirsənsə False saxla
    )
    return upsampler


# Proqram başlayanda modeli bir dəfə yüklə
print("[INFO] Model yüklənir, zəhmət olmasa gözləyin...")
upsampler = load_model()
print(f"[INFO] Model uğurla yükləndi: {MODEL_PATH.name}")


# ════════════════════════════════════════════════════════════
#  ROUTES (SƏHIFƏLƏR)
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """
    templates/index.html faylını render edir.
    template_folder mütləq yol olduğu üçün həmişə tapılır.
    """
    return render_template("index.html")


@app.route("/upscale", methods=["POST"])
def upscale():
    """
    POST /upscale
    Form-data: image = <şəkil faylı>
    Cavab: { "result": "/outputs/<fayl_adı>" }
    """
    # ── 1. Faylı yoxla ──────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "Şəkil göndərilməyib (açar: 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Fayl seçilməyib"}), 400

    # ── 2. Faylı saxla ──────────────────────────────────────
    safe_name   = Path(file.filename).name           # yol injection qarşısını al
    input_path  = UPLOAD_DIR / safe_name
    output_name = f"upscaled_{safe_name}"
    output_path = OUTPUT_DIR / output_name

    file.save(str(input_path))

    # ── 3. Şəkli oxu ────────────────────────────────────────
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({"error": "Şəkil oxuna bilmədi. Dəstəklənən format: jpg, png, bmp"}), 400

    # ── 4. Böyüt ────────────────────────────────────────────
    try:
        output, _ = upsampler.enhance(img, outscale=4)
    except Exception as ex:
        return jsonify({"error": f"Model xətası: {str(ex)}"}), 500

    # ── 5. Nəticəni saxla ───────────────────────────────────
    cv2.imwrite(str(output_path), output)

    return jsonify({
        "result":  f"/outputs/{output_name}",
        "message": "Şəkil uğurla böyüdüldü!"
    })


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Böyüdülmüş şəkilləri brauzerdə göstər / endir."""
    return send_from_directory(str(OUTPUT_DIR), filename)


# ════════════════════════════════════════════════════════════
#  BAŞLAT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[INFO] Template qovluğu : {TEMPLATE_DIR}")
    print(f"[INFO] Upload qovluğu   : {UPLOAD_DIR}")
    print(f"[INFO] Output qovluğu   : {OUTPUT_DIR}")
    print("[INFO] Flask serveri başladı → http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
