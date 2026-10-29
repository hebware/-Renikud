
# Renikud Service x64 - Offline Hebrew G2P
# Requirements (install once with internet):
#   pip install renikud-onnx onnxruntime fastapi uvicorn
# Then run offline:
#   python renikud_service.py
# Will listen on http://localhost:8080

from fastapi import FastAPI
from pydantic import BaseModel
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")

app = FastAPI(title="Renikud x64 Service")

class Req(BaseModel):
    text: str

g2p = None
try:
    from renikud_onnx import G2P
    if os.path.exists(MODEL_PATH):
        g2p = G2P(MODEL_PATH)
    else:
        g2p = G2P()  # will download on first run if internet
    print(f"Loaded model: {MODEL_PATH if os.path.exists(MODEL_PATH) else 'auto-download'}")
except Exception as e:
    print(f"Failed to load G2P: {e}")
    g2p = None

@app.post("/phonemize")
def phonemize(req: Req):
    if g2p:
        phonemes = g2p.phonemize(req.text)
        return {"text": req.text, "phonemes": phonemes}
    else:
        return {"text": req.text, "phonemes": req.text, "warning": "G2P not loaded, passthrough"}

@app.get("/")
def root():
    return {"status": "Renikud x64 running", "model_loaded": g2p is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
