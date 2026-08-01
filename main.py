# Serviço de análise facial — MediaPipe Face Landmarker (478 pontos)
# POST /analyze  { "image_b64": "..." }  →  métricas + pontos em pixel para o overlay do relatório

import base64, io, math
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL = "face_landmarker.task"

app = FastAPI()
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
)

class Req(BaseModel):
    image_b64: str

# pares espelhados usados no índice de simetria (lado esq, lado dir da imagem):
# canto externo dos olhos, canto interno, cantos da boca, asas do nariz, gônio, corpo mandibular, zigomático
PARES_SIMETRIA = [(33, 263), (133, 362), (61, 291), (129, 358), (58, 288), (172, 397), (234, 454)]

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def refletir(p, a, b):
    # reflete o ponto p em torno da reta a→b (eixo de simetria násio→mento)
    ax, ay = a; bx, by = b; px_, py_ = p
    dx, dy = bx - ax, by - ay
    t = ((px_ - ax) * dx + (py_ - ay) * dy) / (dx * dx + dy * dy)
    fx, fy = ax + t * dx, ay + t * dy
    return (2 * fx - px_, 2 * fy - py_)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
def analyze(req: Req):
    b64 = req.image_b64.split(",")[-1]  # tolera prefixo data:image/...;base64,
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img = ImageOps.exif_transpose(img).convert("RGB")  # respeita a rotação EXIF do celular
    w, h = img.size
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
    if not res.face_landmarks:
        return {"face_detected": False}

    lm = res.face_landmarks[0]
    P = [(p.x * w, p.y * h) for p in lm]  # todos os 478 pontos em pixels da imagem original

    nasio, mento = P[168], P[152]
    topo_testa, glabela, subnasal = P[10], P[9], P[2]
    zig_e, zig_d = P[234], P[454]   # largura bizigomática (pontos mais laterais da face)
    gon_e, gon_d = P[58], P[288]    # ângulo da mandíbula (se quiser mais baixo, teste 172/397)

    larg_bizigomatica = dist(zig_e, zig_d)
    larg_bigonial = dist(gon_e, gon_d)
    altura_face = dist(topo_testa, mento)

    # simetria: desvio médio dos pares espelhados em torno do eixo násio→mento,
    # normalizado pela largura da face
    desvios = [dist(refletir(P[e], nasio, mento), P[d]) / larg_bizigomatica for e, d in PARES_SIMETRIA]
    simetria = max(0.0, min(100.0, 100.0 * (1 - float(np.mean(desvios)))))

    # correção de inclinação (roll) da cabeça para as medidas verticais
    olho_e, olho_d = P[33], P[263]
    roll = math.atan2(olho_d[1] - olho_e[1], olho_d[0] - olho_e[0])
    def rot_y(p):
        return -p[0] * math.sin(roll) + p[1] * math.cos(roll)

    # terços: o ponto 10 é o topo da malha (fica abaixo da linha do cabelo), então o
    # terço superior é aproximado; médio e inferior são os valores confiáveis
    y10, y9, y2, y152 = rot_y(topo_testa), rot_y(glabela), rot_y(subnasal), rot_y(mento)
    total = y152 - y10
    tercos = {
        "superior_pct": round(100 * (y9 - y10) / total, 1),
        "medio_pct": round(100 * (y2 - y9) / total, 1),
        "inferior_pct": round(100 * (y152 - y2) / total, 1),
    }

    return {
        "face_detected": True,
        "imagem": {"largura": w, "altura": h},
        "metricas": {
            "simetria_pct": round(simetria, 1),
            "proporcao_altura_largura": round(altura_face / larg_bizigomatica, 3),
            "razao_bigonial_bizigomatica": round(larg_bigonial / larg_bizigomatica, 3),
            "tercos": tercos,
        },
        # pontos em pixels para desenhar o overlay do relatório (esq/dir = lado da IMAGEM)
        "pontos": {
            "zigomatico_esq": [round(zig_e[0]), round(zig_e[1])],
            "zigomatico_dir": [round(zig_d[0]), round(zig_d[1])],
            "gonio_esq": [round(gon_e[0]), round(gon_e[1])],
            "gonio_dir": [round(gon_d[0]), round(gon_d[1])],
            "mento": [round(mento[0]), round(mento[1])],
            "nasio": [round(nasio[0]), round(nasio[1])],
        },
    }
