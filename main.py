from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


# ✅ Définition de Tesseract pour l'arabe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ✅ Initialisation PaddleOCR pour le français
ocr_fr = PaddleOCR(use_angle_cls=True, lang='fr')



# ✅ Initialisation FastAPI
app = FastAPI()

# ✅ Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Fonction de prétraitement des images
def preprocess_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Convertir en niveaux de gris
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Augmenter le contraste
    contrasted = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # 3. Appliquer un seuil adaptatif (binarisation)
    adaptive_threshold = cv2.adaptiveThreshold(
        contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return adaptive_threshold

# ✅ Route unique pour OCR en français avec correction
@app.post("/convert-image-to-textFR/")
async def convert_image_to_text_fr(file: UploadFile = File(...)):
    try:
        # Lire l'image envoyée par l'utilisateur
        img_bytes = await file.read()
        processed_img = preprocess_image(img_bytes)

        # Convertir en format PIL pour Tesseract
        img_pil = Image.fromarray(processed_img)

        # PaddleOCR pour le français
        results = ocr_fr.ocr(np.array(img_pil), cls=True)
        extracted_text = " ".join([line[1][0] for line in results[0]])



        return {
            "extracted_text": extracted_text.strip()
        }

    except Exception as e:
        return {"error": str(e)}

# ✅ Route unique pour OCR en arabe
@app.post("/convert-image-to-textAR/")
async def convert_image_to_text_ar(file: UploadFile = File(...)):
    try:
        # Lire l'image envoyée par l'utilisateur
        img_bytes = await file.read()
        processed_img = preprocess_image(img_bytes)

        # Convertir en format PIL pour Tesseract
        img_pil = Image.fromarray(processed_img)

        # Tesseract pour l'arabe
        extracted_text = pytesseract.image_to_string(img_pil, lang='ara', config='--psm 6 --oem 3')

        return {
            "extracted_text": extracted_text.strip()
        }

    except Exception as e:
        return {"error": str(e)}
