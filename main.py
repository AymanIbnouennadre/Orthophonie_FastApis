from fastapi import FastAPI, UploadFile, File
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

# Spécifie explicitement le chemin vers l'exécutable Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

# Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Permet tous les en-têtes
)

@app.post("/convert-image-to-textAR/")
async def convert_image_to_text(file: UploadFile = File(...)):
    try:
        # Lire l'image envoyée par l'utilisateur
        img_bytes = await file.read()

        # Convertir les bytes en une image OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 2. Augmenter le contraste (optionnel)
        alpha = 1.5 # Contraste
        beta = 0
        contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # 3. Appliquer un seuil adaptatif pour améliorer la binarisation
        adaptive_threshold = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

        # Convertir l'image traitée en format PIL pour l'utiliser avec Tesseract
        img_pil = Image.fromarray(adaptive_threshold)

        # Utiliser pytesseract pour extraire le texte de l'image
        config = '--psm 6 --oem 3 '  # Mode de segmentation de page
        text = pytesseract.image_to_string(img_pil, lang='ara', config=config)

        # Retourner le texte extrait dans une réponse JSON
        return {"extracted_text": text}

    except Exception as e:
        # En cas d'erreur, renvoyer un message d'erreur
        return {"error": str(e)}

@app.post("/convert-image-to-textFR/")
async def convert_image_to_text(file: UploadFile = File(...)):
    try:
        # Lire l'image envoyée par l'utilisateur
        img_bytes = await file.read()

        # Convertir les bytes en une image OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 2. Augmenter le contraste (optionnel)
        alpha = 1.5 # Contraste
        beta = 0
        contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # 3. Appliquer un seuil adaptatif pour améliorer la binarisation
        adaptive_threshold = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

        # Convertir l'image traitée en format PIL pour l'utiliser avec Tesseract
        img_pil = Image.fromarray(adaptive_threshold)

        # Utiliser pytesseract pour extraire le texte de l'image
        config = '--psm 6 --oem 3 '  # Mode de segmentation de page
        text = pytesseract.image_to_string(img_pil, lang='fra', config=config)

        # Retourner le texte extrait dans une réponse JSON
        return {"extracted_text": text}

    except Exception as e:
        # En cas d'erreur, renvoyer un message d'erreur
        return {"error": str(e)}