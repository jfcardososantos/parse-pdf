from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import os
import cv2
import numpy as np
from typing import List, Dict
from PIL import Image
import uvicorn

app = FastAPI()

class VantagensExtractor:
    def __init__(self):
        self.tessconfig = r'--oem 3 --psm 6 -l por'
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        self.money_pattern = re.compile(r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)')  # Padrão para valores BR

    def extract_vantagens_section(self, pdf_path: str) -> Dict:
        """Extrai especificamente a seção de vantagens"""
        try:
            # Tenta extração textual primeiro
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if "VANTAGENS" in text:
                        return self._parse_from_text(text)
            
            # Fallback para OCR se necessário
            images = convert_from_path(pdf_path, dpi=500)  # DPI aumentado
            for img in images:
                processed_img = self._preprocess_image(img)
                text = pytesseract.image_to_string(processed_img, config=self.tessconfig)
                if "VANTAGENS" in text:
                    return self._parse_from_text(text)
            
            raise HTTPException(status_code=400, detail="Seção VANTAGENS não encontrada")
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def _preprocess_image(self, image):
        """Melhora a qualidade da imagem para OCR"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(threshold)

    def _parse_money_value(self, value_str: str) -> float:
        """Converte valores brasileiros (1.234,56) para float"""
        try:
            clean_value = value_str.replace('.', '').replace(',', '.')
            return float(clean_value)
        except:
            raise ValueError(f"Valor monetário inválido: {value_str}")

    def _parse_from_text(self, text: str) -> Dict:
        """Analisa o texto extraído para obter as vantagens"""
        # Encontra a seção de vantagens
        vantagens_section = re.search(
            r"VANTAGENS.*?(cód|COD).*?\n(.*?)(?=TOTAL DE VANTAGENS|\n\n)",
            text, re.DOTALL | re.IGNORECASE
        )
        
        if not vantagens_section:
            raise HTTPException(status_code=422, detail="Formato da tabela de vantagens não reconhecido")
        
        lines = [line.strip() for line in vantagens_section.group(2).split('\n') if line.strip()]
        
        vantagens = []
        for line in lines:
            # Padrão para linhas de vantagens (código, descrição, valor)
            if match := re.match(r"(\d{5})\s+([A-ZÀ-Ú./\s]+?)\s+([\d.,]+)\s*$", line):
                try:
                    vantagens.append({
                        "codigo": match.group(1),
                        "descricao": match.group(2).strip(),
                        "valor": self._parse_money_value(match.group(3))
                    })
                except ValueError as e:
                    continue
            
            # Padrão alternativo para linhas com percentual
            elif match := re.match(r"(\d{5})\s+([A-ZÀ-Ú./\s]+?)\s+([\d.,]+)%?\s+([\d.,]+)", line):
                try:
                    vantagens.append({
                        "codigo": match.group(1),
                        "descricao": match.group(2).strip(),
                        "percentual": float(match.group(3).replace(",", ".")),
                        "valor": self._parse_money_value(match.group(4))
                    })
                except ValueError as e:
                    continue
        
        if not vantagens:
            raise HTTPException(status_code=422, detail="Nenhuma vantagem válida encontrada")
        
        return {"vantagens": vantagens}

@app.post("/extrair-vantagens")
async def extrair_vantagens(file: UploadFile = File(...)):
    """Endpoint dedicado à extração da tabela de vantagens"""
    temp_path = "temp_pdf.pdf"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        extractor = VantagensExtractor()
        return extractor.extract_vantagens_section(temp_path)
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)