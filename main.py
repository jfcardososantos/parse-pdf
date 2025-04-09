from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import os
from typing import Dict
from PIL import Image

app = FastAPI()

def is_scanned_pdf(pdf_path: str) -> bool:
    """Verifica se o PDF é escaneado (imagem)"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Se extrair pouco texto, provavelmente é imagem
            text = pdf.pages[0].extract_text()
            return len(text or "") < 50  # Limite arbitrário
    except:
        return True

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai texto de PDFs textuais ou escaneados"""
    if is_scanned_pdf(pdf_path):
        try:
            images = convert_from_path(pdf_path, dpi=300)
            return "\n".join(pytesseract.image_to_string(img, lang='por') for img in images)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro no OCR: {str(e)}")
    else:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao extrair texto: {str(e)}")

def parse_paycheck(text: str) -> Dict:
    try:
        # Extração de campos básicos (regex robusta)
        nome = re.search(r"(?i)NOME[\s:]*\n(.+)", text)
        matricula = re.search(r"(?i)MATR[ÍI]CULA[\s:]*\n(\d+)", text)
        mes_ano = re.search(r"\d{2}/\d{4}", text)

        # Extração de vantagens (compatível com ambos formatos)
        vantagens = []
        for line in text.split('\n'):
            if match := re.search(r"(\d{5})\s+([A-ZÀ-Ú./\s]+?)\s+([\d.,]+)\s*$", line.strip()):
                codigo, descricao, valor = match.groups()
                vantagens.append({
                    "codigo": codigo,
                    "descricao": descricao.strip(),
                    "valor": float(valor.replace(".", "").replace(",", "."))
                })

        return {
            "nome_completo": nome.group(1).strip() if nome else "NÃO ENCONTRADO",
            "matricula": matricula.group(1).strip() if matricula else "NÃO ENCONTRADA",
            "mes_ano_referencia": mes_ano.group(0) if mes_ano else "NÃO ENCONTRADO",
            "vantagens": vantagens
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no parsing: {str(e)}")

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    temp_path = "temp_pdf.pdf"
    try:
        # Salva o arquivo temporariamente
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Processamento híbrido
        text = extract_text_from_pdf(temp_path)
        result = parse_paycheck(text)
        
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)