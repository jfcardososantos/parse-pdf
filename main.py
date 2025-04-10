from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import os
from typing import Dict, List

app = FastAPI()

class PDFParser:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        try:
            # Tentativa com pdfplumber primeiro
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text(x_tolerance=1, y_tolerance=1) or "" for page in pdf.pages)
                if len(text) > 100:  # Threshold mínimo de texto
                    return text
            
            # Fallback para OCR se necessário
            images = convert_from_path(pdf_path, dpi=400)
            return "\n".join(pytesseract.image_to_string(img, lang='por') for img in images)
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Extraction failed: {str(e)}")

    @staticmethod
    def parse_data(text: str) -> Dict:
        # Padrões otimizados para documentos da Bahia
        header_patterns = {
            'nome': r"(?i)NOME[\s\n]*([A-ZÀ-Ú\s]+)(?=\nMATR|$)",
            'matricula': r"(?i)MATR[ÍI]CULA[\s\n]*(\d+)",
            'mes_ano': r"(REFERÊNCIA|COMPETÊNCIA)[\s\n]*(\d{2}/\d{4})"
        }

        # Extração robusta de campos
        fields = {}
        for field, pattern in header_patterns.items():
            match = re.search(pattern, text)
            fields[field] = match.group(1).strip() if match else "NÃO ENCONTRADO"

        # Extração de vantagens com tratamento de múltiplos formatos
        vantagens = []
        vantagens_section = re.split(r"(?i)VANTAGENS|PROVENTOS", text)[-1]
        
        # Padrão para linhas de vantagens (3 formatos diferentes)
        for line in vantagens_section.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Formato 1: Código | Descrição | Valor
            if match := re.search(r"^(\d{4,5})\s+([A-Z\s\/]+)\s+([\d.,]+)$", line):
                cod, desc, val = match.groups()
                vantagens.append({
                    "codigo": cod,
                    "descricao": desc.strip(),
                    "valor": float(val.replace(".", "").replace(",", "."))
                })
                
            # Formato 2: Descrição | Valor (sem código)
            elif match := re.search(r"^([A-Z][A-Z\sÀ-Ú]+)\s+([\d.,]+)$", line):
                desc, val = match.groups()
                vantagens.append({
                    "codigo": None,
                    "descricao": desc.strip(),
                    "valor": float(val.replace(".", "").replace(",", "."))
                })

        return {
            "nome_completo": fields['nome'],
            "matricula": fields['matricula'],
            "mes_ano_referencia": fields['mes_ano'],
            "vantagens": vantagens
        }

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    temp_path = "temp_parse.pdf"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        text = PDFParser.extract_text(temp_path)
        result = PDFParser.parse_data(text)
        
        if result["nome_completo"] == "NÃO ENCONTRADO":
            raise HTTPException(status_code=422, detail="Estrutura do documento não reconhecida")
            
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)