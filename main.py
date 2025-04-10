from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import os
from typing import Dict, List
import uvicorn

app = FastAPI()

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extrai texto do PDF, usando OCR se necessário"""
        try:
            # Primeiro tenta extração textual normal
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
                if len(text) > 200:  # Se extraiu texto suficiente
                    return text

            # Fallback para OCR
            images = convert_from_path(pdf_path, dpi=400)
            text = "\n".join(pytesseract.image_to_string(img, lang='por') for img in images)
            return text

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Falha na extração: {str(e)}")

    @staticmethod
    def parse_paycheck(text: str) -> Dict:
        """Analisa o texto extraído e retorna dados estruturados"""
        try:
            # Extração dos campos principais
            nome = re.search(r"(?i)NOME[\s\n]*([A-ZÀ-Ú\s]+)(?=\n|MATR|$)", text)
            matricula = re.search(r"(?i)MATR[ÍI]CULA[\s\n]*(\d+)", text)
            mes_ano = re.search(r"(?i)REFER[ÊE]NCIA[\s\n]*(\d{2}/\d{4})", text) or \
                     re.search(r"\d{2}/\d{4}(?=\s+SRI-SISTEMA)", text)

            # Extração de vantagens
            vantagens = []
            vantagens_section = re.split(r"(?i)VANTAGENS", text)[-1][:2000]  # Limita o escopo
            
            # Padrão para linhas de vantagens (ex: "00002 VENCIMENTO 2.693,71")
            for line in vantagens_section.split('\n'):
                if match := re.search(r"(\d{4,5})\s+([A-ZÀ-Ú\s\.]+)\s+([\d\.]+\,\d{2})", line.strip()):
                    cod, desc, val = match.groups()
                    vantagens.append({
                        "codigo": cod,
                        "descricao": desc.strip(),
                        "valor": float(val.replace(".", "").replace(",", "."))
                    })

            return {
                "nome_completo": nome.group(1).strip() if nome else "NÃO ENCONTRADO",
                "matricula": matricula.group(1) if matricula else "NÃO ENCONTRADA",
                "mes_ano_referencia": mes_ano.group(1) if mes_ano else "NÃO ENCONTRADO",
                "vantagens": vantagens
            }

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro na análise: {str(e)}")

@app.post("/parse-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """Endpoint principal para processar PDFs"""
    temp_path = "temp_pdf.pdf"
    try:
        # Salva o arquivo temporariamente
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Processamento do PDF
        text = PDFProcessor.extract_text(temp_path)
        result = PDFProcessor.parse_paycheck(text)
        
        # Validação básica
        if not result["vantagens"]:
            raise HTTPException(status_code=422, detail="Nenhuma vantagem encontrada - estrutura do PDF não reconhecida")
            
        return result
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)