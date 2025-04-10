from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import os
from typing import Dict, List, Optional
import uvicorn

app = FastAPI()

class PDFParser:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extrai texto com fallback para OCR"""
        try:
            # Tentativa 1: Extração normal
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text(x_tolerance=1, y_tolerance=1) or "" for page in pdf.pages)
                if len(text) > 300:
                    return text

            # Tentativa 2: OCR com pré-processamento
            images = convert_from_path(pdf_path, dpi=400, grayscale=True)
            custom_config = r'--oem 3 --psm 6 -l por'
            text = "\n".join(pytesseract.image_to_string(img, config=custom_config) for img in images)
            return text

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Falha na extração: {str(e)}")

    @staticmethod
    def parse_ba_paycheck(text: str) -> Optional[Dict]:
        """Parser específico para contracheques da Bahia"""
        try:
            # Padrões atualizados para documentos baianos
            patterns = {
                'nome': r"(?i)NOME\s*\n([A-ZÀ-Ú\s]+)(?=\n|MATR|$)",
                'matricula': r"(?i)MATR[ÍI]CULA\s*\n(\d+)",
                'mes_ano': r"(COMPET[ÊE]NCIA|REFER[ÊE]NCIA)\s*\n(\d{2}/\d{4})",
                'vantagens': r"(?i)(VANTAGENS|PROVENTOS).*?\n(.*?)(?=\n\s*TOTAL)"
            }

            # Extração dos campos
            fields = {}
            for field, pattern in patterns.items():
                match = re.search(pattern, text, re.DOTALL)
                fields[field] = match.group(1).strip() if match else None

            # Processamento especial para vantagens
            vantagens = []
            if fields['vantagens']:
                for line in fields['vantagens'].split('\n'):
                    if match := re.search(r"(\d{4,5})\s+([A-ZÀ-Ú\s/]+)\s+([\d.,]+)", line.strip()):
                        vantagens.append({
                            "codigo": match.group(1),
                            "descricao": match.group(2).strip(),
                            "valor": float(match.group(3).replace(".", "").replace(",", "."))
                        })

            return {
                "nome_completo": fields['nome'] or "NÃO ENCONTRADO",
                "matricula": fields['matricula'] or "NÃO ENCONTRADA",
                "mes_ano_referencia": fields['mes_ano'] or "NÃO ENCONTRADO",
                "vantagens": vantagens
            }

        except Exception:
            return None

    @staticmethod
    def fallback_parser(text: str) -> Dict:
        """Parser genérico para documentos não identificados"""
        # Implementação simplificada como fallback
        nome = re.search(r"(?i)(NOME|NOME DO SERVIDOR)[\s:]*\n([A-Z\s]+)", text)
        matricula = re.search(r"(?i)(MATR[ÍI]CULA|MATRICULA)[\s:]*\n(\d+)", text)
        
        return {
            "nome_completo": nome.group(2).strip() if nome else "NÃO ENCONTRADO",
            "matricula": matricula.group(2) if matricula else "NÃO ENCONTRADA",
            "mes_ano_referencia": "NÃO IDENTIFICADO",
            "vantagens": []
        }

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    temp_path = "temp_pdf.pdf"
    try:
        # Salvar arquivo temporário
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extração de texto
        text = PDFParser.extract_text(temp_path)
        
        # Tentativa com parser específico
        result = PDFParser.parse_ba_paycheck(text)
        
        # Fallback se necessário
        if not result or not result['vantagens']:
            result = PDFParser.fallback_parser(text)
            if not result['nome_completo'] or result['nome_completo'] == "NÃO ENCONTRADO":
                raise HTTPException(
                    status_code=422,
                    detail="Estrutura do documento não reconhecida. Envie um contracheque padrão do Governo da Bahia"
                )
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/debug-pdf")
async def debug_pdf(file: UploadFile = File(...)):
    """Endpoint para diagnóstico"""
    temp_path = "temp_debug.pdf"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        with pdfplumber.open(temp_path) as pdf:
            metadata = {
                "pages": len(pdf.pages),
                "is_scanned": len(pdf.pages[0].chars) < 50 if pdf.pages[0].chars else True
            }
        
        text = PDFParser.extract_text(temp_path)
        return {
            "metadata": metadata,
            "text_sample": text[:1000] + "..." if len(text) > 1000 else text
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)