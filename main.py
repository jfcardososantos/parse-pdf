from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_path
import pytesseract
import re
import os
from typing import Dict, List
import uvicorn
from PIL import Image
import logging

app = FastAPI()

class PDFProcessor:
    def __init__(self):
        self.tessconfig = r'--oem 3 --psm 6 -l por'
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
    def extract_full_text(self, pdf_path: str) -> str:
        """Extrai todo o texto do PDF sem truncar"""
        try:
            images = convert_from_path(pdf_path, dpi=400, grayscale=True)
            full_text = ""
            for img in images:
                text = pytesseract.image_to_string(img, config=self.tessconfig)
                full_text += self._clean_text(text) + "\n"
            return full_text
        except Exception as e:
            logging.error(f"Erro na extração: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Corrige erros comuns do OCR"""
        corrections = {
            r'\b16i/í120t9\b': '16/11/2019',
            r'\bConfa\.lÉgue\b': 'Contracheque',
            r'\bGOVERNO\s+D[EO]\s+ESTAD[OA]\b': 'GOVERNO DO ESTADO',
            r'\bRAM tu,\b': '',
            r'\bPASSA\b': ''
        }
        for pattern, repl in corrections.items():
            text = re.sub(pattern, repl, text)
        return text

    def parse_paycheck(self, text: str) -> Dict:
        """Extrai dados estruturados do texto completo"""
        try:
            # Extração dos campos principais
            data = {
                "data_referencia": self._extract_field(r"(\d{2}/\d{2}/\d{4})", text),
                "nome_completo": self._extract_field(r"NOME[\s\n]*([A-ZÀ-Ú\s]+)(?=\n|MATR|$)", text),
                "matricula": self._extract_field(r"MATR[ÍI]CULA[\s\n]*(\d+)", text),
                "orgao": self._extract_field(r"ÓRGÃO/ENTIDADE[\s\n]*([^\n]+)", text),
                "vantagens": self._extract_table(text, "VANTAGENS", "DESCONTOS")
            }
            return data
        except Exception as e:
            logging.error(f"Erro no parsing: {str(e)}")
            raise

    def _extract_field(self, pattern: str, text: str) -> str:
        """Extrai um campo específico usando regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else "NÃO ENCONTRADO"

    def _extract_table(self, text: str, start_marker: str, end_marker: str) -> List[Dict]:
        """Extrai dados tabulares"""
        table_section = re.search(
            f"{start_marker}.*?{end_marker}(.*?)(?:\n\s*\n|\Z)",
            text, re.DOTALL
        )
        
        if not table_section:
            return []
            
        table_text = table_section.group(1)
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        items = []
        for line in lines:
            if re.match(r"\d{5}", line):  # Linhas que começam com código
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 3:
                    items.append({
                        "codigo": parts[0],
                        "descricao": parts[1],
                        "valor": parts[-1].replace(".", "").replace(",", ".")
                    })
        return items

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """Endpoint que retorna tanto o texto completo quanto os dados estruturados"""
    temp_path = "temp_pdf.pdf"
    try:
        # Salva o arquivo temporariamente
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        processor = PDFProcessor()
        
        # Extração do texto completo
        full_text = processor.extract_full_text(temp_path)
        
        # Extração de dados estruturados
        parsed_data = processor.parse_paycheck(full_text)
        
        return {
            "texto_completo": full_text,
            "dados_estruturados": parsed_data
        }

    except Exception as e:
        return {"erro": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)