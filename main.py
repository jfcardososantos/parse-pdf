from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import uvicorn
import json

app = FastAPI()

class ContrachequeProcessor:
    def __init__(self):
        self.tessconfig = r'--oem 3 --psm 6 -l por'
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        self.vantagens_alvo = {
            '00002': 'VENCIMENTO',
            '00017': 'GRAT.A.FIS',
            '00018': 'GRAT.A.FIS JUD',
            '00146': 'AD.T.SERV',
            '00153': 'CET-H.ESP',
            '00279': 'PDF',
            '00170': 'AD.NOT.INCORP',
            '00212': 'DIF SALARIO/RRA'
        }

    def extract_data(self, pdf_path: str) -> Dict:
        """Processa o PDF e extrai os dados estruturados"""
        try:
            # Extrai texto do PDF
            text = self._extract_text(pdf_path)
            
            # Parse dos dados
            dados = {
                "nome_completo": self._extract_field(r"NOME[\s\n]*([A-ZÀ-Ú\s]+)(?=\n|MATR|$)", text),
                "matricula": self._extract_field(r"MATR[ÍI]CULA[\s\n]*(\d+)", text),
                "mes_ano_referencia": self._extract_field(r"(\d{2}/\d{4})\s+SRI-SISTEMA", text),
                "vantagens": self._extract_vantagens(text)
            }
            
            # Validação dos campos obrigatórios
            if not all([dados["nome_completo"], dados["matricula"], dados["vantagens"]]):
                raise ValueError("Campos essenciais não encontrados")
                
            return dados
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _extract_text(self, pdf_path: str) -> str:
        """Extrai texto do PDF com fallback para OCR"""
        try:
            # Tenta extração textual
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages)
                if len(text) > 100:  # Limite mínimo de texto
                    return text
                    
            # Fallback para OCR
            images = convert_from_path(pdf_path, dpi=500)
            text = "\n".join(pytesseract.image_to_string(
                self._preprocess_image(img), 
                config=self.tessconfig
            ) for img in images)
            
            return self._clean_text(text)
            
        except Exception as e:
            raise RuntimeError(f"Falha na extração de texto: {str(e)}")

    def _preprocess_image(self, image):
        """Pré-processamento da imagem para OCR"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(threshold)

    def _clean_text(self, text: str) -> str:
        """Corrige erros comuns de OCR"""
        corrections = {
            r'\bGOVERNO\s+D[EO]\s+ESTAD[OA]\b': 'GOVERNO DO ESTADO',
            r'\bMATR[ÍI]CULA\b': 'MATRÍCULA',
            r'\bContra[çc]heque\b': 'Contracheque'
        }
        for pattern, repl in corrections.items():
            text = re.sub(pattern, repl, text)
        return text

    def _extract_field(self, pattern: str, text: str) -> Optional[str]:
        """Extrai um campo usando regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_vantagens(self, text: str) -> List[Dict]:
        """Extrai e filtra as vantagens específicas"""
        vantagens = []
        
        # Encontra a seção de vantagens
        section_match = re.search(
            r"VANTAGENS.*?(cód|COD|DISCRIMINAÇÃO).*?\n(.*?)(?=TOTAL DE VANTAGENS|\n\n)",
            text, re.DOTALL | re.IGNORECASE
        )
        
        if not section_match:
            return vantagens
            
        # Processa cada linha da tabela
        for line in section_match.group(2).split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Padrão para linhas com código, descrição e valor
            if match := re.match(r"(\d{5})\s+([A-ZÀ-Ú./\s-]+?)\s+([\d.,]+)%?\s+([\d.,]+)", line):
                cod, desc, perc, val = match.groups()
            elif match := re.match(r"(\d{5})\s+([A-ZÀ-Ú./\s-]+?)\s+([\d.,]+)\s*$", line):
                cod, desc, val = match.groups()
                perc = None
            else:
                continue
                
            # Filtra apenas as vantagens desejadas
            if cod in self.vantagens_alvo:
                vantagem = {
                    "codigo": cod,
                    "descricao": self.vantagens_alvo[cod],
                    "valor": self._parse_money_value(val)
                }
                
                # Adiciona percentual se existir e não for VENCIMENTO ou PDF
                if perc and cod not in ['00002', '00279']:
                    vantagem["percentual_duracao"] = self._parse_money_value(perc)
                    
                vantagens.append(vantagem)
                
        return vantagens

    def _parse_money_value(self, value_str: str) -> float:
        """Converte valores brasileiros para float"""
        try:
            return float(value_str.replace('.', '').replace(',', '.'))
        except:
            raise ValueError(f"Valor monetário inválido: {value_str}")

@app.post("/processar-contracheque")
async def processar_contracheque(file: UploadFile = File(...)):
    """Endpoint principal que retorna os dados formatados em JSON"""
    temp_path = "temp_pdf.pdf"
    try:
        # Salva o arquivo temporariamente
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Processa o documento
        processor = ContrachequeProcessor()
        dados = processor.extract_data(temp_path)
        
        # Formata a resposta conforme especificado
        response = {
            "nome_completo": dados["nome_completo"],
            "matricula": dados["matricula"],
            "mes_ano_referencia": dados["mes_ano_referencia"],
            "vantagens": dados["vantagens"]
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)