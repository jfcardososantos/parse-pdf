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

app = FastAPI()

class ContrachequeProcessor:
    def __init__(self):
        self.tessconfig = r'--oem 3 --psm 6 -l por'
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        # Mapeamento completo das vantagens alvo
        self.vantagens_alvo = {
            '00002': {'descricao': 'VENCIMENTO', 'tem_percentual': False},
            '00017': {'descricao': 'GRAT.A.FIS', 'tem_percentual': True},
            '00018': {'descricao': 'GRAT.A.FIS JUD', 'tem_percentual': True},
            '00146': {'descricao': 'AD.T.SERV', 'tem_percentual': True},
            '00153': {'descricao': 'CET-H.ESP', 'tem_percentual': True},
            '00279': {'descricao': 'PDF', 'tem_percentual': False},
            '00170': {'descricao': 'AD.NOT.INCORP', 'tem_percentual': True},
            '00212': {'descricao': 'DIF SALARIO/RRA', 'tem_percentual': True}
        }

    def processar_documento(self, pdf_path: str) -> Dict:
        """Processa o documento e retorna os dados estruturados"""
        try:
            text = self._extrair_texto(pdf_path)
            
            # Extração dos campos com fallback
            dados = {
                "nome_completo": self._extrair_campo(r"NOME[\s\n]*([A-ZÀ-Ú\s]+)(?=\n|MATR|ADMISSÃO|$)", text),
                "matricula": self._extrair_campo(r"MATR[ÍI]CULA[\s\n]*(\d+)", text),
                "mes_ano_referencia": self._extrair_campo(r"(?:REFERÊNCIA|COMPETÊNCIA)[\s\n]*(\d{2}/\d{4})", text) 
                          or self._extrair_campo(r"\d{2}/\d{4}(?=\s+SRI-SISTEMA)", text),
                "vantagens": self._extrair_vantagens(text)
            }
            
            # Remove campos vazios/nulos
            return {k: v for k, v in dados.items() if v is not None and v != []}
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")

    def _extrair_texto(self, pdf_path: str) -> str:
        """Extrai texto com fallback para OCR"""
        try:
            # Tenta extração textual primeiro
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                if len(text.strip()) > 100:  # Limite mínimo de texto
                    return text
                    
            # Fallback para OCR
            images = convert_from_path(pdf_path, dpi=400, grayscale=True)
            text = "\n".join(pytesseract.image_to_string(
                self._preprocessar_imagem(img), 
                config=self.tessconfig
            ) for img in images)
            
            return self._corrigir_texto(text)
            
        except Exception as e:
            raise RuntimeError(f"Falha na extração de texto: {str(e)}")

    def _preprocessar_imagem(self, image):
        """Melhora a qualidade da imagem para OCR"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(threshold)

    def _corrigir_texto(self, text: str) -> str:
        """Corrige erros comuns de OCR"""
        correcoes = {
            r'\bGOVERNO\s+D[EO]\s+ESTAD[OA]\b': 'GOVERNO DO ESTADO',
            r'\bMATR[ÍI]CULA\b': 'MATRÍCULA',
            r'\bContra[çc]heque\b': 'Contracheque',
            r'\b(\d)o\b': r'\1º',
            r'\bSRH-?\b': 'SRI-'
        }
        for padrao, substituicao in correcoes.items():
            text = re.sub(padrao, substituicao, text)
        return text

    def _extrair_campo(self, padrao: str, text: str) -> Optional[str]:
        """Extrai um campo específico com tratamento de erro"""
        try:
            match = re.search(padrao, text, re.IGNORECASE)
            return match.group(1).strip() if match else None
        except:
            return None

    def _extrair_vantagens(self, text: str) -> List[Dict]:
        """Extrai e filtra as vantagens específicas"""
        vantagens = []
        
        # Encontra a seção de vantagens (com mais tolerância)
        secao = re.search(
            r"(?:VANTAGENS|PROVENTOS).*?(?:cód|COD|DISCRIMINAÇÃO).*?\n(.*?)(?=TOTAL\s+DE\s+VANTAGENS|\n\n|$)",
            text, re.DOTALL | re.IGNORECASE
        )
        
        if not secao:
            return vantagens
            
        # Padrões para linhas da tabela
        padroes = [
            r"(\d{5})\s+([A-ZÀ-Ú./\s-]+?)\s+([\d.,]+)%?\s+([\d.,]+)",  # Com percentual
            r"(\d{5})\s+([A-ZÀ-Ú./\s-]+?)\s+([\d.,]+)\s*$"  # Sem percentual
        ]
        
        for linha in secao.group(1).split('\n'):
            linha = linha.strip()
            if not linha:
                continue
                
            for padrao in padroes:
                if match := re.match(padrao, linha, re.IGNORECASE):
                    cod = match.group(1)
                    if cod in self.vantagens_alvo:
                        vantagem = {
                            "codigo": cod,
                            "descricao": self.vantagens_alvo[cod]['descricao'],
                            "valor": self._parse_valor(match.group(-1))  # Último grupo é sempre o valor
                        }
                        
                        # Adiciona percentual se aplicável
                        if self.vantagens_alvo[cod]['tem_percentual'] and len(match.groups()) >= 3:
                            vantagem["percentual_duracao"] = self._parse_valor(match.group(3))
                            
                        vantagens.append(vantagem)
                    break
                    
        return vantagens

    def _parse_valor(self, valor_str: str) -> float:
        """Converte valores brasileiros para float"""
        try:
            return float(valor_str.replace('.', '').replace(',', '.'))
        except:
            return 0.0  # Retorna 0 se não conseguir converter

@app.post("/processar")
async def processar_contracheque(file: UploadFile = File(...)):
    """Endpoint otimizado para processamento de contracheques"""
    temp_path = "temp_contracheque.pdf"
    try:
        # Salva o arquivo temporariamente
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        processor = ContrachequeProcessor()
        return processor.processar_documento(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)