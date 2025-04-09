from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import re
from typing import List, Dict, Optional
import os

app = FastAPI()

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao extrair texto do PDF: {str(e)}")

def parse_paycheck(text: str) -> Dict:
    try:
        # Padrões atualizados para extração
        nome_match = re.search(r"(\d{9})\s([A-ZÀ-Ú\s]+)\s+GP:", text)
        mes_ano_match = re.search(r"(JANEIRO|FEVEREIRO|MAR[ÇC]O|ABRIL|MAIO|JUNHO|JULHO|AGOSTO|SETEMBRO|OUTUBRO|NOVEMBRO|DEZEMBRO)\/\d{4}", text, re.IGNORECASE)
        
        nome = nome_match.group(2).strip() if nome_match else "NÃO ENCONTRADO"
        matricula = nome_match.group(1).strip() if nome_match else "NÃO ENCONTRADA" 
        mes_ano = mes_ano_match.group(0).strip() if mes_ano_match else "NÃO ENCONTRADO"

        # Extrair vantagens - padrão atualizado
        vantagens = []
        lines = text.split('\n')
        start_processing = False
        
        for line in lines:
            if "CNS/VDS" in line and "CONSIG/VANT/DESC" in line:
                start_processing = True
                continue
            if "******** TOTAL VENTAGENS" in line:
                start_processing = False
            
            if start_processing and line.strip() and '/' in line[:10]:
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 4:
                    codigo = parts[0].replace('/', '').strip()
                    descricao = parts[1].strip()
                    percentual = float(parts[2].replace(',', '.')) if parts[2].strip() and parts[2].strip() != '-' else None
                    valor = float(parts[3].replace('.', '').replace(',', '.'))
                    
                    vantagens.append({
                        "codigo": codigo,
                        "descricao": descricao,
                        "percentual_duracao": percentual,
                        "valor": valor
                    })

        return {
            "nome_completo": nome,
            "matricula": matricula,
            "mes_ano_referencia": mes_ano,
            "vantagens": vantagens
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar contracheque: {str(e)}")

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        temp_pdf_path = "temp_contracheque.pdf"
        with open(temp_pdf_path, "wb") as buffer:
            buffer.write(await file.read())

        text = extract_text_from_pdf(temp_pdf_path)
        
        # Processar múltiplos contracheques
        sections = re.split(r'(?=GOVERNO DO ESTADO DA BAHIA)', text)
        contracheques = []
        
        for section in sections:
            if section.strip():
                result = parse_paycheck(section)
                if result['vantagens']:  # Só adiciona se encontrar vantagens
                    contracheques.append(result)

        os.remove(temp_pdf_path)
        return {"contracheques": contracheques}
    except Exception as e:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)