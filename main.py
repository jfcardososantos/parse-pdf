from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import re
from typing import List, Dict, Optional
import os

app = FastAPI()

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao extrair texto do PDF: {str(e)}")

# Função para processar um contracheque individual
def parse_single_paycheck(text: str) -> Dict:
    try:
        # Extrair nome e matrícula
        nome_match = re.search(r"\d+\s+([A-Z\s]+)\s+GP:", text)
        matricula_match = re.search(r"(\d+)\s+[A-Z\s]+GP:", text)
        mes_ano_match = re.search(r"(JANEIRO|FEVEREIRO|MARÇO|ABRIL|MAIO|JUNHO|JULHO|AGOSTO|SETEMBRO|OUTUBRO|NOVEMBRO|DEZEMBRO)/\d{4}", text, re.IGNORECASE)

        nome = nome_match.group(1).strip() if nome_match else "NÃO ENCONTRADO"
        matricula = matricula_match.group(1).strip() if matricula_match else "NÃO ENCONTRADA"
        mes_ano = mes_ano_match.group(0).strip() if mes_ano_match else "NÃO ENCONTRADO"

        # Extrair vantagens
        vantagens = []
        in_vantagens = False
        for line in text.split("\n"):
            if "CNS/VDS" in line and "CONSIG/VANT/DESC" in line:
                in_vantagens = True
                continue
            if "******** TOTAL VENTAGENS" in line:
                in_vantagens = False

            if in_vantagens and line.strip() and "/" in line.split()[0]:
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 4:
                    codigo = parts[0].replace("/", "").strip()
                    descricao = parts[1].strip()
                    percentual = float(parts[2].replace(",", ".")) if parts[2].strip() else None
                    valor = float(parts[3].replace(".", "").replace(",", "."))
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

# Rota da API para processar PDF
@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        # Salvar o PDF temporariamente
        temp_pdf_path = "temp_contracheque.pdf"
        with open(temp_pdf_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extrair texto e dividir por contracheque (se houver múltiplos)
        text = extract_text_from_pdf(temp_pdf_path)
        contracheques = []
        
        # Separar seções de contracheque (ex: "GOVERNO DO ESTADO DA BAHIA")
        sections = re.split(r"(?=GOVERNO DO ESTADO DA BAHIA)", text)
        for section in sections:
            if section.strip():
                contracheque = parse_single_paycheck(section)
                if contracheque:
                    contracheques.append(contracheque)

        # Limpar arquivo temporário
        os.remove(temp_pdf_path)

        return {"contracheques": contracheques}
    except Exception as e:
        if os.path.exists("temp_contracheque.pdf"):
            os.remove("temp_contracheque.pdf")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)