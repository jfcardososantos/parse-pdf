import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import re
import os
import logging
from pdf2image import convert_from_path
import pyopencl as cl
import pytesseract
from typing import Dict, List
from PIL import Image
import uvicorn

# Configuração ROCm para AMD GPU
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["GPU_MAX_HEAP_SIZE"] = "100"

app = FastAPI()


def get_amd_platform():
    """Seleciona automaticamente a plataforma AMD"""
    platforms = cl.get_platforms()
    for i, platform in enumerate(platforms):
        if 'AMD' in platform.name:
            return platform
    return platforms[0]  

class AMDProcessor:
    def __init__(self):
        try:
            # Seleção automática da plataforma AMD
            amd_platform = get_amd_platform()
            self.ctx = cl.Context(
                dev_type=cl.device_type.GPU,
                properties=[(cl.context_properties.PLATFORM, amd_platform)]
            )
            self.queue = cl.CommandQueue(self.ctx)
            self._build_kernels()
            self._init_ocr()
            logging.info(f"Plataforma selecionada: {amd_platform.name}")
        except Exception as e:
            logging.error(f"Erro na inicialização: {str(e)}")
            raise

    def _build_kernels(self):
        """Compila kernels OpenCL"""
        self.kernels = {}
        try:
            # Kernel de limiarização adaptativa
            self.kernels['threshold'] = cl.Program(self.ctx, """
            __kernel void threshold(
                __global const uchar *input,
                __global uchar *output,
                int width,
                int height,
                int threshold_value)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int idx = y * width + x;
                
                if (x < width && y < height) {
                    output[idx] = input[idx] > threshold_value ? 255 : 0;
                }
            }
            """).build()

            # Kernel de denoising
            self.kernels['denoise'] = cl.Program(self.ctx, """
            __kernel void denoise(
                __global const uchar *input,
                __global uchar *output,
                int width,
                int height)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);
                
                if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
                    int sum = 0;
                    for (int j = -1; j <= 1; j++) {
                        for (int i = -1; i <= 1; i++) {
                            sum += input[(y+j) * width + (x+i)];
                        }
                    }
                    output[y * width + x] = sum / 9;
                }
            }
            """).build()

        except Exception as e:
            logging.error(f"Erro ao compilar kernels: {str(e)}")
            raise

    def _init_ocr(self):
        """Configura Tesseract OCR"""
        self.tessconfig = r'--oem 3 --psm 6 -l por'
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    def _gpu_preprocess(self, image: Image) -> Image:
        """Pré-processamento na GPU"""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            height, width = img_array.shape
            mf = cl.mem_flags

            # Buffer de entrada
            input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_array)
            
            # Buffer de saída
            output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, img_array.nbytes)

            # Aplica denoising
            self.kernels['denoise'].denoise(
                self.queue, (width, height), None,
                input_buf, output_buf,
                np.int32(width), np.int32(height)
            )
            
            # Copia resultado
            denoised = np.empty_like(img_array)
            cl.enqueue_copy(self.queue, denoised, output_buf)

            # Aplica threshold
            self.kernels['threshold'].threshold(
                self.queue, (width, height), None,
                input_buf, output_buf,
                np.int32(width), np.int32(height), np.int32(128)
            )

            final = np.empty_like(img_array)
            cl.enqueue_copy(self.queue, final, output_buf)

            return Image.fromarray(final)

        except Exception as e:
            logging.warning(f"Fallback para CPU: {str(e)}")
            return self._cpu_preprocess(image)

    def _cpu_preprocess(self, image: Image) -> Image:
        """Fallback para pré-processamento em CPU"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(threshold)

    def extract_text(self, pdf_path: str) -> str:
        """Extrai texto do PDF"""
        try:
            images = convert_from_path(
                pdf_path,
                dpi=400,
                grayscale=True,
                thread_count=4
            )

            full_text = ""
            for img in images:
                processed = self._gpu_preprocess(img)
                text = pytesseract.image_to_string(processed, config=self.tessconfig)
                full_text += self._correct_text(text) + "\n"

            return full_text

        except Exception as e:
            logging.error(f"Erro na extração: {str(e)}")
            raise HTTPException(status_code=400, detail="Falha ao processar PDF")

    def _correct_text(self, text: str) -> str:
        """Correções de texto pós-OCR"""
        corrections = {
            r'GOVERNO\s+D[EO]\s+ESTAD[OA]': 'GOVERNO DO ESTADO',
            r'MATR[ÍI]CULA': 'MATRÍCULA',
            r'(\d{2})[/](\d{4})': r'\1/\2',
            r'Contra[çc]heque': 'Contracheque'
        }
        for pattern, repl in corrections.items():
            text = re.sub(pattern, repl, text)
        return text

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """Endpoint principal"""
    temp_path = "temp_pdf.pdf"
    try:
        # Salva arquivo temporário
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Processamento
        processor = AMDProcessor()
        text = processor.extract_text(temp_path)
        
        # Exemplo de parser - implemente sua lógica específica aqui
        result = {
            "text": text[:1000] + ("..." if len(text) > 1000 else ""),
            "pages": text.count('\n\n') + 1
        }
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)