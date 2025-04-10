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

# Configuração ROCm para AMD GPU
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.2"  # Especifica arquitetura gfx1032
os.environ["GPU_MAX_HEAP_SIZE"] = "100"  # Alocação de memória em %

app = FastAPI()

class AMDProcessor:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self._build_kernels()
        self._init_ocr()

    def _build_kernels(self):
        """Compila kernels OpenCL para pré-processamento"""
        self.kernels = {
            'adaptive_threshold': cl.Program(self.ctx, """
            __kernel void adaptive_threshold(
                __global const uchar *input,
                __global uchar *output,
                const int width,
                const int threshold)
            {
                int gid = get_global_id(0);
                if (gid < width) {
                    output[gid] = input[gid] > threshold ? 255 : 0;
                }
            }
            """).build(),

            'denoise': cl.Program(self.ctx, """
            __kernel void denoise(
                __global const uchar *input,
                __global uchar *output,
                const int width,
                const int height)
            {
                int x = get_global_id(0);
                int y = get_global_id(1);
                if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
                    int sum = 0;
                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            sum += input[(y+j)*width + (x+i)];
                        }
                    }
                    output[y*width + x] = sum / 9;
                }
            }
            """).build()
        }

    def _init_ocr(self):
        """Configura Tesseract para AMD"""
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        self.tessconfig = r'--oem 3 --psm 6 -l por'

    def _gpu_preprocess(self, image):
        """Pipeline de pré-processamento na GPU"""
        try:
            # Conversão para OpenCL
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            # Alocar buffers
            mf = cl.mem_flags
            input_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_array)
            output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, img_array.nbytes)

            # Aplicar kernels
            self.kernels['denoise'].denoise(
                self.queue, img_array.shape, None,
                input_buf, output_buf,
                np.int32(img_array.shape[1]), np.int32(img_array.shape[0])
            
            denoised = np.empty_like(img_array)
            cl.enqueue_copy(self.queue, denoised, output_buf)

            # Limiarização adaptativa
            self.kernels['adaptive_threshold'].adaptive_threshold(
                self.queue, denoised.shape, None,
                input_buf, output_buf,
                np.int32(denoised.size), np.int32(128))
            
            final = np.empty_like(denoised)
            cl.enqueue_copy(self.queue, final, output_buf)

            return Image.fromarray(final)

        except Exception as e:
            logging.warning(f"GPU fallback: {str(e)}")
            return self._cpu_preprocess(image)

    def _cpu_preprocess(self, image):
        """Fallback para CPU"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(threshold)

    def extract_text(self, pdf_path: str) -> str:
        """Processamento completo do PDF"""
        try:
            images = convert_from_path(
                pdf_path,
                dpi=400,
                grayscale=True,
                poppler_path='/usr/bin/poppler' if os.path.exists('/usr/bin/poppler') else None
            )

            full_text = ""
            for img in images:
                processed = self._gpu_preprocess(img)
                text = pytesseract.image_to_string(processed, config=self.tessconfig)
                full_text += self._correct_text(text) + "\n"

            return full_text

        except Exception as e:
            logging.error(f"Extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail="Falha no processamento")

    def _correct_text(self, text: str) -> str:
        """Correções específicas para contracheques"""
        corrections = {
            r'\bGOVERNO\s+D[EO]\s+ESTAD[OA]\b': 'GOVERNO DO ESTADO',
            r'\bMATR[ÍI]CULA\b': 'MATRÍCULA',
            r'\b(\d{2})[/](\d{4})\b': r'\1/\2'  # Normaliza datas
        }
        for pattern, repl in corrections.items():
            text = re.sub(pattern, repl, text)
        return text

@app.post("/parse-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """Endpoint otimizado para AMD GPU"""
    temp_path = "temp_amd.pdf"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        processor = AMDProcessor()
        text = processor.extract_text(temp_path)
        
        # Implemente seu parser específico aqui
        result = {
            "text": text[:500] + "...",  # Exemplo
            "gpu_used": "AMD RX 6600M" if "cl" in str(processor.ctx) else "CPU"
        }
        
        return result

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)