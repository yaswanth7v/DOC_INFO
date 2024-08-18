import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

import torch
from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

from PIL import Image
import easyocr
import fitz  # PyMuPDF
import io
import pytesseract
import os
import time

load_dotenv()

forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def size_normalization(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

def binarization(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def smoothing(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_eroded = cv2.erode(image, kernel, iterations=1)
    image_dilated = cv2.dilate(image_eroded, kernel, iterations=1)
    return image_dilated

def opening_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image_opened

def closing_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image_closed

def edge_detection(image, kernel_size):
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, np.ones((kernel_size, kernel_size), np.uint8))
    return gradient

def preprocess_image(image, size, threshold, kernel_size, zoom):
    image = np.array(image)
    height, width = image.shape[:2]
    new_size = (int(width * zoom), int(height * zoom))
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    
    normalized_image = size_normalization(image, size)
    binary_image = binarization(normalized_image, threshold)
    smoothed_image = smoothing(binary_image, kernel_size)
    opened_image = opening_filter(smoothed_image, kernel_size)
    closed_image = closing_filter(opened_image, kernel_size)
    edged_image = edge_detection(closed_image, kernel_size)
    
    return closed_image

def extract_data(text):
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=1000, temperature=0.1, token=HF_TOKEN)

    #instruction = "Extract specific data: Roll No, first name, middle name, last name, dob, age , Adhar no, mobile no, address, pin code, city, pan no, state. English only and json format."
    instruction = f"Extract all entities from the text = {text}. English only and json format."
    
    question = f"{instruction}, ocr_text={text}"
    template = "Question: {question}\nAnswer: "
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = prompt | llm

    result = llm_chain.invoke(question)
    return result

def extract_text_from_blocks(blocks):
    text = []
    for block in blocks:
        for line in block['lines']:
            for word in line['words']:
                text.append(word['value'])
    return " ".join(text)

def perform_doctr(det_archs, reco_archs):
    """Build a streamlit layout"""
    st.title("docTR: Document Text Recognition")
    st.write("\n")
    st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")

    cols = st.columns((1, 1, 1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    cols[2].subheader("OCR output")
    cols[3].subheader("Page reconstitution")

    st.sidebar.title("Document selection")
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[page_idx]
        cols[0].image(page)

    st.sidebar.title("Model selection")
    st.sidebar.markdown("**Backend**: PyTorch")
    det_arch = st.sidebar.selectbox("Text detection model", det_archs)
    reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)

    st.sidebar.write("\n")
    assume_straight_pages = st.sidebar.checkbox("Assume straight pages", value=True)
    st.sidebar.write("\n")
    straighten_pages = st.sidebar.checkbox("Straighten pages", value=False)
    st.sidebar.write("\n")
    bin_thresh = st.sidebar.slider("Binarization threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    st.sidebar.write("\n")
    box_thresh = st.sidebar.slider("Box threshold", min_value=0.1, max_value=0.9, value=0.1, step=0.1)

    if st.sidebar.button("Analyze page"):
        if uploaded_file is None:
            st.sidebar.write("Please upload a document")
        else:
            with st.spinner("Loading model..."):
                predictor = load_predictor(
                    det_arch, reco_arch, assume_straight_pages, straighten_pages, bin_thresh, box_thresh, forward_device
                )

            with st.spinner("Analyzing..."):
                seg_map = forward_image(predictor, page, forward_device)
                seg_map = np.squeeze(seg_map)
                seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis("off")
                cols[1].pyplot(fig)

                out = predictor([page])
                fig = visualize_page(out.pages[0].export(), out.pages[0].page, interactive=False, add_labels=False)
                cols[2].pyplot(fig)

                page_export = out.pages[0].export()
                if assume_straight_pages or (not assume_straight_pages and straighten_pages):
                    img = out.pages[0].synthesize()
                    cols[3].image(img, clamp=True)

                st.markdown("\nHere are your analysis results in JSON format:")
                st.json(page_export, expanded=False)

                extracted_text = extract_text_from_blocks(page_export['blocks'])
                st.markdown("\nHere is the extracted text:")
                result = extract_data(extracted_text)
                st.write(result)

def pdf_to_images(pdf_file, zoom):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        image_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(image_data))
        images.append(image)
        st.image(image, caption=f'Page {page_num + 1}', use_column_width=True)
    return images

def pdf_to_text(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def perform_ocr_on_image(image, ocr_tool):
    if ocr_tool == 'EasyOCR':
        reader = easyocr.Reader(['en'], gpu=False)
        bounds = reader.readtext(np.array(image))
        ocr_text = " ".join([bound[1] for bound in bounds])
    elif ocr_tool == 'Tesseract OCR':
        ocr_text = pytesseract.image_to_string(image)
    return ocr_text

def perform_ocr_on_images(images, ocr_tool):
    text = ""
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(perform_ocr_on_image, image, ocr_tool) for image in images]
        for future in futures:
            text += future.result() + "\n"
    return text

def perform_ocr_on_pdf(uploaded_file, ocr_tool, zoom):
    images = pdf_to_images(uploaded_file, zoom)
    ocr_text = perform_ocr_on_images(images, ocr_tool)
    return ocr_text

def main():
    st.title("Multi-tool Document Processing and Data Extraction")
    #st.write("Upload an image file (.jpg, .png) or a PDF")

    ocr_tool = st.selectbox("Choose OCR Tool", ["EasyOCR", "Tesseract OCR", "doctr"])
    
    if ocr_tool == "doctr":
        perform_doctr(DET_ARCHS, RECO_ARCHS)
    
    else:
        extraction_method = st.selectbox("Choose Extraction Method", ["OCR", "PDF Loader"])

        uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg", "pdf"])
        
        method = st.selectbox("Choose Preprocessing method", ["Original", "Preprocess Image"])

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension.lower() in ['jpg', 'jpeg', 'png']:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                if method == "Preprocess Image":
                    width = st.slider('Select width for normalization', 100, 2000, 1024, step=50)
                    height = st.slider('Select height for normalization', 100, 2000, 1024, step=50)
                    threshold = st.slider('Select threshold for binarization', 0, 255, 127)
                    kernel_size = st.slider('Select kernel size for morphological operations', 1, 20, 5)
                    zoom = st.slider('Select zoom factor', 1.0, 3.0, 1.0, step=0.1)

                    preprocessed_image = preprocess_image(image, (width, height), threshold, kernel_size, zoom)
                    st.header("Preprocessed Image")
                    st.image(preprocessed_image, caption='Final Preprocessed Image', use_column_width=True)

                else:
                    preprocessed_image = image
            
                if st.button('Perform OCR'):
                    start = time.time()
                    ocr_text = perform_ocr_on_image(preprocessed_image, ocr_tool)
                    end = time.time()
                    ocr_time = end - start

                    st.header("OCR Output")
                    st.write(ocr_text)
                    st.write(f"OCR Time: {ocr_time:.2f} seconds")

                    start = time.time()
                    data = extract_data(ocr_text)
                    end = time.time()
                    llm_time = end - start

                    st.header("Extracted Data")
                    st.write(data)
                    st.write(f"LLM Processing Time: {llm_time:.2f} seconds")

            elif file_extension.lower() == 'pdf':
                if extraction_method == "OCR":
                    zoom = st.slider('Select zoom factor for PDF', 1.0, 3.0, 1.0, step=0.1)
                    
                    if st.button('Perform OCR on PDF'):
                        start = time.time()
                        ocr_text = perform_ocr_on_pdf(uploaded_file, ocr_tool, zoom)
                        end = time.time()
                        ocr_time = end - start

                        st.header("OCR Output from PDF")
                        st.write(ocr_text)
                        st.write(f"OCR Time: {ocr_time:.2f} seconds")

                        start = time.time()
                        data = extract_data(ocr_text)
                        end = time.time()
                        llm_time = end - start

                        st.header("Extracted Data")
                        st.write(data)
                        st.write(f"LLM Processing Time: {llm_time:.2f} seconds")
                        
                elif extraction_method == "PDF Loader":
                    if st.button('Extract Text from PDF'):
                        start = time.time()
                        pdf_text = pdf_to_text(uploaded_file)
                        end = time.time()
                        extract_time = end - start

                        st.header("Text Extracted from PDF")
                        st.write(pdf_text)
                        st.write(f"Extraction Time: {extract_time:.2f} seconds")

                        start = time.time()
                        data = extract_data(pdf_text)
                        end = time.time()
                        llm_time = end - start

                        st.header("Extracted Data")
                        st.write(data)
                        st.write(f"LLM Processing Time: {llm_time:.2f} seconds")

if __name__ == '__main__':
    main()
