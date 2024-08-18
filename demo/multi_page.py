import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import os
import logging
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
import json

load_dotenv()

forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image preprocessing functions
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

def download_json(data):
    json_str = json.dumps(data, indent=4)
    return json_str

def display_data_dict(data_dict):
    for key in list(data_dict.keys())[::2]:
        col1, col2 = st.columns(2)
        
        with col1:
            if isinstance(data_dict[key], (dict, list)):
                with st.expander(f"**{key}**"):
                    display_nested_data(data_dict[key])
            else:
                with st.expander(f"**{key}**"):
                    st.write(data_dict[key])
        
        if key != list(data_dict.keys())[-1]:
            next_key = list(data_dict.keys())[list(data_dict.keys()).index(key) + 1]
            with col2:
                if isinstance(data_dict[next_key], (dict, list)):
                    with st.expander(f"**{next_key}**"):
                        display_nested_data(data_dict[next_key])
                else:
                    with st.expander(f"**{next_key}**"):
                        st.write(data_dict[next_key])

def display_nested_data(data):
    if isinstance(data, dict):
        for sub_key, sub_value in data.items():
            if isinstance(sub_value, (dict, list)):
                with st.expander(f"**{sub_key}**"):
                    display_nested_data(sub_value)
            else:
                st.write(f"{sub_key}: {sub_value}")
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, (dict, list)):
                with st.expander(f"**Item {index + 1}**"):
                    display_nested_data(item)
            else:
                st.write(f"Item {index + 1}: {item}")

def extract_data(text):
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=1000, temperature=0.1, token=HF_TOKEN)

    st.write(text)
    instruction = f"extract all data from the text = {text} as json object without any note."
    question = f"{instruction}, ocr_text={text}"
    template = "Question: {question}\nAnswer: "
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = prompt | llm
    llm_output = llm_chain.invoke(question)

    # Print the raw output for debugging
    st.write("Raw model output:", llm_output)

    try:
        start_index = llm_output.find('{')
        end_index = llm_output.rfind('}') + 1

        # Extract JSON string
        json_str = llm_output[start_index:end_index]

        if not json_str:
            st.error("No JSON data found in the model output.")
            return

        # Load the JSON data
        data_dict = json.loads(json_str)
        st.write(data_dict)

        display_data_dict(data_dict)

        st.write("### Download JSON")
        json_str = download_json(data_dict)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="entities.json",
            mime="application/json"
        )
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON: {e}")
        logging.error(f"Failed to decode JSON: {e}")
        st.write(f"Raw model output: {llm_output}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}")

def extract_text_from_blocks(blocks):
    text = []
    for block in blocks:
        for line in block['lines']:
            for word in line['words']:
                text.append(word['value'])
    return " ".join(text)

def perform_doctr(det_archs, reco_archs):
    st.title("docTR: Document Text Recognition")
    st.write("\n")
    st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")

    cols = st.columns((1, 1, 1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    cols[2].subheader("OCR output")
    cols[3].subheader("Page reconstitution")

    st.sidebar.title("Document selection")
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["pdf"])
    
    if uploaded_file is not None:
        doc = DocumentFile.from_pdf(uploaded_file.read())
        st.sidebar.title("Page selection")
        page_indices = st.multiselect("Select pages", range(1, len(doc) + 1), default=list(range(1, len(doc) + 1)))
        
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
        st.sidebar.write("\n")
        zoom = st.sidebar.slider('Select zoom factor', 1.0, 12.0, 1.0, step=0.1)
        
        results = []
        
        for page_idx in page_indices:
            page = doc[page_idx - 1]
            cols[0].image(page)

            image = np.array(page)
            height, width = image.shape[:2]
            new_size = (int(width * zoom), int(height * zoom))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            page = image

            with st.spinner(f"Loading model for page {page_idx}..."):
                predictor = load_predictor(
                    det_arch, reco_arch, assume_straight_pages, straighten_pages, bin_thresh, box_thresh, forward_device
                )

            with st.spinner(f"Analyzing page {page_idx}..."):
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

                st.markdown(f"\nHere are the analysis results for page {page_idx} in JSON format:")
                st.json(page_export, expanded=False)

                extracted_text = extract_text_from_blocks(page_export['blocks'])
                st.markdown(f"\nHere is the extracted text for page {page_idx}:")
                extract_data(extracted_text)
                
                results.append({
                    'page': page_idx,
                    'text': extracted_text,
                    'json': page_export
                })
        
        if results:
            st.write("### Summary of Results")
            for result in results:
                st.write(f"**Page {result['page']}**")
                st.json(result['json'], expanded=False)
                st.write(result['text'])
                st.write("\n")

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
    text = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text.append(page.get_text())
    return "\n".join(text)

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
                    extract_data(ocr_text)
                    end = time.time()
                    llm_time = end - start

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
                        extract_data(ocr_text)
                        end = time.time()
                        llm_time = end - start

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
                        extract_data(pdf_text)
                        end = time.time()
                        llm_time = end - start

                        st.write(f"LLM Processing Time: {llm_time:.2f} seconds")

if __name__ == '__main__':
    main()
