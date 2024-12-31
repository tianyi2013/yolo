import cv2
import numpy as np
import os
import logging
from pdf_generator import PDFGenerator
import urllib.request
from PIL import Image

def setup_logger():
    logger = logging.getLogger("yolo_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("yolo.log")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def download_file(url, filename, logger):
    if not os.path.exists(filename):
        logger.info(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Downloaded {filename}")

def load_yolo_model(logger):
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    config_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"
    
    download_file(weights_url, weights_path, logger)
    download_file(config_url, config_path, logger)
    download_file(names_url, names_path, logger)
    
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers

def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, width, height

def convert_to_jpeg(img_path, output_path):
    if not img_path.lower().endswith(".jpg") and not img_path.lower().endswith(".jpeg"):
        with Image.open(img_path) as im:
            im.convert("RGB").save(output_path, "JPEG")
    else:
        output_path = img_path
    return output_path

def draw_labels(outs, width, height, img, classes):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    logger = setup_logger()
    try:
        net, classes, output_layers = load_yolo_model(logger)
        input_dir = "./resources/images"
        output_dir = "./resources/output"
        os.makedirs(output_dir, exist_ok=True)

        for img_filename in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_filename)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            outs, width, height = detect_objects(img, net, output_layers)
            img = draw_labels(outs, width, height, img, classes)

            # Create a separate folder for each image's output
            img_output_dir = os.path.join(output_dir, os.path.splitext(img_filename)[0])
            os.makedirs(img_output_dir, exist_ok=True)

            # Save the annotated image
            annotated_img_path = os.path.join(img_output_dir, f"annotated_{img_filename}")
            cv2.imwrite(annotated_img_path, img)            

            # Convert the original image to jpeg if necessary
            original_img_jpeg_path = convert_to_jpeg(img_path, os.path.join(img_output_dir, f"original_{os.path.splitext(img_filename)[0]}.jpg"))

            # Convert the annotated image to jpeg if necessary
            annotated_img_jpeg_path = convert_to_jpeg(annotated_img_path, os.path.join(img_output_dir, f"annotated_{os.path.splitext(img_filename)[0]}.jpg"))

            # Create a PDF with the original and annotated images side by side
            pdf = PDFGenerator()
            pdf.add_page()
            pdf.chapter_title("Original Image")
            pdf.chapter_body(original_img_jpeg_path)
            pdf.chapter_title("Annotated Image")
            pdf.chapter_body(annotated_img_path)

            pdf_output_path = os.path.join(img_output_dir, f"output_{os.path.splitext(img_filename)[0]}.pdf")
            pdf.output(pdf_output_path)

            logger.info(f"Processed {img_filename}, PDF saved to {pdf_output_path}")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
