from ultralytics import YOLO
import cv2
import easyocr

# Load your custom YOLO model
model = YOLO("best.pt")

# Load the image
imgPath = "images.jpeg"
img = cv2.imread(imgPath)

# Run YOLO detection
results = model(img)
result = results[0]

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Add other languages if needed

# Get class names
class_names = model.names

# Process each detection
for box in result.boxes:
    # Get coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # Crop the detected region
    roi = img[y1:y2, x1:x2]

    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Run OCR on grayscale region
    ocr_result = reader.readtext(gray_roi)

    # Draw the bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display OCR results
    for detection in ocr_result:
        text = detection[1]  # The detected text
        print(f"OCR Text: {text}")
        # You can also draw the text on the image
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 2)

# Show the final image
cv2.imshow("YOLO + EasyOCR", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
