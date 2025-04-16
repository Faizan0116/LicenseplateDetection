from ultralytics import YOLO
import cv2
import easyocr

import mysql.connector
from mysql.connector import Error

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="rootadmin",
        database="TechPassDB"
    )

    if conn.is_connected():
        print("✅ Connected to MySQL database!")

        # Optional: get server info
        db_info = conn.get_server_info()
        print("MySQL Server version:", db_info)

        # Optional: get database name
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE();")
        record = cursor.fetchone()
        print("You're connected to:", record[0])

except Error as e:
    print("❌ Error while connecting to MySQL:", e)

finally:
    

    # Load your custom YOLO model
    model = YOLO("best.pt")

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Add other languages if needed

    # Load the video (replace 'video.mp4' with your video file path or use 0 for webcam)
    video_path = "VID_20250416_191128.mp4"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    # Check if the video file or webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Process each frame
    while True:
        ret, frame = cap.read()
        # if not ret:
        #     break  # Exit loop if no more frames

        # Run YOLO detection
        results = model(frame)
        result = results[0]

        # Get class names
        class_names = model.names

        # Process each detection
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Crop the detected region
            roi = frame[y1:y2, x1:x2]

            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Run OCR on grayscale region
            ocr_result = reader.readtext(gray_roi)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display OCR results
            for detection in ocr_result:
                text = detection[1]  # The detected text
                print(f"OCR Text: {text}")
                # Draw the text on the frame
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 255), 2)

        # Display the frame with detections
        cv2.imshow("YOLO + EasyOCR", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        print("🔌 MySQL connection closed.")