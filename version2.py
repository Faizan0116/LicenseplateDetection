import cv2
from ultralytics import YOLO
import pytesseract
import pyodbc
import re
import cvzone  # Import cvzone library

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path if needed

# Define the connection parameters
server = r'DESKTOP-6D1H4NL\SQLEXPRESS'  # Replace with your server name
database = r'NumberPlate'  # Replace with your database name
username = r'sa'  # Replace with your username
password = r'img123'  # Replace with your password

# Create the connection string
connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"

try:
    # Establish the connection
    conn = pyodbc.connect(connection_string)
    print("Connection to SQL Server database successful!")

    # Create a cursor object
    cursor = conn.cursor()

except pyodbc.Error as e:
    print("Error connecting to SQL Server:", e)
    exit()

# Path to the video file
video_path = "sample2.mp4"  # Replace with the path to your video file

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
output_path = "annotated_output.mp4"  # Path to save the output video
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the FPS of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the input video
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the input video

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

model = YOLO("best.pt")  # Load your custom YOLO model

# Read and display the video frames
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame.")
        break

    # Run YOLO detection on the video frame
    results = model.predict(source=frame, conf=0.45, stream=True, save=False, show=False)

    # Draw the detection results on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract the detected area (ROI)
            roi = frame[y1+3:y2-8, x1+15:x2-15]

            # Display the extracted area in a separate window
            if roi.size > 0:  # Ensure the ROI is valid
                magnified_roi = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                magnified_roi_gray = cv2.cvtColor(magnified_roi, cv2.COLOR_BGR2GRAY)

                # Pass the filtered ROI to PyTesseract
                ocr_text = pytesseract.image_to_string(magnified_roi, config='--psm 6')

                # Remove all non-alphanumeric characters from the detected text
                cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', ocr_text).upper()
                print(f"Detected Text: {cleaned_text}")

                # Query the database to check if the cleaned text matches NumPlate
                cursor.execute("SELECT NumPlate, CPLC FROM PlateDetection WHERE NumPlate = ?", cleaned_text)
                result = cursor.fetchone()

                if result:
                    # If a match is found, check the CPLC value
                    numplate, cplc = result
                    if cplc != 1:  # If CPLC is not 1, display "THEFT"
                        cvzone.putTextRect(frame, "Status: THEFT", (x1, y1 - 30), scale=1, thickness=2,colorT=(255,255,255), colorR=(255, 0, 0), offset=10)
                    else:  # If CPLC is 1, display "NORMAL"
                        cvzone.putTextRect(frame, "Status: NORMAL", (x1, y1 - 30), scale=1, thickness=2,colorT=(0,0,0), colorR=(0, 255, 0), offset=10)
                else:
                    # If no match is found, display "NOT FOUND"
                    cvzone.putTextRect(frame, "No Record Found", (x1, y1 - 30), scale=1, thickness=2,colorT=(0,0,0), colorR=(255, 255, 255), offset=10)

    # Display the frame
    cv2.imshow("Video", frame)

    # Write the annotated frame to the output video
    out.write(frame)

    # Exit the video display when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Release the VideoWriter object
out.release()

# Close the database connection
cursor.close()
conn.close()