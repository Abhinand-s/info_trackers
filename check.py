import cv2

# Load the pre-trained classifier for car detection
cascade_path = "path/to/haarcascade_car.xml"
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# List of known objects (you can customize this list)
known_objects = ["car", "bus", "truck"]

# Open a video capture object (you can replace 0 with the path to a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # List to store recognized objects
    recognized_objects = []

    # Draw rectangles around the detected cars and add the recognized objects to the list
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        recognized_objects.append("car")

    # Compare the recognized objects with the known objects
    for obj in recognized_objects:
        if obj in known_objects:
            print(f"Detected: {obj}")

    # Display the resulting frame
    cv2.imshow('Object Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

