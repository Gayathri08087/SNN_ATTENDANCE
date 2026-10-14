import cv2
import os
from config import STREAM_URL


def main():
    # ---------------------------------------------------------
    # 1. Ask for Student ID
    # ---------------------------------------------------------
    student_id = input("Enter Student ID : ").strip()

    if student_id == "":
        print("Student ID cannot be empty. Exiting.")
        return

    # ---------------------------------------------------------
    # 2. Create Dataset/<StudentID>/ folder if it doesn't exist
    # ---------------------------------------------------------
    dataset_dir = os.path.join("..", "Dataset", student_id)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # ---------------------------------------------------------
    # 3. Load HaarCascade
    # ---------------------------------------------------------
    cascade_path = os.path.join("..", "HaarCascade", "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Could not load HaarCascade file at " + cascade_path)
        return

    # ---------------------------------------------------------
    # 4. Connect to ESP32-CAM stream
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Error: Could not connect to stream at " + STREAM_URL)
        return

    saved_count = 0
    max_images = 5

    print("Live feed started. Press 'S' to capture a face, 'Q' to quit.")

    while saved_count < max_images:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Warning: Failed to read frame from stream.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        display_frame = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status_text = "Saved : " + str(saved_count) + "/" + str(max_images)
        cv2.putText(
            display_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("ESP32-CAM Enrollment", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # ---------------------------------------------------------
        # Q = Quit at any time
        # ---------------------------------------------------------
        if key == ord('q') or key == ord('Q'):
            print("Enrollment cancelled by user.")
            break

        # ---------------------------------------------------------
        # S = Capture ONE face
        # ---------------------------------------------------------
        elif key == ord('s') or key == ord('S'):

            if len(faces) == 0:
                print("No face detected. Try again.")
                continue

            # Use the first detected face
            (x, y, w, h) = faces[0]
            cropped_face = frame[y:y + h, x:x + w]

            # ---------------------------------------------------------
            # Show cropped face preview and wait for Y/N/Q
            # ---------------------------------------------------------
            decision_made = False

            while not decision_made:

                cv2.imshow("Captured Face", cropped_face)
                key2 = cv2.waitKey(0) & 0xFF

                if key2 == ord('y') or key2 == ord('Y'):
                    saved_count += 1
                    filename = student_id + "_" + str(saved_count) + ".jpg"
                    filepath = os.path.join(dataset_dir, filename)
                    cv2.imwrite(filepath, cropped_face)
                    print("Saved : " + filepath)

                    cv2.destroyWindow("Captured Face")
                    decision_made = True

                elif key2 == ord('n') or key2 == ord('N'):
                    print("Retaking image...")
                    cv2.destroyWindow("Captured Face")
                    decision_made = True

                elif key2 == ord('q') or key2 == ord('Q'):
                    print("Enrollment cancelled by user.")
                    cv2.destroyWindow("Captured Face")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        # ---------------------------------------------------------
        # Check completion
        # ---------------------------------------------------------
        if saved_count == max_images:
            print("Enrollment Completed Successfully")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()