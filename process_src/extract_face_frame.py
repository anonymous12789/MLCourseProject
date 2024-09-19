import cv2
import mediapipe as mp
import os
import concurrent.futures
import glob

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def extract_faces_from_video(video_path, output_dir, max_faces=40):
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Initialize face detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        face_count = 0
        frame_count = 0

        # Process the video frame by frame
        while cap.isOpened() and face_count < max_faces:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform face detection
            results = face_detection.process(rgb_frame)

            # If faces are detected, extract them
            if results.detections:
                for detection in results.detections:
                    # Get bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Crop the face from the frame
                    cropped_face = frame[y:y+h, x:x+w]

                    # Create unique file name for each face
                    face_file_name = f"{os.path.basename(video_path)}_face_{frame_count}_{face_count}.jpg"
                    face_file_path = os.path.join(output_dir, face_file_name)
                    
                    # Save the cropped face image
                    cv2.imwrite(face_file_path, cropped_face)

                    face_count += 1

                    if face_count >= max_faces:
                        break
            else:
                cv2.imwrite(f"{output_dir}/no_face_{frame_count}.jpg", frame)

            frame_count += 1

    # Release video capture
    cap.release()
    print(f"Extracted {face_count} faces from {video_path}.")


def process_videos_in_folder(folder_path, output_base_dir, max_faces=41, max_threads=4):
    # Get list of all video files in the folder
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))  # Adjust extension if needed (e.g., ".avi", ".mov")

    # Create output directories for each video
    os.makedirs(output_base_dir, exist_ok=True)

    # Multithreading setup
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Create tasks for each video
        futures = []
        for video_path in video_files:
            video_output_dir = output_base_dir
            os.makedirs(video_output_dir, exist_ok=True)
            futures.append(executor.submit(extract_faces_from_video, video_path, video_output_dir, max_faces))

        # Wait for all threads to finish
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # To catch any exception raised during execution
            except Exception as e:
                print(f"Error occurred: {e}")

    print("Processing completed for all videos.")

if __name__ == "__main__":
    folder_path = 'data_train/fake/'  # Folder containing videos
    output_base_dir = 'data_processed/fake/'   # Folder to store cropped faces
    process_videos_in_folder(folder_path, output_base_dir, max_faces=40, max_threads=16)
    folder_path = 'data_train/real/'  # Folder containing videos
    output_base_dir = 'data_processed/real/'   # Folder to store cropped faces
    process_videos_in_folder(folder_path, output_base_dir, max_faces=40, max_threads=16)
