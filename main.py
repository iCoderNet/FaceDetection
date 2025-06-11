import os
import cv2
import face_recognition
import numpy as np
import time

class FaceRecognizer:
    def __init__(self, images_folder="rasmlar/"):
        self.images_folder = images_folder
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Variables for FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
    def load_known_faces(self):
        """Load known faces from the images folder"""
        print("Loading known faces...")
        
        for filename in os.listdir(self.images_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.images_folder, filename)
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    # Get face locations first
                    face_locations = face_recognition.face_locations(image)
                    
                    if len(face_locations) > 0:
                        # Get encodings using face locations
                        face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        print(f"Loaded {name}'s face")
                    else:
                        print(f"No faces found in {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        print(f"Loaded {len(self.known_face_names)} known faces")
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        face_encodings = []
        face_names = []
        
        if face_locations:
            # Get face encodings using the face locations
            face_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)
            
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Noma'lum"
                
                # Find the best match
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                
                face_names.append(name)
        
        # Scale back up face locations
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4) 
                         for (top, right, bottom, left) in face_locations]
        
        return face_locations, face_names
    
    def draw_face_annotations(self, frame, face_locations, face_names):
        """Draw boxes and labels around faces"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label with name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
    
    def calculate_fps(self, frame):
        """Calculate and display FPS"""
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def run(self):
        """Run the face recognition system"""
        # Try different camera indices if 0 doesn't work
        for camera_index in [0, 1, 2]:
            video_capture = cv2.VideoCapture(camera_index)
            if video_capture.isOpened():
                print(f"Using camera index {camera_index}")
                break
        else:
            print("Could not open any camera")
            return
        
        print("Starting face recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Recognize faces
            face_locations, face_names = self.recognize_faces(frame)
            
            # Draw annotations
            self.draw_face_annotations(frame, face_locations, face_names)
            
            # Calculate FPS
            self.calculate_fps(frame)
            
            # Display the resulting frame
            cv2.imshow('Face Recognition', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.run()