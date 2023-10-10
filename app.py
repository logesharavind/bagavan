from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64
import cv2
import numpy as np
from flask_cors import CORS, cross_origin
import imutils
import mediapipe as mp
from engineio.payload import Payload

Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)

@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)

global fps, prev_recv_time, cnt, fps_array
fps = 30
prev_recv_time = 0
cnt = 0
fps_array = [0]

@socketio.on('image')
def image(data_image):
    global fps, cnt, prev_recv_time, fps_array
    recv_time = time.time()
    text = 'FPS: ' + str(fps)
    frame = readb64(data_image)

    frame = detect_faces(frame)
    
    #frame = ps.putBText(frame, text, text_offset_x=20, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0, background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))
    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

    fps = 1 / (recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)), 1)
    prev_recv_time = recv_time
    # print(fps_array)
    cnt += 1
    if cnt == 30:
        fps_array = [fps]
        cnt = 0

# Initialize the MediaPipe Face Detection component
#mp_face_detection = mp.solutions.face_detection
#face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Replace with the path of your default necklace image
necklace_image_path = 'static/Image/Necklace/necklace_1.png'
# Load the necklace image
necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)

# Load the MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

def detect_faces(frame):
    # Get the width and height of the frame
    height, width, _ = frame.shape

    # Calculate the width of each section
    center_width = int(width * 0.85)
    side_width = (width - center_width) // 2

    # Divide the frame into three sections
    left_section = frame[:, :side_width]
    center_section = frame[:, side_width:side_width + center_width]
    right_section = frame[:, side_width + center_width:]

    # Draw vertical lines to split the screen
    cv2.line(frame, (side_width, 0), (side_width, height), (0, 255, 0), 2)
    cv2.line(frame, (side_width + center_width, 0), (side_width + center_width, height), (0, 255, 0), 2)


    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        # Process the center section for face detection and overlay
        frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                # Iterate over the landmarks and draw them on the frame
                for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                    # Get the pixel coordinates of the landmark
                    cx, cy = int(landmark.x * center_width), int(landmark.y * height)

                    # Draw a circle on the center section at the landmark position
                    # cv2.circle(center_section, (cx, cy), 5, (0, 255, 0), -1)

                    # Check if hand is within the valid region
                    if cx >= 0 and cx <= center_width:
                        hand_in_frame = True

                    # Extract the bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    hC, wC, _ = center_section.shape
                    xminC = int(bboxC.xmin * wC)
                    yminC = int(bboxC.ymin * hC)
                    widthC = int(bboxC.width * wC)
                    heightC = int(bboxC.height * hC)
                    xmaxC = xminC + widthC
                    ymaxC = yminC + heightC

                    # Calculate the bottom bounding box coordinates
                    bottom_ymin = ymaxC + 10
                    bottom_ymax = min(ymaxC + 150, hC)

                    # Increase the width of the red bounding box
                    xminC -= 20  # Decrease the left side
                    xmaxC += 20  # Increase the right side

                    # Check if the bounding box dimensions are valid
                    if widthC > 0 and heightC > 0 and xmaxC > xminC and bottom_ymax > bottom_ymin:
                        # Resize necklace image to fit the bounding box size
                        resized_image = cv2.resize(necklace_image, (xmaxC - xminC, bottom_ymax - bottom_ymin))

                        # Calculate the start and end coordinates for the necklace image
                        start_x = xminC
                        start_y = bottom_ymin
                        end_x = start_x + (xmaxC - xminC)
                        end_y = start_y + (bottom_ymax - bottom_ymin)

                        # Create a mask from the alpha channel
                        alpha_channel = resized_image[:, :, 3]
                        mask = alpha_channel[:, :, np.newaxis] / 255.0

                        # Apply the mask to the necklace image
                        overlay = resized_image[:, :, :3] * mask

                        # Create a mask for the input image region
                        mask_inv = 1 - mask

                        # Apply the inverse mask to the input image
                        region = center_section[start_y:end_y, start_x:end_x]
                        resized_mask_inv = None
                        if region.shape[1] > 0 and region.shape[0] > 0:
                            resized_mask_inv = cv2.resize(mask_inv, (region.shape[1], region.shape[0]))
                            resized_mask_inv = resized_mask_inv[:, :, np.newaxis]  # Add an extra dimension to match the number of channels

                        if resized_mask_inv is not None:
                            region_inv = region * resized_mask_inv

                            # Combine the resized image and the input image region
                            resized_overlay = None
                            if region_inv.shape[1] > 0 and region_inv.shape[0] > 0:
                                resized_overlay = cv2.resize(overlay, (region_inv.shape[1], region_inv.shape[0]))

                            # Combine the resized overlay and region_inv
                            region_combined = cv2.add(resized_overlay, region_inv)

                            # Replace the neck region in the input image with the combined region
                            center_section[start_y:end_y, start_x:end_x] = region_combined

        # Concatenate the sections back into a single frame
        frame = np.concatenate((left_section, center_section, right_section), axis=1)

    return frame

if __name__ == '__main__':
    socketio.run(app, port=8080, debug=True)
