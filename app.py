from flask import Flask, render_template, Response
import cv2
import mediapipe as mp #check version
import pyautogui
import signal
import sys
import threading
import time
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import numpy as np # Import numpy for calculating RMS (Root Mean Square)
from transformers import pipeline
#confirm you 'Microsoft Visual C++' version (latest)
#download 'FFmpeg' in cmd (essential tool for recording)
#check you microphone, adjust value of RMS_THRESHOLD when not perform well
#first time run need more time to upload model
#LOOK ABOVE!! REALLY IMPORTANT!!!


# -------------------------- Hugging Face Whisper Setup --------------------------
# Load the Whisper model pipeline (using the tiny model for speed)
WHISPER_MODEL = "openai/whisper-tiny"
try:
    # Set device to 'cpu' if you don't have a CUDA-enabled GPU or want to ensure compatibility
    whisper_pipe = pipeline("automatic-speech-recognition", model=WHISPER_MODEL) 
    print(f"INFO: Successfully loaded Hugging Face Whisper model: {WHISPER_MODEL}")
except Exception as e:
    print(f"ERROR: Failed to load Whisper model. Please check dependencies: {e}")
    whisper_pipe = None

# -------------------------- Audio & Threading Config --------------------------
SAMPLE_RATE = 16000     # Required sample rate for Whisper
# MODIFIED: Use 3.5 seconds for quicker start/stop responsiveness, closer to original value
RECORD_SECONDS = 3.5      
CLICK_COOLDOWN = 0.5    # Cooldown time for mouse click (in seconds)
last_click_time = 0     # Timestamp of the last successful click

# NEW: Configure the specific input device index after checking the list below
# *** 重要：请根据控制台输出的设备列表，将此值改为您麦克风的序号 (index) ***
# 默认为 None，表示使用系统默认设备。如果您遇到低音量问题，请修改此值。
INPUT_DEVICE_INDEX = None 

is_running = True       # Global flag to control the entire application exit
prev_x, prev_y = 0, 0   # Global initialization for smoothing variables

# NEW LISTENER CONTROL VARIABLES
# 用于存储语音监听线程对象
listener_thread = None
# 用于控制语音监听线程内部循环的生命周期
is_listener_active = False 

# NEW: Common "silence filler" phrases to ignore (must be lowercase)
# 注意：现在需要包含更多的单次噪音词汇，因为我们移除了 > 1 词的限制
SILENCE_FILLERS = [
    "thank you.",
    "you.",
    "you",
    "thanks.",
    "music",  # Whisper sometimes confuses background noise with "music"
    "yeah",
    "uh huh",
    "oh", # 常见的单字噪音
    "ah", # 常见的单字噪音
    "the", # 常见的单字噪音
    "is", # 常见的单字噪音
    "to", # 常见的单字噪音
]

# NEW: Voice Activity Detection (VAD) Threshold
# Higher value = less sensitive to quiet speech.
# 已根据用户请求调整为 24
RMS_THRESHOLD = 24 



app = Flask(__name__)
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
is_capturing = False # Set initial capture state to False

# Function to print audio input devices for user configuration
def print_input_devices():
    devices = sd.query_devices()
    print("\n---------------- AUDIO INPUT DEVICE LIST ----------------")
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    
    if not input_devices:
        print("未找到任何可用的输入设备。")
        return
        
    for i, d in enumerate(input_devices):
        print(f"Index: {d['index']} | Name: {d['name']} | Max Channels: {d['max_input_channels']}")
    print("---------------------------------------------------------")
    print("如果您遇到音量过低问题，请将 INPUT_DEVICE_INDEX (在代码顶部) 设置为您麦克风的 Index。")
    print("---------------------------------------------------------\n")

# Function running in a separate, permanent thread for continuous transcription
def continuous_transcription_worker():
    """Handles continuous audio recording, Whisper transcription, and text input via pyautogui."""
    # Note: is_listener_active is used to control the loop instead of is_capturing
    global is_listener_active, INPUT_DEVICE_INDEX, RMS_THRESHOLD
    
    if not whisper_pipe:
        print("Whisper pipeline is not initialized. Cannot start continuous listening.")
        return

    temp_wav_file = "temp_input.wav"
    print("--- VOICE INPUT: Continuous listening mode started ---")
    
    # Loop as long as the listener is active (set True in /start_capture, False in /stop_capture)
    while is_listener_active:
        
        # 1. Start Recording (blocking for RECORD_SECONDS)
        print(f"VOICE INPUT: Recording segment ({RECORD_SECONDS}s)...")
        recording = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='int16',
            device=INPUT_DEVICE_INDEX
        )
        sd.wait() # Wait until recording is finished
        
        # 2. VAD: Calculate RMS (Root Mean Square) to check volume
        rms = np.sqrt(np.mean(recording**2))
        print(f"VOICE INPUT: RMS level detected: {int(rms)}. Threshold: {RMS_THRESHOLD}")
        
        # Check if the volume is below the silence threshold
        if rms < RMS_THRESHOLD:
            continue # Skip the rest of the loop and start the next recording segment

        # 3. Save to Temporary File (Only if volume is above threshold)
        write_wav(temp_wav_file, SAMPLE_RATE, recording)
        
        # 4. Whisper Transcription
        try:
            result = whisper_pipe(temp_wav_file)
            transcribed_text = result["text"].strip()
            
            # --- MODIFIED: More robust filtering logic ---
            normalized_text = transcribed_text.lower().replace('.', '').replace('!', '').replace('?', '').strip()

            # 5. Use PyAutoGUI to type the text ONLY if meaningful speech is detected
            if transcribed_text and normalized_text not in SILENCE_FILLERS:
                print(f"VOICE INPUT: Transcribed and typing: '{transcribed_text}'")
                pyautogui.write(transcribed_text + ' ', interval=0.01) 
            
        except Exception as e:
            if "ffmpeg was not found" not in str(e):
                print(f"Transcription failed unexpectedly: {e}")
                
    print("--- VOICE INPUT: Listener gracefully stopped ---")


def gen_frames():
    """Generates the video stream frames and handles mouse control logic."""
    # Declare global variables used across iterations
    global is_capturing, last_click_time, prev_x, prev_y
    # Mouse control parameters
    sensitivity = 1.5   
    smoothing = 0.5    
    
    while True:
        if not is_capturing:
            # Continue to loop to serve Flask request, but skip processing
            time.sleep(0.1)
            continue
            
        _, frame = cam.read()
        if not _:
            print("Failed to read frame from camera.")
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark
            
            # Use center point of both eyes for stable tracking
            left_eye_center = [(landmarks[33].x + landmarks[133].x)/2, 
                               (landmarks[33].y + landmarks[133].y)/2]
            right_eye_center = [(landmarks[362].x + landmarks[263].x)/2,
                                (landmarks[362].y + landmarks[263].y)/2]
            
            eye_center_x = int((left_eye_center[0] + right_eye_center[0])/2 * frame_w)
            # Reverting eye center Y calculation to ensure tracking stability
            eye_center_y = int((left_eye_center[1] + right_eye_center[1])/2 * frame_h)
            
            # Draw tracking point
            cv2.circle(frame, (eye_center_x, eye_center_y), 3, (0, 255, 255), -1)
            
            # Map camera coordinates to screen coordinates
            screen_x = screen_w * (eye_center_x / frame_w) * sensitivity
            screen_y = screen_h * (eye_center_y / frame_h) * sensitivity
            
            # Smoothing filter (uses global/persistent prev_x, prev_y)
            screen_x = prev_x * smoothing + screen_x * (1 - smoothing)
            screen_y = prev_y * smoothing + screen_y * (1 - smoothing)
            
            # Update mouse position
            pyautogui.moveTo(int(screen_x), int(screen_y))
            # Update the global/persistent variables
            prev_x, prev_y = screen_x, screen_y 
            
            # --------------------- Blink Detection (NOW: Mouse Click) ---------------------
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x,y), 3, (0, 255, 0)) # Green circle for visual feedback
            
            current_time = time.time()
            
            # If blink detected AND cooldown passed, execute mouse click
            if (left[0].y - left[1].y) < 0.003 and \
               (current_time - last_click_time) > CLICK_COOLDOWN:
                
                pyautogui.click()
                last_click_time = current_time # Reset cooldown

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Global cleanup for graceful exit
def signal_handler(sig, frame):
    global is_running
    print('Exiting gracefully')
    # Signal the background transcription thread to stop
    is_running = False 
    # Give the thread a moment to clean up before closing camera
    time.sleep(0.5) 
    cam.release()
    cv2.destroyAllWindows()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

@app.route('/start_capture')
def start_capture():
    global is_capturing, listener_thread, is_listener_active
    is_capturing = True
    
    # NEW LOGIC: Start the transcription thread only if it's not already running
    if listener_thread is None or not listener_thread.is_alive():
        print("Starting voice listener thread.")
        is_listener_active = True
        listener_thread = threading.Thread(target=continuous_transcription_worker, daemon=True)
        listener_thread.start()
    
    return 'OK'

@app.route('/stop_capture')
def stop_capture():
    global is_capturing, listener_thread, is_listener_active
    is_capturing = False
    
    # NEW LOGIC: Stop the transcription thread gracefully
    if listener_thread and listener_thread.is_alive():
        print("Stopping voice listener thread. Waiting for current recording cycle to finish...")
        is_listener_active = False # Set flag to exit worker loop
        # We rely on the daemon thread nature and the flag. The print statement in the worker confirms the stop.
        
    return 'OK'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # --- IMPORTANT: Print device list for user configuration ---
    print_input_devices()
    
    # --- REMOVED: Initial listener thread start. It is now started in /start_capture ---
    
    try:
        print("Starting application. Press Ctrl+C to exit.")
        # use_reloader=False is crucial when using threading
        app.run(debug=True, use_reloader=False) 
    finally:
        # Final cleanup if app exits.
        is_running = False
        cam.release()
        cv2.destroyAllWindows()
