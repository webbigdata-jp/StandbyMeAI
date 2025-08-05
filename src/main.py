#!/usr/bin/env python3
"""
StandbyMeAI: Webcam-integrated AI Assistant with Symbolic Face Display
"""

# --- Standard Libraries ---
import os
import sys
import asyncio
from datetime import datetime, timedelta
from PIL import Image
import cv2

from face_display import FaceDisplay
from image_processing import ImageGreetingGenerator
from tts_system import TextToSpeechSystem


# --- Debug Settings ---
DEBUG = True

# -----------------------------------------------------------------------------
# Application Class
# -----------------------------------------------------------------------------
class Application:
    def __init__(self):
        print("StandbyMeAI: Webcam-integrated AI Assistant (Press Ctrl+C to exit)")
        print("=" * 70)
        self.greeting_generator = None
        self.tts_system = None
        self.face_display = None
        self.cap = None
        self.output_dir = "audio_outputs"
        
    def initialize_systems(self):
        try:
            self.greeting_generator = ImageGreetingGenerator()
            self.tts_system = TextToSpeechSystem()
            self.face_display = FaceDisplay()
            
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"\nAudio will be saved in the '{self.output_dir}' folder.")
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("✗ Could not open webcam.")
                return False
            return True
        except Exception as e:
            print(f"\n✗ Initialization failed: {e}")
            return False
            
    async def main_loop(self):
        self.face_display.show("waiting")
        try:
            while True:
                print("\n" + "=" * 70 + f"\n{datetime.now():%H:%M:%S} - Starting next cycle...")
                
                scanning_task = asyncio.create_task(self.face_display.animate_scanning())
                
                ret, frame = self.cap.read()
                if not ret:
                    print("✗ Failed to capture frame.")
                    break
                
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                response = self.greeting_generator.generate_greeting_from_image(image_pil)
                
                scanning_task.cancel()
                await asyncio.sleep(0) # Allow cancellation to be processed

                if response and response.strip().upper() != "NO":
                    print(f"\n💬 Generated Response: {response}")
                    save_fn = os.path.join(self.output_dir, f"output_{datetime.now():%Y%m%d_%H%M%S}.wav")
                    audio_data, duration = await self.tts_system.generate_speech_data(response, save_fn)
                    
                    if audio_data is not None and duration > 0:
                        speak_animation_task = asyncio.create_task(self.face_display.animate_speaking(duration))
                        play_audio_task = asyncio.to_thread(self.tts_system.play_audio, audio_data)
                        await asyncio.gather(speak_animation_task, play_audio_task)
                    else:
                        self.face_display.show("waiting")
                else:
                    if response is None:
                        print("\n-> Response generation failed.")
                    else:
                        print("\n-> Target person not detected.")
                    self.face_display.show("waiting")

                wait_seconds = 60
                next_run_time = datetime.now() + timedelta(seconds=wait_seconds)
                print(f"\n--- Waiting for {wait_seconds}s (Next run at: {next_run_time:%H:%M:%S}) ---")
                for _ in range(wait_seconds):
                    await asyncio.sleep(1)
                    cv2.waitKey(1) # Keep window responsive

        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        finally:
            self.cleanup()
            
    def cleanup(self):
        print("Cleaning up resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Webcam released.")
        cv2.destroyAllWindows()
        print("Windows destroyed.")
        print("Program finished.")
        
    async def run(self):
        if self.initialize_systems():
            await self.main_loop()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    app = Application()
    try:
        asyncio.run(app.run())
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")