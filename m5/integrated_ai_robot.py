import asyncio
import json
import numpy as np
import webrtcvad
import requests
import io
import wave
import httpx
import re
import torch
from snac import SNAC
import pyaudio
import threading
import queue
import time
import websockets
import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer
import logging


"""
build\bin\llama-server -hf unsloth/gemma-3n-E2B-it-GGUF:gemma-3n-E2B-it-UD-Q4_K_XL.gguf --prio 3 -c 2048 -e -n -2 -ngl 99 --port 8080 --host 0.0.0.0 --no-webui -v --cont-batching

.\build\bin\whisper-server.exe -m models/ggml-kotoba-whisper-v2.0.bin --host 0.0.0.0 --port 8082 -l ja --print-colors

build\bin\llama-server -hf webbigdata/VoiceCore_gguf:VoiceCore-Q4_K-f16.gguf --prio 3 -c 2048 -e -n -2 --port 8081 --host 0.0.0.0 -ngl 99 --no-webui -v --cont-batching
"""


# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedVoiceChatSystem:
    def __init__(self, serial_port='COM3'):
        # サーバーURL設定
        self.whisper_url = "http://localhost:8082"
        self.gemma_url = "http://localhost:8080"
        self.voicecore_url = "http://localhost:8081"
        
        # シリアル通信設定（AtomS3制御）
        self.serial_port = serial_port
        self.serial_connection = None
        self._init_serial()
        
        # 音声設定
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # VAD設定
        self.vad = webrtcvad.Vad(2)
        self.MIN_AUDIO_LENGTH = 0.5
        self.SILENCE_THRESHOLD = 30
        self.VOLUME_THRESHOLD = 0.01
        
        # VAD無効化制御（PC音声再生中の誤認識防止）
        self.is_vad_enabled = True
        self.vad_disable_start_time = None
        
        # 音声再生設定
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        self.playback_sample_rate = 24000
        
        # バッファリング制御
        self.initial_buffer = []  # 初回バッファリング用
        self.is_initial_buffering = True
        self.buffer_start_time = None
        
        # AI設定
        self.system_prompt = "あなたは親切なアシスタントです。簡潔に1文で返答してください。"
        self.conversation_history = []
        
        # モデル初期化
        self.snac_model = None
        self.tokenizer = None
        self._load_models()
    
    def _init_serial(self):
        """AtomS3シリアル接続初期化"""
        try:
            self.serial_connection = serial.Serial(self.serial_port, 115200, timeout=1)
            logger.info(f"AtomS3接続成功: {self.serial_port}")
            time.sleep(2)  # 接続安定化待機
        except serial.SerialException as e:
            logger.warning(f"AtomS3接続失敗: {e}")
            self.serial_connection = None
    
    def send_emotion_command(self, emotion):
        """AtomS3に感情コマンド送信（改行コード付き）"""
        if not self.serial_connection or not self.serial_connection.is_open:
            logger.warning("AtomS3が接続されていません")
            return
        
        try:
            command_map = {
                'smile': 's',
                'cry': 'c'
            }
            
            if emotion in command_map:
                command = command_map[emotion]
                # 改行コード付きで送信（AtomS3のreadline()対応）
                self.serial_connection.write(f'{command}\n'.encode('utf-8'))
                logger.info(f"AtomS3に送信: '{command}\\n'")
            else:
                logger.debug(f"ニュートラル感情のため送信なし: {emotion}")
        except Exception as e:
            logger.error(f"AtomS3コマンド送信エラー: {e}")
    
    def analyze_emotion_from_text(self, text):
        """テキストから感情分析"""
        text_lower = text.lower()
        
        # 喜び・挨拶のキーワード
        smile_keywords = [
            'こんにちは', 'おはよう', 'こんばんは', 'ありがとう', 'うれしい', 
            '嬉しい', '楽しい', '良い', 'いい', '素晴らしい', '最高', 'よかった',
            'ありがとうございます', 'おめでとう', '喜び', '幸せ', 'happy'
        ]
        
        # 悲しみのキーワード  
        cry_keywords = [
            '悲しい', 'かなしい', 'つらい', '辛い', '残念', '困った', 
            '心配', 'だめ', 'ダメ', '失敗', '嫌', 'いや', '最悪',
            'がっかり', 'ショック', '泣き', '涙', 'sad', 'sorry'
        ]
        
        # キーワードマッチング
        for keyword in smile_keywords:
            if keyword in text_lower:
                return 'smile'
        
        for keyword in cry_keywords:
            if keyword in text_lower:
                return 'cry'
        
        return 'neutral'
    
    def _load_models(self):
        """SNACモデルとトークナイザーを読み込み"""
        logger.info("モデルを読み込み中...")
        try:
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
            self.snac_model.to("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E2B-it")
            logger.info("モデル読み込み完了")
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
    
    def is_speech(self, audio_frame):
        """VADで音声判定（PC音声再生中は無効化）"""
        # VADが無効化されている場合は常にFalse
        if not self.is_vad_enabled:
            return False
        
        # VAD無効化期間をチェック
        if self.vad_disable_start_time is not None:
            elapsed_time = time.time() - self.vad_disable_start_time
            if elapsed_time < 8.0:  # 8秒間VAD無効
                return False
            else:
                # 8秒経過したらVAD再有効化
                self.vad_disable_start_time = None
                self.is_vad_enabled = True
                logger.info("🎤 VAD再有効化 - 音声入力受付再開")
        
        try:
            rms = np.sqrt(np.mean(audio_frame ** 2))
            if rms < self.VOLUME_THRESHOLD:
                return False
            
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            return self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
        except:
            return False
    
    def disable_vad_temporarily(self):
        """VADを一時的に無効化（PC音声再生開始時に呼び出し）"""
        self.is_vad_enabled = False
        self.vad_disable_start_time = time.time()
        logger.info("🔇 VAD無効化 - PC音声再生中（8秒間）")
    
    def _audio_to_wav_bytes(self, audio_data):
        """音声データをWAVバイトに変換"""
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    async def transcribe(self, audio_data):
        """Whisperで音声を文字に変換"""
        try:
            wav_bytes = self._audio_to_wav_bytes(audio_data)
            
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
            data = {
                'language': 'ja',
                'temperature': '0.0',
                'response_format': 'json'
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.whisper_url}/inference", files=files, data=data, timeout=30)
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                return text if text else None
            
        except Exception as e:
            logger.error(f"転写エラー: {e}")
        return None
    
    def _format_chat_prompt(self, user_input):
        """Gemma3N用プロンプト作成"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        prompt = ""
        for i, msg in enumerate(messages):
            is_last = (i == len(messages) - 1)
            if msg["role"] in ["user", "system"]:
                prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
                if is_last:
                    prompt += "<start_of_turn>model\n"
            elif msg["role"] == "assistant":
                prompt += f"<start_of_turn>model\n{msg['content']}"
                if not is_last:
                    prompt += "<end_of_turn>\n"
        
        return prompt
    
    async def chat_with_gemma(self, user_input):
        """Gemma3Nで応答生成"""
        try:
            prompt = self._format_chat_prompt(user_input)
            
            payload = {
                "prompt": prompt,
                "temperature": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "n_predict": 512,
                "stop": ["<end_of_turn>", "<|end_of_text|>"]
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.gemma_url}/completion", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('content', '').strip()
                    
                    # 履歴更新
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": content})
                    
                    # 履歴制限
                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-10:]
                    
                    return content
        except Exception as e:
            logger.error(f"Gemmaエラー: {e}")
        return None
    
    def _redistribute_codes(self, tokens):
        """SNACコード形式に変換"""
        code_list = [t - 128266 for t in tokens]
        layer_1, layer_2, layer_3 = [], [], []
        
        for i in range(len(code_list) // 7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
        
        return [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]
    
    def snac_decode_to_audio(self, tokens):
        """SNACで音声データに変換"""
        if not tokens:
            return None
        
        code_length = (len(tokens) // 7) * 7
        if code_length == 0:
            return None
        
        tokens = tokens[:code_length]
        codes = self._redistribute_codes(tokens)
        
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
            return audio_hat.detach().squeeze().cpu().numpy()
    
    async def generate_voice_tokens_streaming(self, text, callback=None):
        """
        VoiceCoreで音声トークンをストリーミング生成
        70トークン毎にコールバックを呼び出し（ストリーミング処理完全版）
        """
        voice_prompt = f"<custom_token_3><|begin_of_text|>amitaro_female[neutral]: {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
        
        payload = {
            "prompt": voice_prompt,
            "temperature": 0.8,
            "top_p": 0.95,
            "n_predict": 2048,
            "repeat_penalty": 1.1,
            "repeat_last_n": 70,
            "stream": True  # ストリーミング有効
        }
        
        collected_tokens = []
        all_tokens = []
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    'POST',
                    f"{self.voicecore_url}/completion",
                    json=payload,
                    headers={"Accept": "text/event-stream"}
                ) as response:
                    
                    logger.info(f"📡 ストリーミング接続確立: {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            # SSEプレフィックス除去
                            if line.startswith("data: "):
                                line = line[6:]
                            
                            try:
                                data = json.loads(line)
                                
                                # トークン抽出
                                if "content" in data:
                                    matches = re.findall(r'<custom_token_(\d+)>', data["content"])
                                    for match in matches:
                                        token_id = 128256 + int(match)
                                        collected_tokens.append(token_id)
                                        all_tokens.append(token_id)
                                        
                                        if len(collected_tokens) % 10 == 0:
                                            logger.debug(f"🎯 トークン受信中: {len(collected_tokens)}")
                                        
                                        # 70トークン毎にストリーミング処理
                                        if len(collected_tokens) >= 70:
                                            processable_count = (len(collected_tokens) // 7) * 7
                                            if processable_count > 0:
                                                tokens_to_process = collected_tokens[:processable_count]
                                                remaining_tokens = collected_tokens[processable_count:]
                                                
                                                logger.info(f"🎯 70トークン到達! ストリーミング処理: {len(tokens_to_process)}トークン")
                                                if callback:
                                                    await callback(tokens_to_process)
                                                
                                                collected_tokens = remaining_tokens
                                
                                # ストリーミング終了判定
                                if data.get("stop", False):
                                    logger.info("📍 ストリーミング終了")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"❌ トークン処理エラー: {e}")
                                continue
                
                # 残りトークンを処理
                if collected_tokens:
                    processable_count = (len(collected_tokens) // 7) * 7
                    if processable_count > 0:
                        tokens_to_process = collected_tokens[:processable_count]
                        logger.info(f"📦 最終バッチ処理: {len(tokens_to_process)}トークン")
                        if callback:
                            await callback(tokens_to_process)
            
            logger.info(f"✅ ストリーミング完了 - 総トークン数: {len(all_tokens)}")
            return all_tokens
            
        except Exception as e:
            logger.error(f"❌ VoiceCoreストリーミングエラー: {e}")
            return []
    
    def start_playback(self):
        """音声再生開始"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.is_playing = True
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
    
    def stop_playback(self):
        """音声再生停止"""
        self.is_playing = False
        self.audio_queue.put(None)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
    
    def _playback_worker(self):
        """音声再生ワーカー"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.playback_sample_rate,
                output=True,
                frames_per_buffer=2048
            )
            
            while self.is_playing:
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                    if audio_data is None:
                        break
                    
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                    
                    stream.write(audio_data.astype(np.float32).tobytes())
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"再生エラー: {e}")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            logger.error(f"再生ワーカーエラー: {e}")
    
    async def _streaming_audio_callback(self, tokens):
        """ストリーミング音声のコールバック関数（初回バッファリング対応）"""
        try:
            logger.debug(f"🔄 音声変換開始: {len(tokens)}トークン")
            audio_data = self.snac_decode_to_audio(tokens)
            if audio_data is not None:
                logger.debug(f"✅ 音声変換完了: {len(audio_data)}サンプル")
                
                if self.is_initial_buffering:
                    # 初回バッファリング中は配列に蓄積
                    self.initial_buffer.append(audio_data)
                    
                    if self.buffer_start_time is None:
                        self.buffer_start_time = time.time()
                    
                    # 2.5秒経過したら初回バッファリング終了
                    if time.time() - self.buffer_start_time >= 2.5:
                        logger.info("🎵 初回バッファリング完了 - 再生開始")
                        self.is_initial_buffering = False
                        
                        # 再生スレッド開始
                        self.start_playback()
                        
                        # 蓄積した音声データを全てキューに追加
                        for buffered_audio in self.initial_buffer:
                            self.audio_queue.put(buffered_audio)
                        
                        logger.info(f"📨 初回バッファ {len(self.initial_buffer)}チャンクをキューに追加")
                        self.initial_buffer = []  # バッファクリア
                else:
                    # 通常時は直接キューに追加
                    self.audio_queue.put(audio_data)
                    logger.debug(f"📨 音声チャンクをキューに追加 - サイズ: {self.audio_queue.qsize()}")
                    # 1秒間隔
                    await asyncio.sleep(1.0)
            else:
                logger.warning(f"❌ 音声変換失敗: トークン数={len(tokens)}")
        except Exception as e:
            logger.error(f"❌ 音声変換エラー: {e}")
    
    async def process_conversation(self, text, websocket=None):
        """音声対話処理の全体フロー（VAD制御付き）"""
        logger.info(f"ユーザー: {text}")
        
        # 1. Gemmaで応答生成
        response = await self.chat_with_gemma(text)
        if not response:
            return
        
        logger.info(f"応答: {response}")
        
        # 2. 感情分析とAtomS3制御
        emotion = self.analyze_emotion_from_text(response)
        logger.info(f"感情分析結果: {emotion}")
        self.send_emotion_command(emotion)
        
        # 3. WebSocketクライアントに感情情報送信
        if websocket:
            await websocket.send_json({
                "type": "emotion_detected", 
                "emotion": emotion
            })
        
        # 4. VAD無効化（PC音声再生の誤認識防止）
        self.disable_vad_temporarily()
        
        # 5. バッファリング状態をリセット
        self.is_initial_buffering = True
        self.initial_buffer = []
        self.buffer_start_time = None
        self.is_playing = False
        
        # キューをクリア
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # 6. ストリーミング音声生成開始
        logger.info("🎵 ストリーミング音声生成開始（初回2.5秒バッファリング）")
        
        try:
            # VoiceCoreでストリーミング音声トークン生成
            all_tokens = await self.generate_voice_tokens_streaming(
                response, 
                callback=self._streaming_audio_callback
            )
            
            logger.info(f"📊 総音声トークン数: {len(all_tokens)}")
            
            # まだバッファリング中の場合は強制的に再生開始
            if self.is_initial_buffering and self.initial_buffer:
                logger.info("🎵 生成完了 - 残りバッファを再生開始")
                self.is_initial_buffering = False
                self.start_playback()
                
                for buffered_audio in self.initial_buffer:
                    self.audio_queue.put(buffered_audio)
                logger.info(f"📨 最終バッファ {len(self.initial_buffer)}チャンクをキューに追加")
                self.initial_buffer = []
            
            # 音声再生完了まで待機
            estimated_duration = (len(all_tokens) // 70) * 1.0 + 5.0
            logger.info(f"⏳ 推定総再生時間: {estimated_duration:.1f}秒")
            
            start_wait = time.time()
            while not self.audio_queue.empty():
                await asyncio.sleep(0.2)
                if time.time() - start_wait > estimated_duration:
                    logger.warning("⚠️ 再生タイムアウト")
                    break
            
            await asyncio.sleep(0.5)
            logger.info("🎵 ストリーミング音声再生完了")
            
        except Exception as e:
            logger.error(f"❌ ストリーミング処理エラー: {e}")
        finally:
            logger.info("🛑 音声再生停止")
            self.stop_playback()
            # VADは自動的に8秒後に再有効化される
            logger.debug("感情表示継続")
    
    def close_serial(self):
        """シリアル接続を閉じる"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            logger.info("AtomS3接続を閉じました")

# FastAPIアプリケーション
app = FastAPI()
voice_system = IntegratedVoiceChatSystem()

@app.on_event("shutdown")
def shutdown_event():
    """アプリ終了時にシリアル接続を閉じる"""
    voice_system.close_serial()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket接続確立")
    
    is_recording = False
    audio_buffer = []
    silence_counter = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            if len(audio_data) >= voice_system.frame_size:
                frame = audio_data[:voice_system.frame_size]
                is_speech = voice_system.is_speech(frame)
                
                if is_speech:
                    if not is_recording:
                        is_recording = True
                        audio_buffer = []
                        logger.info("音声検出開始")
                        await websocket.send_json({"type": "speech_started"})
                    
                    audio_buffer.extend(audio_data)
                    silence_counter = 0
                else:
                    if is_recording:
                        audio_buffer.extend(audio_data)
                        silence_counter += 1
                        
                        if silence_counter >= voice_system.SILENCE_THRESHOLD:
                            audio_array = np.array(audio_buffer, dtype=np.float32)
                            duration = len(audio_array) / voice_system.sample_rate
                            
                            if duration >= voice_system.MIN_AUDIO_LENGTH:
                                logger.info(f"転写開始: {duration:.1f}秒")
                                await websocket.send_json({"type": "processing"})
                                
                                # 転写
                                text = await voice_system.transcribe(audio_array)
                                if text:
                                    logger.info(f"転写結果: {text}")
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "text": text
                                    })
                                    
                                    # 音声対話処理
                                    await voice_system.process_conversation(text, websocket)
                                    
                                    await websocket.send_json({
                                        "type": "response_complete"
                                    })
                            
                            is_recording = False
                            audio_buffer = []
                            silence_counter = 0
                            await websocket.send_json({"type": "speech_ended"})
    
    except WebSocketDisconnect:
        logger.info("WebSocket切断")
    except Exception as e:
        logger.error(f"WebSocketエラー: {e}")

@app.get("/")
async def root():
    return {"message": "統合音声対話システム稼働中"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 音声録音クライアント（シリアル通信なし）
class VoiceClient:
    def __init__(self, server_url="ws://localhost:8003/ws"):
        self.server_url = server_url
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio = pyaudio.PyAudio()
        # ClientはAtomS3にアクセスしない
        
    async def start_recording(self):
        """音声録音とWebSocket送信（接続安定化版）"""
        import websockets
        from websockets.exceptions import ConnectionClosed, ConnectionClosedError
        
        # マイク設定
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🎤 音声録音開始！話してください...")
        print("Ctrl+Cで終了")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # WebSocket接続設定（大幅な安定化）
                async with websockets.connect(
                    self.server_url,
                    ping_interval=None,   # pingを無効化（タイムアウト原因）
                    ping_timeout=None,    # pingタイムアウトを無効化
                    close_timeout=10,     # 接続終了タイムアウト延長
                    max_size=2**20,      # 最大メッセージサイズを1MB
                    max_queue=32         # キューサイズ制限
                ) as websocket:
                    
                    print(f"✅ サーバーに接続しました (試行 {retry_count + 1}/{max_retries})")
                    retry_count = 0  # 接続成功したらリセット
                    
                    while True:
                        try:
                            # 音声データ読み取り
                            audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                            
                            # WebSocketで送信
                            await websocket.send(audio_data)
                            
                            # サーバーからの応答受信（タイムアウトを延長）
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                                message = json.loads(response)
                                
                                if message["type"] == "speech_started":
                                    print("🎯 音声検出...")
                                elif message["type"] == "processing":
                                    print("🔄 処理中...")
                                elif message["type"] == "transcription":
                                    print(f"📝 認識: {message['text']}")
                                elif message["type"] == "emotion_detected":
                                    print(f"😊 感情: {message['emotion']}")
                                elif message["type"] == "response_complete":
                                    print("✅ 応答完了\n")
                            except asyncio.TimeoutError:
                                pass  # 応答なし（正常）
                            
                            await asyncio.sleep(0.02)  # 少し長めに
                        
                        except (ConnectionClosed, ConnectionClosedError) as e:
                            print(f"⚠️ WebSocket接続が切断されました: {e}")
                            break  # 内側ループを抜けて再接続
                        
                        except Exception as e:
                            print(f"⚠️ 音声送信エラー: {e}")
                            await asyncio.sleep(0.1)
                            continue
                            
            except KeyboardInterrupt:
                print("\n🛑 録音終了")
                break
            
            except (ConnectionClosed, ConnectionClosedError, OSError) as e:
                retry_count += 1
                print(f"❌ 接続エラー (試行 {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    print(f"🔄 {retry_count * 3}秒後に再接続を試行...")
                    await asyncio.sleep(retry_count * 3)  # より長い待機時間
                else:
                    print("❌ 最大再試行回数に達しました。サーバーが起動しているか確認してください。")
                    break
            
            except Exception as e:
                print(f"❌ 予期しないエラー: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"🔄 {retry_count * 2}秒後に再試行...")
                    await asyncio.sleep(retry_count * 2)
                else:
                    break
        
        # クリーンアップ
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        print("🎤 音声録音を終了しました")

# メイン実行部分
async def run_server():
    """サーバー起動"""
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8003, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        voice_system.close_serial()

async def run_client():
    """クライアント起動"""
    await asyncio.sleep(2)  # サーバー起動待機
    client = VoiceClient()
    await client.start_recording()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        # クライアントモード（シリアル通信なし）
        print("🎤 音声クライアント起動")
        print("※ AtomS3制御はサーバー側で実行されます")
        try:
            asyncio.run(VoiceClient().start_recording())
        except KeyboardInterrupt:
            print("終了")
    else:
        # サーバーモード（AtomS3制御あり）
        print("🚀 音声対話サーバー起動")
        print("AtomS3がCOM3に接続されていることを確認してください")
        print("別ターミナルで 'python このファイル名.py client' を実行してください")
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            print("サーバー終了")
        finally:
            voice_system.close_serial()