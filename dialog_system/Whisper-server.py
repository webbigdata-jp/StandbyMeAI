import asyncio
import json
import logging
import numpy as np
import webrtcvad
import requests
import io
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from collections import deque
import time

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElderlyOptimizedAudioProcessor:
    def __init__(self, whisper_server_url="http://localhost:8082", demo_mode=True):
        # Whisper.cppサーバーのURL
        self.whisper_server_url = whisper_server_url
        
        # デモモードフラグ
        self.DEMO_MODE = demo_mode
        
        # 高齢者対応パラメーター（デモモードで調整）
        self.ELDERLY_MODE = True
        self.MIN_AUDIO_LENGTH = 0.8 if not demo_mode else 0.5  # 最小音声長
        self.MIN_SPEECH_DURATION = 0.3  # 音声検出開始
        #self.SILENCE_THRESHOLD = 75 if not demo_mode else 15  # デモ: 0.5秒で転写
        self.SILENCE_THRESHOLD = 75 if not demo_mode else 50
        
        #self.SPEECH_END_TIMEOUT = 5.0 if not demo_mode else 2.0  # タイムアウト
        self.SPEECH_END_TIMEOUT = 5.0 if not demo_mode else 4.0  # タイムアウト
        
        self.MIN_TRANSCRIPTION_INTERVAL = 1.0 if not demo_mode else 0.5  # 転写間隔
        self.MAX_SINGLE_UTTERANCE = 15.0 if not demo_mode else 10.0  # 最大発話長
        self.VOLUME_THRESHOLD = 0.010  # より低いノイズ閾値に変更
        self.MIN_STANDALONE_LENGTH = 1.5  # 前発話との結合閾値
        
        # 音声の前後バッファ（重要：文頭の欠落を防ぐ）
        self.PRE_SPEECH_BUFFER = 0.3  # 音声検出前の0.3秒を含める（短縮）
        self.POST_SPEECH_BUFFER = 0.2  # 音声終了後の0.2秒を含める（短縮）
        
        # VADの初期化（高感度に設定）
        # self.vad = webrtcvad.Vad(2 if demo_mode else 1)  # デモモードでは高感度
        self.vad = webrtcvad.Vad(2)

        # 初回音声の切れを防ぐための追加パラメータ
        self.CONTINUOUS_SPEECH_THRESHOLD = 10  # 連続した音声フレーム数
        self.continuous_speech_count = 0  # 連続音声カウンター

        
        # 音声バッファとタイミング管理
        self.audio_buffer = deque(maxlen=1600 * 60)  # 60秒分のバッファ
        self.pre_speech_buffer = deque(maxlen=int(16000 * self.PRE_SPEECH_BUFFER))  # 前バッファ
        self.silence_count = 0
        self.speech_detected = False
        self.last_transcription_time = 0
        self.recording_start_time = 0
        self.last_speech_time = 0
        self.speech_start_index = 0  # 音声開始位置を記録
        
        # 音声設定
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # 前発話との結合用
        self.pending_audio = None
        self.pending_start_time = 0
        
        # Whisper.cppサーバーの接続確認
        self._check_whisper_server()
        
    def _check_whisper_server(self):
        """Whisper.cppサーバーの接続確認"""
        try:
            response = requests.get(f"{self.whisper_server_url}/", timeout=5)
            logger.info(f"Whisper.cppサーバー応答: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Whisper.cppサーバーに接続しました (デモモード: {self.DEMO_MODE})")
            else:
                logger.warning("Whisper.cppサーバーの応答が異常です")
        except requests.exceptions.RequestException as e:
            logger.error(f"Whisper.cppサーバーに接続できません: {e}")
            logger.info("サーバーが起動していることを確認してください:")
            logger.info("./server -m models/your-model.bin --host 0.0.0.0 --port 8082")
    
    def is_speech(self, audio_frame):
        """VADで音声かどうかを判定"""
        try:
            # 音量チェック
            rms = np.sqrt(np.mean(audio_frame ** 2))
            if rms < self.VOLUME_THRESHOLD:
                return False
            
            # 16bit PCMに変換してVAD判定
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except:
            return False
    
    def add_audio(self, audio_data):
        """音声データをバッファに追加"""
        # 音声検出前は前バッファに保存
        if not self.speech_detected:
            self.pre_speech_buffer.extend(audio_data)
        
        self.audio_buffer.extend(audio_data)
    
    def get_audio_for_transcription(self):
        """転写用の音声データを取得（前後のバッファを含む）"""
        # 現在のバッファの実際の音声長を確認
        current_buffer_length = len(self.audio_buffer) / self.sample_rate
        
        # 実際の音声が短すぎる場合は、前バッファを含めない
        if current_buffer_length < 1.0:  # 1秒未満の短い音声
            # バッファのデータのみ使用
            audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
        else:
            # 長い音声の場合は前バッファを含める
            all_audio = list(self.audio_buffer)
            
            if self.speech_start_index > 0:
                # 音声開始前の一部を含める（最大0.3秒）
                pre_samples = int(self.sample_rate * 0.3)  # 0.5秒→0.3秒に短縮
                start_index = max(0, self.speech_start_index - pre_samples)
                audio_data = all_audio[start_index:]
            else:
                # 前バッファの最後の0.2秒のみを追加
                pre_buffer_data = list(self.pre_speech_buffer)[-int(self.sample_rate * 0.2):]
                audio_data = pre_buffer_data + all_audio
            
            audio_array = np.array(audio_data, dtype=np.float32)
        
        current_length = len(audio_array) / self.sample_rate
        
        # 最小音声長チェック
        if current_length < self.MIN_AUDIO_LENGTH:
            return None, None
        
        # 正規化
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # メタデータ作成
        current_time = time.time()
        metadata = {
            "duration": current_length,
            "actual_speech_duration": current_buffer_length,  # 実際の音声長
            "pause_before": current_time - self.last_transcription_time if self.last_transcription_time > 0 else 0,
            "recording_start": self.recording_start_time,
            "timestamp": current_time,
            "demo_mode": self.DEMO_MODE
        }
        
        return audio_array, metadata
    
    def _audio_to_wav_bytes(self, audio_data):
        """音声データをWAVバイトストリームに変換（16-bit PCM, 16kHz確実に）"""
        # デバッグ：入力データの確認
        logger.debug(f"入力音声データ: min={np.min(audio_data):.3f}, max={np.max(audio_data):.3f}, mean={np.mean(np.abs(audio_data)):.3f}")
        
        # 無音チェック
        if np.max(np.abs(audio_data)) < 0.001:
            logger.warning("警告: 音声データがほぼ無音です")
        
        # 正規化（-1.0 to 1.0の範囲に）
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9  # 少し余裕を持たせる
        
        # 16bit PCMに変換（-32768 to 32767）
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # WAVファイルヘッダーを含むバイトストリームを作成
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)        # モノラル
            wav_file.setsampwidth(2)        # 16bit (2 bytes)
            wav_file.setframerate(16000)    # 16kHz
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_bytes = wav_buffer.getvalue()
        wav_buffer.close()
        
        # デバッグ情報
        logger.debug(f"WAV詳細: {len(wav_bytes)} bytes, {len(audio_int16)} samples, {len(audio_int16)/16000:.1f}秒")
        
        return wav_bytes
    
    def should_merge_with_previous(self, duration):
        """前の発話と結合すべきかを判定"""
        return (duration < self.MIN_STANDALONE_LENGTH and 
                self.pending_audio is not None and
                time.time() - self.pending_start_time < 10.0)  # 10秒以内
    
    async def transcribe(self, audio_data, metadata):
        """Whisper.cppサーバーで音声を転写"""
        try:
            # 短い音声の結合判定
            if self.should_merge_with_previous(metadata["duration"]):
                logger.info(f"短い発話を前発話と結合: {metadata['duration']:.1f}秒")
                # 前の音声と結合
                combined_audio = np.concatenate([self.pending_audio, audio_data])
                self.pending_audio = combined_audio
                metadata["duration"] = len(combined_audio) / self.sample_rate
                audio_data = combined_audio
            else:
                # 新しい発話として保存
                self.pending_audio = audio_data.copy()
                self.pending_start_time = time.time()
            
            # 音声データをWAV形式に変換（16-bit PCM, 16kHz確実に）
            wav_bytes = self._audio_to_wav_bytes(audio_data)
            
            logger.info(f"WAV生成: {len(wav_bytes)} bytes, {metadata['duration']:.1f}秒")
            
            # Whisper.cppサーバーにPOSTリクエスト（最適化されたパラメータ）
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
            
            # より良い認識のためのパラメータ
            data = {
                'language': 'ja',  # 日本語を明示的に指定
                'temperature': '0.0',  # 決定的な出力
                'temperature_inc': '0.2',
                'response_format': 'json',
                'beam_size': '5',  # ビームサーチを使用
                'best_of': '5',    # 複数の候補から最良を選択
                'prompt': '日本語の日常会話。明瞭に話す。',  # プロンプトを改善
                'no_speech_threshold': '0.6',  # 無音判定の閾値
                'compression_ratio_threshold': '2.4',
                'logprob_threshold': '-1.0'
            }
            
            # 短い音声の場合は、より慎重なパラメータを使用
            if metadata.get('actual_speech_duration', metadata['duration']) < 1.0:
                data['no_speech_threshold'] = '0.8'  # より厳しい無音判定
                data['temperature'] = '0.2'  # 少し柔軟に
            
            # 非同期リクエスト
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.whisper_server_url}/inference",
                    files=files,
                    data=data,
                    timeout=30
                )
            )
            
            logger.info(f"Whisper.cpp応答: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                
                # レスポンス詳細をログ出力
                logger.debug(f"Whisper応答詳細: {result}")
                
                # 結果にメタデータを追加
                full_result = {
                    "text": text,
                    "metadata": metadata,
                    "confidence": "high" if len(text) > 0 else "low",
                    "raw_response": result  # デバッグ用
                }
                
                if text:
                    # 発話タイプの簡易推定
                    if any(greeting in text for greeting in ['おはよう', 'こんにちは', 'こんばんは', 'ありがとう', 'すみません']):
                        full_result["speech_type"] = "greeting"
                    elif metadata["duration"] > 10:
                        full_result["speech_type"] = "long_story"
                    else:
                        full_result["speech_type"] = "conversation"
                    
                    logger.info(f"転写結果: [{full_result['speech_type']}] {text} ({metadata['duration']:.1f}秒)")
                else:
                    logger.warning(f"転写結果が空です。応答: {result}")
                
                self.last_transcription_time = time.time()
                return full_result
            else:
                logger.error(f"Whisper.cppサーバーエラー: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"転写エラー: {e}")
            import traceback
            logger.error(f"スタックトレース: {traceback.format_exc()}")
            return None
    
    def should_start_transcription(self):
        """転写を開始すべきかを判定（改善版）"""
        current_time = time.time()
        
        # 最小間隔チェック
        if current_time - self.last_transcription_time < self.MIN_TRANSCRIPTION_INTERVAL:
            return False
        
        # 実際の音声長を確認
        actual_speech_length = len(self.audio_buffer) / self.sample_rate
        
        # デモモードでの追加チェック
        if self.DEMO_MODE:
            # 短い音声でも最低限の長さは必要
            if actual_speech_length < 0.5:  # 0.3→0.5秒に増やす
                return False
            
            # 連続音声が少ない場合は、より長い無音を要求
            if self.continuous_speech_count < self.CONTINUOUS_SPEECH_THRESHOLD:
                # より長い無音が必要（通常の1.5倍）
                if self.silence_count < self.SILENCE_THRESHOLD * 1.5:
                    return False
        
        # 十分な音声長があるかチェック
        if actual_speech_length < self.MIN_AUDIO_LENGTH:
            return False
        
        # 強制分割チェック（長い発話）
        if (self.recording_start_time > 0 and 
            current_time - self.recording_start_time > self.MAX_SINGLE_UTTERANCE):
            logger.info("長い発話を強制分割します")
            return True
        
        # 通常の無音判定
        return self.silence_count >= self.SILENCE_THRESHOLD

# FastAPIアプリケーション
app = FastAPI()

# 静的ファイルの配信
app.mount("/static", StaticFiles(directory="static"), name="static")

# 音声処理インスタンス（デモモードで起動）
audio_processor = ElderlyOptimizedAudioProcessor(demo_mode=True)  # デモモードON


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket接続が確立されました (デモモード: {audio_processor.DEMO_MODE})")
    
    # シンプルな状態管理
    is_recording = False
    audio_buffer = []
    silence_counter = 0
    SILENCE_THRESHOLD = 50  # 約1.5秒の無音で転写
    
    try:
        while True:
            # クライアントからの音声データを受信
            data = await websocket.receive_bytes()
            
            # バイナリデータをfloat32配列に変換
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # フレームごとにVADチェック
            if len(audio_data) >= audio_processor.frame_size:
                frame = audio_data[:audio_processor.frame_size]
                current_time = time.time()
                
                is_speech = audio_processor.is_speech(frame)
                
                if is_speech:
                    # 音声検出
                    if not is_recording:
                        is_recording = True
                        audio_buffer = []  # 新しい録音開始
                        logger.info("音声検出開始")
                        await websocket.send_json({"type": "speech_started", "timestamp": current_time})
                    
                    # 音声データを追加
                    audio_buffer.extend(audio_data)
                    silence_counter = 0  # 無音カウンターリセット
                    
                else:
                    # 無音検出
                    if is_recording:
                        # 録音中の無音はバッファに追加（音声の終端を含めるため）
                        audio_buffer.extend(audio_data)
                        silence_counter += 1
                        
                        # 無音が閾値を超えたら転写
                        if silence_counter >= SILENCE_THRESHOLD:
                            # 音声データを準備
                            audio_array = np.array(audio_buffer, dtype=np.float32)
                            duration = len(audio_array) / audio_processor.sample_rate
                            
                            # 最小長チェック
                            if duration >= 0.5:  # 0.5秒以上の音声のみ転写
                                logger.info(f"転写開始: {duration:.1f}秒")
                                await websocket.send_json({"type": "processing", "timestamp": current_time})
                                
                                # メタデータ作成
                                metadata = {
                                    "duration": duration,
                                    "timestamp": current_time,
                                    "demo_mode": audio_processor.DEMO_MODE
                                }
                                
                                # 転写実行
                                result = await audio_processor.transcribe(audio_array, metadata)
                                
                                if result and result["text"]:
                                    logger.info(f"転写成功: {result['text']}")
                                    await websocket.send_json({
                                        "type": "transcription", 
                                        "result": result, 
                                        "timestamp": current_time
                                    })
                            
                            # 録音終了・リセット
                            is_recording = False
                            audio_buffer = []
                            silence_counter = 0
                            logger.info("録音終了・待機状態へ")
                            await websocket.send_json({"type": "speech_ended", "timestamp": current_time})
    
    except WebSocketDisconnect:
        logger.info("WebSocket接続が切断されました")
    except Exception as e:
        logger.error(f"WebSocketエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        

@app.get("/")
async def read_root():
    mode = "デモモード（高速）" if audio_processor.DEMO_MODE else "高齢者モード（標準）"
    return {
        "message": f"高齢者向けリアルタイム音声認識サーバーが稼働中です - {mode}",
        "mode": "elderly_optimized",
        "demo_mode": audio_processor.DEMO_MODE,
        "whisper_server": audio_processor.whisper_server_url
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "whisper_server": audio_processor.whisper_server_url,
        "elderly_mode": audio_processor.ELDERLY_MODE,
        "demo_mode": audio_processor.DEMO_MODE,
        "parameters": {
            "min_audio_length": audio_processor.MIN_AUDIO_LENGTH,
            "silence_threshold": audio_processor.SILENCE_THRESHOLD,
            "speech_end_timeout": audio_processor.SPEECH_END_TIMEOUT
        }
    }

@app.get("/config")
async def get_config():
    """現在の設定を取得"""
    return {
        "elderly_mode": audio_processor.ELDERLY_MODE,
        "demo_mode": audio_processor.DEMO_MODE,
        "min_audio_length": audio_processor.MIN_AUDIO_LENGTH,
        "min_speech_duration": audio_processor.MIN_SPEECH_DURATION,
        "silence_threshold": audio_processor.SILENCE_THRESHOLD,
        "speech_end_timeout": audio_processor.SPEECH_END_TIMEOUT,
        "min_transcription_interval": audio_processor.MIN_TRANSCRIPTION_INTERVAL,
        "max_single_utterance": audio_processor.MAX_SINGLE_UTTERANCE,
        "volume_threshold": audio_processor.VOLUME_THRESHOLD,
        "min_standalone_length": audio_processor.MIN_STANDALONE_LENGTH,
        "pre_speech_buffer": audio_processor.PRE_SPEECH_BUFFER,
        "post_speech_buffer": audio_processor.POST_SPEECH_BUFFER
    }

@app.post("/config/mode")
async def set_mode(demo_mode: bool = True):
    """デモモードの切り替え"""
    audio_processor.DEMO_MODE = demo_mode
    
    # パラメータを調整
    if demo_mode:
        audio_processor.MIN_AUDIO_LENGTH = 0.5
        audio_processor.SILENCE_THRESHOLD = 15  # 0.5秒
        audio_processor.SPEECH_END_TIMEOUT = 2.0
        audio_processor.MIN_TRANSCRIPTION_INTERVAL = 0.5
        audio_processor.MAX_SINGLE_UTTERANCE = 10.0
        audio_processor.vad = webrtcvad.Vad(2)  # 高感度
    else:
        audio_processor.MIN_AUDIO_LENGTH = 0.8
        audio_processor.SILENCE_THRESHOLD = 75  # 2.5秒
        audio_processor.SPEECH_END_TIMEOUT = 5.0
        audio_processor.MIN_TRANSCRIPTION_INTERVAL = 1.0
        audio_processor.MAX_SINGLE_UTTERANCE = 15.0
        audio_processor.vad = webrtcvad.Vad(1)  # 標準感度
    
    return {"demo_mode": demo_mode, "message": f"モードを{'デモ' if demo_mode else '標準'}に切り替えました"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)