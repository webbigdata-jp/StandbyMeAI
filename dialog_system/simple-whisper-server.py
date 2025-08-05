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
import os
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SimpleAudioProcessor:
    def __init__(self, whisper_server_url="http://localhost:8082", save_debug_wav=True):
        # Whisper.cppサーバーのURL
        self.whisper_server_url = whisper_server_url
        
        # デバッグ用WAV保存設定
        self.save_debug_wav_enabled = save_debug_wav
        self.debug_wav_dir = "debug_wav"
        if self.save_debug_wav_enabled and not os.path.exists(self.debug_wav_dir):
            os.makedirs(self.debug_wav_dir)
        
        # シンプルなパラメーター
        self.MIN_AUDIO_LENGTH = 0.5  # 最小音声長（秒）
        self.SILENCE_THRESHOLD = 30  # 無音フレーム数（約1秒）
        self.VOLUME_THRESHOLD = 0.01  # 音量閾値
        
        # VADの初期化（中感度）
        self.vad = webrtcvad.Vad(2)
        
        # 音声設定
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # Whisper.cppサーバーの接続確認
        self._check_whisper_server()
        
    def _check_whisper_server(self):
        """Whisper.cppサーバーの接続確認"""
        try:
            response = requests.get(f"{self.whisper_server_url}/", timeout=5)
            logger.info(f"Whisper.cppサーバー応答: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Whisper.cppサーバーに接続しました")
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
        except Exception as e:
            logger.error(f"VADエラー: {e}")
            return False
    
    def _audio_to_wav_bytes(self, audio_data):
        """音声データをWAVバイトストリームに変換"""
        # デバッグ：入力データの確認
        logger.debug(f"入力音声データ: shape={audio_data.shape}, min={np.min(audio_data):.3f}, max={np.max(audio_data):.3f}, mean={np.mean(np.abs(audio_data)):.3f}")
        
        # 無音チェック
        if np.max(np.abs(audio_data)) < 0.001:
            logger.warning("警告: 音声データがほぼ無音です")
        
        # 正規化（-1.0 to 1.0の範囲に）
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        
        # 16bit PCMに変換
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
    
    def save_debug_wav(self, wav_bytes, prefix="audio"):
        """デバッグ用にWAVファイルを保存"""
        if not self.save_debug_wav_enabled:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{prefix}_{timestamp}.wav"
        filepath = os.path.join(self.debug_wav_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(wav_bytes)
        
        logger.info(f"デバッグWAV保存: {filepath}")
        return filepath
    
    async def transcribe(self, audio_data, metadata):
        """Whisper.cppサーバーで音声を転写"""
        try:
            # 音声データをWAV形式に変換
            wav_bytes = self._audio_to_wav_bytes(audio_data)
            
            # デバッグ用にWAVファイルを保存
            wav_path = self.save_debug_wav(wav_bytes, prefix="transcribe")
            
            logger.info(f"転写開始: {metadata['duration']:.1f}秒, WAVサイズ: {len(wav_bytes)} bytes")
            
            # Whisper.cppサーバーにPOSTリクエスト
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
            
            # 日本語用の最適化されたパラメータ
            data = {
                'language': 'ja',               # 日本語を明示的に指定
                'temperature': '0.0',           # 決定的な出力
                'temperature_inc': '0.2',       # 温度増分
                'response_format': 'json',      # JSON形式で応答
                'beam_size': '5',               # ビームサーチ幅
                'best_of': '5',                 # 複数候補から最良を選択
                'no_timestamps': 'false',       # タイムスタンプを含める
                'word_timestamps': 'false',     # 単語レベルのタイムスタンプは不要
                'max_context': '-1',            # コンテキストの最大トークン数（-1で無制限）
                'max_len': '0',                 # セグメントの最大長（0で無制限）
                'split_on_word': 'false',       # 単語境界で分割しない
                'no_speech_threshold': '0.6',   # 無音判定の閾値
                'compression_ratio_threshold': '2.4',
                'logprob_threshold': '-1.0',
                'no_fallback': 'false',         # フォールバックを許可
                'single_segment': 'false',      # 単一セグメントモードを無効化
                'print_special': 'false',
                'print_progress': 'false',
                'print_realtime': 'false',
                'print_timestamps': 'false'
            }
            
            # デバッグ：送信するパラメータをログ出力
            logger.debug(f"Whisperパラメータ: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
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
            logger.debug(f"応答ヘッダー: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                
                # レスポンス詳細をログ出力
                logger.debug(f"Whisper応答JSON: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                # テキストの抽出（複数のセグメントがある場合は結合）
                text = ""
                if 'text' in result:
                    text = result['text'].strip()
                elif 'segments' in result:
                    # セグメントがある場合は結合
                    segments = result.get('segments', [])
                    text = ' '.join([seg.get('text', '').strip() for seg in segments])
                    logger.debug(f"セグメント数: {len(segments)}")
                    for i, seg in enumerate(segments):
                        logger.debug(f"セグメント{i}: {seg}")
                
                # 結果を作成
                full_result = {
                    "text": text,
                    "metadata": metadata,
                    "segments": result.get('segments', []),
                    "raw_response": result
                }
                
                if text:
                    logger.info(f"転写成功: '{text}' ({metadata['duration']:.1f}秒)")
                else:
                    logger.warning(f"転写結果が空です。生の応答: {result}")
                
                return full_result
            else:
                logger.error(f"Whisper.cppサーバーエラー: {response.status_code}")
                logger.error(f"エラー応答: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"転写エラー: {e}")
            import traceback
            logger.error(f"スタックトレース: {traceback.format_exc()}")
            return None

# FastAPIアプリケーション
app = FastAPI()

# 静的ファイルの配信
app.mount("/static", StaticFiles(directory="static"), name="static")

# 音声処理インスタンス
audio_processor = SimpleAudioProcessor(save_debug_wav=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket接続が確立されました")
    
    # シンプルな状態管理
    is_recording = False
    audio_buffer = []
    silence_counter = 0
    speech_frame_count = 0
    total_frame_count = 0
    
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
                total_frame_count += 1
                
                is_speech = audio_processor.is_speech(frame)
                
                # デバッグ：定期的にステータスを出力
                if total_frame_count % 100 == 0:
                    logger.debug(f"フレーム{total_frame_count}: 録音中={is_recording}, 無音カウント={silence_counter}")
                
                if is_speech:
                    speech_frame_count += 1
                    # 音声検出
                    if not is_recording:
                        is_recording = True
                        audio_buffer = []
                        logger.info("=== 音声検出開始 ===")
                        await websocket.send_json({
                            "type": "speech_started", 
                            "timestamp": current_time
                        })
                    
                    # 音声データを追加
                    audio_buffer.extend(audio_data)
                    silence_counter = 0
                    
                else:
                    # 無音検出
                    if is_recording:
                        # 録音中の無音はバッファに追加
                        audio_buffer.extend(audio_data)
                        silence_counter += 1
                        
                        # 無音が閾値を超えたら転写
                        if silence_counter >= audio_processor.SILENCE_THRESHOLD:
                            # 音声データを準備
                            audio_array = np.array(audio_buffer, dtype=np.float32)
                            duration = len(audio_array) / audio_processor.sample_rate
                            
                            logger.info(f"=== 転写処理開始 ===")
                            logger.info(f"音声長: {duration:.1f}秒, 音声フレーム数: {speech_frame_count}")
                            
                            # 最小長チェック
                            if duration >= audio_processor.MIN_AUDIO_LENGTH:
                                await websocket.send_json({
                                    "type": "processing", 
                                    "timestamp": current_time
                                })
                                
                                # メタデータ作成
                                metadata = {
                                    "duration": duration,
                                    "timestamp": current_time,
                                    "speech_frames": speech_frame_count,
                                    "total_frames": total_frame_count
                                }
                                
                                # 転写実行
                                result = await audio_processor.transcribe(audio_array, metadata)
                                
                                if result:
                                    if result["text"]:
                                        logger.info(f"=== 転写成功: '{result['text']}' ===")
                                        await websocket.send_json({
                                            "type": "transcription", 
                                            "result": result, 
                                            "timestamp": current_time
                                        })
                                    else:
                                        logger.warning("=== 転写結果が空 ===")
                                        await websocket.send_json({
                                            "type": "transcription_empty",
                                            "timestamp": current_time
                                        })
                                else:
                                    logger.error("=== 転写エラー ===")
                                    await websocket.send_json({
                                        "type": "transcription_error",
                                        "timestamp": current_time
                                    })
                            else:
                                logger.info(f"音声が短すぎます: {duration:.1f}秒 < {audio_processor.MIN_AUDIO_LENGTH}秒")
                            
                            # 録音終了・リセット
                            is_recording = False
                            audio_buffer = []
                            silence_counter = 0
                            speech_frame_count = 0
                            logger.info("=== 録音終了・待機状態へ ===")
                            await websocket.send_json({
                                "type": "speech_ended", 
                                "timestamp": current_time
                            })
    
    except WebSocketDisconnect:
        logger.info("WebSocket接続が切断されました")
    except Exception as e:
        logger.error(f"WebSocketエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())

@app.get("/")
async def read_root():
    return {
        "message": "シンプル音声認識サーバー（デバッグ版）が稼働中です",
        "whisper_server": audio_processor.whisper_server_url,
        "debug_wav_enabled": audio_processor.save_debug_wav,
        "debug_wav_dir": audio_processor.debug_wav_dir
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_server": audio_processor.whisper_server_url,
        "parameters": {
            "min_audio_length": audio_processor.MIN_AUDIO_LENGTH,
            "silence_threshold": audio_processor.SILENCE_THRESHOLD,
            "volume_threshold": audio_processor.VOLUME_THRESHOLD
        }
    }

@app.get("/debug/list_wav")
async def list_debug_wav():
    """デバッグ用WAVファイルのリストを取得"""
    if not audio_processor.save_debug_wav_enabled:
        return {"error": "Debug WAV saving is disabled"}
    
    wav_files = []
    if os.path.exists(audio_processor.debug_wav_dir):
        for file in sorted(os.listdir(audio_processor.debug_wav_dir)):
            if file.endswith('.wav'):
                filepath = os.path.join(audio_processor.debug_wav_dir, file)
                stat = os.stat(filepath)
                wav_files.append({
                    "filename": file,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
    
    return {"wav_files": wav_files, "count": len(wav_files)}

@app.delete("/debug/clear_wav")
async def clear_debug_wav():
    """デバッグ用WAVファイルをクリア"""
    if not audio_processor.save_debug_wav_enabled:
        return {"error": "Debug WAV saving is disabled"}
    
    count = 0
    if os.path.exists(audio_processor.debug_wav_dir):
        for file in os.listdir(audio_processor.debug_wav_dir):
            if file.endswith('.wav'):
                os.remove(os.path.join(audio_processor.debug_wav_dir, file))
                count += 1
    
    return {"message": f"Cleared {count} WAV files"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)