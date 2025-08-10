import serial
import time

# --- 設定項目 ---
# AtomS3が接続されているCOMポート名を指定します。
# ★これは環境によって変わるので、必ず確認・変更してください★
# Windowsなら 'COM3', 'COM4'など
# Macなら '/dev/tty.usbserial-xxxxxxxx'など
SERIAL_PORT = 'COM3' 
#SERIAL_PORT = 'COM4' 

BAUD_RATE = 115200 # AtomS3の標準的な通信速度

# --- メイン処理 ---
try:
    # シリアルポートを開く
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"接続成功: {SERIAL_PORT}")
    time.sleep(2) # 接続が安定するまで少し待つ

    # ユーザーに入力を促し続ける
    while True:
        command = input("コマンドを入力 ('s' for smile, 'c' for cry, 'q' to quit): ")

        if command == 'q':
            print("プログラムを終了します。")
            break
        
        if command in ['s', 'c']:
            # AtomS3にコマンドを送信します。
            # b''はバイト形式に変換、'\n'はAtomS3側のreadline()が認識するための改行コードです。
            print(f"'{command}' を送信中...")
            ser.write(f'{command}\n'.encode('utf-8'))
        else:
            print("無効なコマンドです。's', 'c', 'q'のいずれかを入力してください。")

except serial.SerialException as e:
    print(f"エラー: ポート '{SERIAL_PORT}' が見つからないか、開けません。")
    print("正しいCOMポート名を確認してください。")
    print(f"詳細: {e}")

finally:
    # プログラム終了時に必ずポートを閉じる
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("接続を閉じました。")