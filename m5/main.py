import M5
import sys
import time

# 画像ファイルのパス
IMG_SMILE = "/flash/smile.png"
IMG_CRY = "/flash/cry.png"

def setup():
    M5.begin()
    M5.Lcd.clear()
    M5.Lcd.setFont(M5.Lcd.FONTS.DejaVu18) # 念のためフォントも初期化
    M5.Lcd.drawCenterString("Waiting...", 64, 55)
    print("AtomS3 is ready for image commands. Send 's' or 'c'.")

def loop():
    M5.update()
    command = sys.stdin.readline().strip()

    if command:
        print(f"Received command: '{command}'")

        if command == 's':
            # "smile.png"を画面の左上(座標0,0)から描画します
            M5.Lcd.drawImage(IMG_SMILE, 0, 0)
        elif command == 'c':
            # "cry.png"を画面の左上(座標0,0)から描画します
            M5.Lcd.drawImage(IMG_CRY, 0, 0)
        else:
            # 不明なコマンドの場合は画面をクリアする
            M5.Lcd.clear()
            M5.Lcd.drawCenterString("?", 64, 55)


if __name__ == '__main__':
    try:
        setup()
        while True:
            loop()
    except Exception as e:
        print(f"An error occurred: {e}")