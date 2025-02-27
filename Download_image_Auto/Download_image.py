import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

# ฟังก์ชันสำหรับดาวน์โหลดรูป
def download_images(fruit_name):
    # สร้างโฟลเดอร์สำหรับเก็บรูป
    os.makedirs(fruit_name, exist_ok=True)

    # เปิด Chrome
    driver = webdriver.Chrome()
    driver.get(f"https://www.google.com/search?q={fruit_name}&tbm=isch")

    # เลื่อนหน้าเพื่อโหลดรูปเพิ่ม
    for _ in range(5):
        driver.execute_script("window.scrollBy(0,1000);")
        time.sleep(2)

    # ดึง URL รูป
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img")
    image_urls = [img.get_attribute("src") for img in image_elements[:100] if img.get_attribute("src")]

    # ปิดเบราว์เซอร์
    driver.quit()

    # ดาวน์โหลดและบันทึกรูปภาพ
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(f"{fruit_name}/{fruit_name}_{i+1}.jpg", "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"✅ ดาวน์โหลดรูป {i+1} ของ {fruit_name} สำเร็จ")
            else:
                print(f"❌ รูป {i+1} ของ {fruit_name} ดาวน์โหลดไม่สำเร็จ (HTTP {response.status_code})")
        except Exception as e:
            print(f"⚠️ รูป {i+1} ของ {fruit_name} มีข้อผิดพลาด: {e}")

    print(f"🎉 ดาวน์โหลดครบ 100 รูปของ {fruit_name} แล้ว!")

# เรียกใช้ฟังก์ชันสำหรับแต่ละผลไม้
fruits = ["banana", "mango", "pineapple", "watermelon"]
for fruit in fruits:
    download_images(fruit)
