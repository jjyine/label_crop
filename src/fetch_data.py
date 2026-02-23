# fetch_data.py
import os
from dotenv import load_dotenv
import pymysql
import re
import requests
import json
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# .env 파일 로드
load_dotenv()

# 환경 변수에서 데이터베이스 정보 가져오기
db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

def select_largest_vivino_image(image_data):
    selected_image = None
    max_area = -1
    selected_key = None

    for image_url in image_data.keys():
        if "vivino.com" in image_url:
            match = re.search(r"(\d+)x(\d+)", image_url)
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
                area = width * height
            else:
                match = re.search(r"x(\d+)", image_url)
                if match:
                    size = int(match.group(1))
                    area = size * size
                else:
                    area = 0

            # ✅ 가장 큰 이미지 선택
            if area > max_area:
                max_area = area
                selected_image = image_url
                selected_key = image_data[image_url]

    return selected_image, selected_key


def fetch_data(limit=5, total_results=5):
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=3306
        )

        offset = 0
        fetched_results = 0

        while fetched_results < total_results:
            with connection.cursor() as cursor:
                sql = f"SELECT s3_images FROM wine LIMIT {limit} OFFSET {offset};"
                cursor.execute(sql)
                results = cursor.fetchall()

                if not results:
                    print("No more images to fetch.")
                    break

                for row in results:
                    image_data = json.loads(row[0])
                    largest_vivino_image, selected_key = select_largest_vivino_image(image_data)

                    if largest_vivino_image and selected_key:
                        final_image_url = f"https://vin-social.s3.amazonaws.com/{selected_key}"
                        print("Fetching image from URL:", final_image_url)

                        response = requests.get(final_image_url)
                        response.raise_for_status()

                        # 항상 RGB로 강제 변환 (CMYK/P/L/RGBA 방지)
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                        fetched_results += 1
                        yield img_cv, final_image_url

                offset += limit

    except pymysql.MySQLError as e:
        print("DB connect failed:", e)
    finally:
        if connection:
            connection.close()
