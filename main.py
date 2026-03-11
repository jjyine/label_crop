import logging
import time
from src.fetch_data import fetch_data
from src.crop_labels import crop_labels
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

def main():
    limit = 1
    total_results = 1

    for i, (wine_id, image, data_url, original_url) in enumerate(fetch_data(limit=limit, total_results=total_results)):
        logging.info(f"API 호출 중: 이미지 {i + 1}/{total_results} 가져오는 중...")

        while True:
            try:
                result = crop_labels(wine_id, image, data_url, original_url, i)

                if result and result.get("s3_url"):
                    logging.info(f"이미지 {i + 1} 크롭 성공: {data_url}")
                else:
                    logging.warning(f"이미지 {i + 1} 크롭/업로드 실패: {data_url}")

                break
            except Exception as e:
                if 'retryDelay' in str(e):
                    wait_time = float(str(e).split('retryDelay')[1].split(':')[1].split(',')[0])
                    logging.warning(f"할당량 초과, {wait_time}초 후에 재시도합니다.")
                    time.sleep(wait_time)
                else:
                    logging.error(f"이미지 {i + 1} 크롭 실패: {e}")
                    break

if __name__ == "__main__":
    main()