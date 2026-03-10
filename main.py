import logging
import time
from src.fetch_data import fetch_data
from src.crop_labels import crop_labels
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

def main():
    limit = 1  # 한 번에 가져올 이미지 개수
    total_results = 120  # 총 가져올 이미지 수

    for i, (image, data_url) in enumerate(fetch_data(limit=limit, total_results=total_results)):
        logging.info(f"API 호출 중: 이미지 {i + 1}/{total_results} 가져오는 중...")

        while True:
            try:
                # 라벨 크롭 수행
                crop_labels(image, data_url, i)  # 인덱스 전달
                logging.info(f"이미지 {i + 1} 크롭 성공: {data_url}")
                break  # 성공 시 루프 종료
            except Exception as e:
                # 오류 메시지에서 대기 시간 추출
                if 'retryDelay' in str(e):
                    # 대기 시간 추출 (예: 32.050838237s)
                    wait_time = float(str(e).split('retryDelay')[1].split(':')[1].split(',')[0])
                    logging.warning(f"할당량 초과, {wait_time}초 후에 재시도합니다.")
                    time.sleep(wait_time)  # 대기 후 재시도
                else:
                    logging.error(f"이미지 {i + 1} 크롭 실패: {e}")
                    break  # 오류가 다른 경우 루프 종료

if __name__ == "__main__":
    main()