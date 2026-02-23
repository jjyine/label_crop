# main.py
from src.fetch_data import fetch_data
from src.crop_labels import crop_labels

def main():
    limit = 5  # 한 번에 가져올 이미지 개수
    total_results = 5  # 총 가져올 이미지 수

    for i, (image, data_url) in enumerate(fetch_data(limit=limit, total_results=total_results)):
        # 라벨 크롭 수행
        crop_labels(image, data_url, i)  # 인덱스 전달

if __name__ == "__main__":
    main()