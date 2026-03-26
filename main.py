import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.fetch_data import fetch_data
from src.crop_labels import crop_labels, _thread_local

# 필요 없으면 지워도 됨
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 병렬 워커 구간
# 리뷰 50개 이상 ~ 200개 미만 우선
RANGES = [
    ("worker_50_99", 50, 99),
    ("worker_100_149", 100, 149),
    ("worker_150_199", 150, 199),
]


def get_worker_logger(worker_name: str) -> logging.Logger:
    logger = logging.getLogger(worker_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = os.path.join(LOG_DIR, f"{worker_name}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = True  # 콘솔에도 같이 출력

    return logger


def process_one(worker_name, logger, i, wine_id, image, data_url, original_url):
    while True:
        try:
            logger.info(f"처리 시작: idx={i + 1}, wine_id={wine_id}, url={data_url}")

            result = crop_labels(wine_id, image, data_url, original_url, i)

            if result and result.get("s3_url"):
                logger.info(f"이미지 {i + 1} 크롭 성공: wine_id={wine_id}, url={data_url}")
            else:
                logger.warning(f"이미지 {i + 1} 크롭/업로드 실패: wine_id={wine_id}, url={data_url}")

            return True

        except Exception as e:
            error_text = str(e)

            if "retryDelay" in error_text:
                try:
                    wait_time = float(
                        error_text.split("retryDelay")[1].split(":")[1].split(",")[0]
                    )
                except Exception:
                    wait_time = 10.0

                logger.warning(f"할당량 초과, {wait_time}초 후 재시도합니다. wine_id={wine_id}")
                time.sleep(wait_time)
            else:
                logger.exception(f"이미지 {i + 1} 크롭 실패: wine_id={wine_id}, error={e}")
                return False


def worker(worker_name, min_count, max_count, limit):
    logger = get_worker_logger(worker_name)

    # crop_labels.py 내부 logger 연결
    _thread_local.logger = logger

    logger.info(f"시작 - range=({min_count}, {max_count}), limit={limit}")

    success_count = 0
    fail_count = 0
    fetched_count = 0

    try:
        for i, (wine_id, image, data_url, original_url) in enumerate(
            fetch_data(
                limit=limit,
                min_count=min_count,
                max_count=max_count
            )
        ):
            fetched_count += 1
            logger.info(f"조회 완료: idx={i + 1}, wine_id={wine_id}, url={data_url}")

            ok = process_one(worker_name, logger, i, wine_id, image, data_url, original_url)

            if ok:
                success_count += 1
            else:
                fail_count += 1
    except Exception as e:
        logger.exception(
            f"조회 루프 실패 - range=({min_count}, {max_count}), fetched={fetched_count}, error={e}"
        )
        return {
            "worker": worker_name,
            "success": success_count,
            "fail": fail_count,
            "fetched": fetched_count,
            "min_count": min_count,
            "max_count": max_count,
            "aborted": True,
            "error": str(e),
        }

    logger.info(
        f"종료 - 조회={fetched_count}, 성공={success_count}, 실패={fail_count}, range=({min_count}, {max_count})"
    )

    return {
        "worker": worker_name,
        "success": success_count,
        "fail": fail_count,
        "fetched": fetched_count,
        "min_count": min_count,
        "max_count": max_count,
        "aborted": False,
        "error": None,
    }


def main():
    limit = 100

    logging.info("=== 병렬 작업 시작 ===")

    with ThreadPoolExecutor(max_workers=len(RANGES)) as executor:
        futures = [
            executor.submit(worker, worker_name, min_count, max_count, limit)
            for worker_name, min_count, max_count in RANGES
        ]

        total_success = 0
        total_fail = 0
        total_fetched = 0
        total_aborted = 0

        for future in as_completed(futures):
            result = future.result()
            total_success += result["success"]
            total_fail += result["fail"]
            total_fetched += result["fetched"]

            if result["aborted"]:
                total_aborted += 1
                logging.error(
                    f"[실패] {result['worker']} "
                    f"(range={result['min_count']}~{result['max_count']}) "
                    f"조회={result['fetched']}, 성공={result['success']}, 실패={result['fail']}, "
                    f"error={result['error']}"
                )
            else:
                logging.info(
                    f"[완료] {result['worker']} "
                    f"(range={result['min_count']}~{result['max_count']}) "
                    f"조회={result['fetched']}, 성공={result['success']}, 실패={result['fail']}"
                )

        if total_aborted:
            logging.error(
                f"[전체 완료] 총 조회={total_fetched}, 총 성공={total_success}, 총 실패={total_fail}, "
                f"중단 워커={total_aborted}"
            )
        else:
            logging.info(
                f"[전체 완료] 총 조회={total_fetched}, 총 성공={total_success}, 총 실패={total_fail}"
            )


if __name__ == "__main__":
    main()
