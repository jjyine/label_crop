import heapq
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import count
from queue import Empty, Queue
from typing import Optional


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MIN_REVIEW_COUNT = 5000
MAX_REVIEW_COUNT = 10000
RANGE_STEP = 100
MAX_WORKERS = 10
FETCH_LIMIT = 100
DEFAULT_START_ID = 400
MAX_TASK_RETRIES = 3
RETRY_BACKOFF_SECONDS = (2, 4, 8)

PROCESSOR_STOP = object()


@dataclass(frozen=True)
class RangeTask:
    min_count: int
    max_count: int
    start_id: int = DEFAULT_START_ID
    attempt: int = 1
    fetched: int = 0
    success: int = 0
    fail: int = 0

    def with_updates(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class ImageJob:
    wine_id: int
    image: object
    data_url: str
    original_url: str
    range_key: tuple[int, int]
    item_index: int


@dataclass
class RangeProgress:
    min_count: int
    max_count: int
    start_id: int = DEFAULT_START_ID
    attempt: int = 1
    fetched: int = 0
    success: int = 0
    fail: int = 0
    exhausted: bool = False
    in_flight: int = 0
    error: Optional[str] = None
    worker: str = ""

    def to_task(
        self,
        *,
        start_id: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> RangeTask:
        return RangeTask(
            min_count=self.min_count,
            max_count=self.max_count,
            start_id=self.start_id if start_id is None else start_id,
            attempt=self.attempt if attempt is None else attempt,
            fetched=self.fetched,
            success=self.success,
            fail=self.fail,
        )


class TaskQueue:
    def __init__(self, tasks=None):
        self._condition = threading.Condition()
        self._heap = []
        self._sequence = count()
        self._unfinished = 0

        for task in tasks or []:
            self.add_task(task)

    def _push_task(self, task: RangeTask, ready_at: float):
        heapq.heappush(self._heap, (ready_at, next(self._sequence), task))

    def add_task(self, task: RangeTask, delay_seconds: float = 0.0):
        with self._condition:
            self._unfinished += 1
            self._push_task(task, time.monotonic() + delay_seconds)
            self._condition.notify()

    def reschedule(self, task: RangeTask, delay_seconds: float = 0.0):
        with self._condition:
            self._push_task(task, time.monotonic() + delay_seconds)
            self._condition.notify()

    def get(self) -> Optional[RangeTask]:
        with self._condition:
            while True:
                if self._unfinished == 0:
                    return None

                if not self._heap:
                    self._condition.wait()
                    continue

                ready_at, _, task = self._heap[0]
                wait_time = ready_at - time.monotonic()
                if wait_time <= 0:
                    heapq.heappop(self._heap)
                    return task

                self._condition.wait(timeout=wait_time)

    def complete_task(self):
        with self._condition:
            if self._unfinished == 0:
                return

            self._unfinished -= 1
            self._condition.notify_all()


class RangeCoordinator:
    def __init__(self, task_queue: TaskQueue, result_queue: Queue):
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._lock = threading.Lock()
        self._progress_by_key: dict[tuple[int, int], RangeProgress] = {}

    def start_task(self, task: RangeTask, worker_name: str) -> tuple[int, int]:
        key = (task.min_count, task.max_count)

        with self._lock:
            progress = self._progress_by_key.get(key)
            if progress is None:
                progress = RangeProgress(
                    min_count=task.min_count,
                    max_count=task.max_count,
                    start_id=task.start_id,
                    attempt=task.attempt,
                    fetched=task.fetched,
                    success=task.success,
                    fail=task.fail,
                    worker=worker_name,
                )
                self._progress_by_key[key] = progress
            else:
                progress.start_id = task.start_id
                progress.attempt = task.attempt
                progress.worker = worker_name
                progress.error = None
                progress.exhausted = False
                progress.fetched = max(progress.fetched, task.fetched)
                progress.success = max(progress.success, task.success)
                progress.fail = max(progress.fail, task.fail)

        return key

    def record_fetched(self, key: tuple[int, int]) -> int:
        with self._lock:
            progress = self._progress_by_key[key]
            item_index = progress.fetched
            progress.fetched += 1
            progress.in_flight += 1
            return item_index

    def snapshot_task(self, key: tuple[int, int], *, start_id: Optional[int] = None) -> RangeTask:
        with self._lock:
            progress = self._progress_by_key[key]
            return progress.to_task(start_id=start_id)

    def record_processed(self, key: tuple[int, int], ok: bool):
        with self._lock:
            progress = self._progress_by_key.get(key)
            if progress is None:
                return None

            if ok:
                progress.success += 1
            else:
                progress.fail += 1

            if progress.in_flight > 0:
                progress.in_flight -= 1

            result = self._finalize_locked(key)

        if result is not None:
            self._emit_result(result)

        return result

    def mark_exhausted(self, key: tuple[int, int], worker_name: str):
        with self._lock:
            progress = self._progress_by_key[key]
            progress.exhausted = True
            progress.worker = worker_name
            result = self._finalize_locked(key)

        if result is not None:
            self._emit_result(result)

        return result

    def mark_failed(self, key: tuple[int, int], worker_name: str, error: str):
        with self._lock:
            progress = self._progress_by_key[key]
            progress.error = error
            progress.worker = worker_name
            result = self._finalize_locked(key)

        if result is not None:
            self._emit_result(result)

        return result

    def _finalize_locked(self, key: tuple[int, int]):
        progress = self._progress_by_key.get(key)
        if progress is None or progress.in_flight > 0:
            return None

        if progress.error is not None:
            result = build_result(
                progress.worker,
                progress.to_task(),
                aborted=True,
                error=progress.error,
            )
        elif progress.exhausted:
            result = build_result(progress.worker, progress.to_task(), aborted=False)
        else:
            return None

        del self._progress_by_key[key]
        return result

    def _emit_result(self, result):
        self._result_queue.put(result)
        log_result(result)
        self._task_queue.complete_task()


def get_fetch_tools():
    from src.fetch_data import FetchRangeError, fetch_data

    return fetch_data, FetchRangeError


def get_crop_tools():
    from src.crop_labels import _thread_local, crop_labels

    return crop_labels, _thread_local


def build_tasks() -> list[RangeTask]:
    tasks = []

    for max_count in range(MAX_REVIEW_COUNT, MIN_REVIEW_COUNT - 1, -RANGE_STEP):
        min_count = max(MIN_REVIEW_COUNT, max_count - RANGE_STEP + 1)
        tasks.append(RangeTask(min_count=min_count, max_count=max_count))

    return tasks


def format_task(task: RangeTask) -> str:
    return (
        f"range=({task.min_count}, {task.max_count}), "
        f"attempt={task.attempt}, start_id={task.start_id}"
    )


def get_fetch_worker_count(max_workers: int) -> int:
    return min(2, max_workers)


def get_job_queue_maxsize(max_workers: int) -> int:
    return max_workers * 4


def get_worker_logger(worker_name: str) -> logging.Logger:
    logger = logging.getLogger(worker_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = os.path.join(LOG_DIR, f"{worker_name}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        logger.addHandler(file_handler)
        logger.propagate = True

    return logger


def build_result(worker_name: str, task: RangeTask, aborted: bool, error: Optional[str] = None):
    return {
        "worker": worker_name,
        "success": task.success,
        "fail": task.fail,
        "fetched": task.fetched,
        "min_count": task.min_count,
        "max_count": task.max_count,
        "attempt": task.attempt,
        "aborted": aborted,
        "error": error,
    }


def process_one(worker_name, logger, i, wine_id, image, data_url, original_url):
    crop_labels, _ = get_crop_tools()

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


def log_result(result):
    if result["aborted"]:
        logging.error(
            f"[실패] {result['worker']} "
            f"(range={result['min_count']}~{result['max_count']}, attempt={result['attempt']}) "
            f"조회={result['fetched']}, 성공={result['success']}, 실패={result['fail']}, "
            f"error={result['error']}"
        )
    else:
        logging.info(
            f"[완료] {result['worker']} "
            f"(range={result['min_count']}~{result['max_count']}, attempt={result['attempt']}) "
            f"조회={result['fetched']}, 성공={result['success']}, 실패={result['fail']}"
        )


def fetcher_loop(
    fetcher_index: int,
    task_queue: TaskQueue,
    job_queue: Queue,
    coordinator: RangeCoordinator,
    limit: int,
):
    worker_name = f"fetcher_{fetcher_index:02d}"
    logger = get_worker_logger(worker_name)
    _, thread_local = get_crop_tools()
    fetch_data, FetchRangeError = get_fetch_tools()

    thread_local.logger = logger
    logger.info("fetcher 시작")

    while True:
        task = task_queue.get()
        if task is None:
            logger.info("fetcher 종료 - 대기 task 없음")
            return

        key = coordinator.start_task(task, worker_name)
        logger.info(f"task 시작 - {format_task(task)}, limit={limit}")

        try:
            for wine_id, image, data_url, original_url in fetch_data(
                limit=limit,
                start_id=task.start_id,
                min_count=task.min_count,
                max_count=task.max_count,
            ):
                item_index = coordinator.record_fetched(key)
                snapshot = coordinator.snapshot_task(key)
                logger.info(
                    "조회 완료: "
                    f"idx={item_index + 1}, wine_id={wine_id}, url={data_url}, "
                    f"range=({snapshot.min_count}, {snapshot.max_count}), "
                    f"attempt={snapshot.attempt}"
                )
                job_queue.put(
                    ImageJob(
                        wine_id=wine_id,
                        image=image,
                        data_url=data_url,
                        original_url=original_url,
                        range_key=key,
                        item_index=item_index,
                    )
                )
        except FetchRangeError as e:
            retry_task = coordinator.snapshot_task(key, start_id=e.last_id)
            logger.warning(
                "task 재시도 대상 - "
                f"{format_task(retry_task)}, fetched={retry_task.fetched}, "
                f"success={retry_task.success}, fail={retry_task.fail}, error={e}"
            )

            if retry_task.attempt <= MAX_TASK_RETRIES:
                delay_index = min(retry_task.attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                delay_seconds = RETRY_BACKOFF_SECONDS[delay_index]
                next_task = retry_task.with_updates(attempt=retry_task.attempt + 1)
                logger.warning(
                    "task 재큐잉 - "
                    f"range=({next_task.min_count}, {next_task.max_count}), "
                    f"next_attempt={next_task.attempt}, next_start_id={next_task.start_id}, "
                    f"delay={delay_seconds}s"
                )
                task_queue.reschedule(next_task, delay_seconds=delay_seconds)
                continue

            coordinator.mark_failed(key, worker_name, str(e))
            continue
        except Exception as e:
            snapshot = coordinator.snapshot_task(key)
            logger.exception(
                "task 실패 - "
                f"{format_task(snapshot)}, fetched={snapshot.fetched}, "
                f"success={snapshot.success}, fail={snapshot.fail}, error={e}"
            )
            coordinator.mark_failed(key, worker_name, str(e))
            continue

        coordinator.mark_exhausted(key, worker_name)


def processor_loop(worker_index: int, job_queue: Queue, coordinator: RangeCoordinator):
    worker_name = f"worker_{worker_index:02d}"
    logger = get_worker_logger(worker_name)
    _, thread_local = get_crop_tools()

    thread_local.logger = logger
    logger.info("worker 시작")

    while True:
        job = job_queue.get()
        if job is PROCESSOR_STOP:
            job_queue.task_done()
            logger.info("worker 종료 - 대기 job 없음")
            return

        ok = process_one(
            worker_name,
            logger,
            job.item_index,
            job.wine_id,
            job.image,
            job.data_url,
            job.original_url,
        )
        coordinator.record_processed(job.range_key, ok)
        job_queue.task_done()


def run_workers(tasks: list[RangeTask], limit: int = FETCH_LIMIT, max_workers: int = MAX_WORKERS):
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    task_queue = TaskQueue(tasks)
    result_queue = Queue()
    coordinator = RangeCoordinator(task_queue, result_queue)
    fetch_workers = get_fetch_worker_count(max_workers)
    job_queue = Queue(maxsize=get_job_queue_maxsize(max_workers))

    with ThreadPoolExecutor(max_workers=max_workers + fetch_workers) as executor:
        fetcher_futures = [
            executor.submit(fetcher_loop, fetcher_index + 1, task_queue, job_queue, coordinator, limit)
            for fetcher_index in range(fetch_workers)
        ]
        processor_futures = [
            executor.submit(processor_loop, worker_index + 1, job_queue, coordinator)
            for worker_index in range(max_workers)
        ]

        for future in fetcher_futures:
            future.result()

        for _ in range(max_workers):
            job_queue.put(PROCESSOR_STOP)

        for future in processor_futures:
            future.result()

    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except Empty:
            break

    return results


def log_summary(results):
    total_success = sum(result["success"] for result in results)
    total_fail = sum(result["fail"] for result in results)
    total_fetched = sum(result["fetched"] for result in results)
    total_aborted = sum(1 for result in results if result["aborted"])

    if total_aborted:
        logging.error(
            f"[전체 완료] 총 조회={total_fetched}, 총 성공={total_success}, 총 실패={total_fail}, "
            f"중단 task={total_aborted}"
        )
    else:
        logging.info(
            f"[전체 완료] 총 조회={total_fetched}, 총 성공={total_success}, 총 실패={total_fail}"
        )


def main():
    logging.info("=== 병렬 작업 시작 ===")
    results = run_workers(build_tasks(), limit=FETCH_LIMIT, max_workers=MAX_WORKERS)
    log_summary(results)


if __name__ == "__main__":
    main()
