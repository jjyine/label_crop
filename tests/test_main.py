import logging
import os
import tempfile
import threading
import time
import unittest
import importlib.util
import sys
import types
from unittest.mock import patch

import main


class FakeFetchRangeError(RuntimeError):
    def __init__(self, last_id: int, min_count=None, max_count=None):
        self.last_id = last_id
        self.min_count = min_count
        self.max_count = max_count
        super().__init__(
            "fetch_data DB query failed after retries "
            f"(last_id={last_id}, range=({min_count}, {max_count}))"
        )


class FakeCursor:
    def __init__(self, executed):
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self._executed.append((sql, params))


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.ping_calls = 0
        self.commit_calls = 0
        self.closed = 0

    def ping(self, reconnect=True):
        self.ping_calls += 1

    def cursor(self):
        return FakeCursor(self.executed)

    def commit(self):
        self.commit_calls += 1

    def close(self):
        self.closed += 1


class MainSchedulingTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._original_log_dir = main.LOG_DIR
        main.LOG_DIR = self._temp_dir.name
        os.makedirs(main.LOG_DIR, exist_ok=True)
        self._reset_loggers()

    def tearDown(self):
        self._reset_loggers()
        main.LOG_DIR = self._original_log_dir
        self._temp_dir.cleanup()

    def _reset_loggers(self):
        for name, logger in list(logging.root.manager.loggerDict.items()):
            if not isinstance(logger, logging.Logger):
                continue
            if not (name.startswith("worker_") or name.startswith("fetcher_")):
                continue
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def test_build_tasks_descending_by_review_bucket(self):
        tasks = main.build_tasks()

        self.assertEqual(len(tasks), 15)
        self.assertEqual((tasks[0].min_count, tasks[0].max_count), (190, 199))
        self.assertEqual((tasks[-1].min_count, tasks[-1].max_count), (50, 59))
        self.assertEqual(
            [(task.min_count, task.max_count) for task in tasks[:3]],
            [(190, 199), (180, 189), (170, 179)],
        )

    def test_pipeline_uses_more_than_two_processors_when_two_ranges_remain(self):
        thread_local = threading.local()
        processed_by = set()
        processed_lock = threading.Lock()

        def fake_crop(wine_id, image, data_url, original_url, index):
            worker_name = thread_local.logger.name
            with processed_lock:
                processed_by.add(worker_name)
            time.sleep(0.03)
            return {"s3_url": f"mock://{wine_id}"}

        def fake_fetch_data(limit, start_id, min_count, max_count):
            for idx in range(6):
                yield (
                    min_count * 100 + idx,
                    object(),
                    f"https://example.com/{min_count}-{idx}.jpg",
                    f"https://images.example.com/{min_count}-{idx}.jpg",
                )

        tasks = [
            main.RangeTask(min_count=50, max_count=59),
            main.RangeTask(min_count=60, max_count=69),
        ]

        with patch.object(main, "get_fetch_tools", return_value=(fake_fetch_data, FakeFetchRangeError)), patch.object(
            main, "get_crop_tools", return_value=(fake_crop, thread_local)
        ):
            results = main.run_workers(tasks, limit=10, max_workers=6)

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(len(processed_by), 3)

    def test_range_result_tracks_fetched_success_and_fail_counts(self):
        thread_local = threading.local()

        def fake_crop(wine_id, image, data_url, original_url, index):
            if wine_id == 2:
                raise RuntimeError("crop failed")
            return {"s3_url": f"mock://{wine_id}"}

        def fake_fetch_data(limit, start_id, min_count, max_count):
            for wine_id in (1, 2, 3):
                yield (
                    wine_id,
                    object(),
                    f"https://example.com/{wine_id}.jpg",
                    f"https://images.example.com/{wine_id}.jpg",
                )

        with patch.object(main, "get_fetch_tools", return_value=(fake_fetch_data, FakeFetchRangeError)), patch.object(
            main, "get_crop_tools", return_value=(fake_crop, thread_local)
        ):
            results = main.run_workers([main.RangeTask(min_count=50, max_count=59)], limit=10, max_workers=2)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(result["aborted"])
        self.assertEqual(result["fetched"], 3)
        self.assertEqual(result["success"], 2)
        self.assertEqual(result["fail"], 1)

    def test_retry_preserves_enqueued_jobs_and_completes_range(self):
        thread_local = threading.local()
        start_ids = []

        def fake_crop(wine_id, image, data_url, original_url, index):
            return {"s3_url": f"mock://{wine_id}"}

        def fake_fetch_data(limit, start_id, min_count, max_count):
            start_ids.append(start_id)

            if start_id == 400:
                for wine_id in (101, 102):
                    yield (
                        wine_id,
                        object(),
                        f"https://example.com/{wine_id}.jpg",
                        f"https://images.example.com/{wine_id}.jpg",
                    )
                raise FakeFetchRangeError(last_id=999, min_count=min_count, max_count=max_count)

            yield (
                103,
                object(),
                "https://example.com/103.jpg",
                "https://images.example.com/103.jpg",
            )

        with patch.object(main, "get_fetch_tools", return_value=(fake_fetch_data, FakeFetchRangeError)), patch.object(
            main, "get_crop_tools", return_value=(fake_crop, thread_local)
        ), patch.object(main, "MAX_TASK_RETRIES", 1), patch.object(main, "RETRY_BACKOFF_SECONDS", (0.01,)):
            results = main.run_workers([main.RangeTask(min_count=190, max_count=199)], limit=10, max_workers=2)

        self.assertEqual(start_ids, [400, 999])
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(result["aborted"])
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(result["fetched"], 3)
        self.assertEqual(result["success"], 3)
        self.assertEqual(result["fail"], 0)

    def test_task_becomes_permanent_failure_after_retry_budget(self):
        thread_local = threading.local()
        start_ids = []

        def fake_crop(wine_id, image, data_url, original_url, index):
            return {"s3_url": f"mock://{wine_id}"}

        def fake_fetch_data(limit, start_id, min_count, max_count):
            start_ids.append(start_id)

            if start_id == 400:
                for wine_id in (201, 202):
                    yield (
                        wine_id,
                        object(),
                        f"https://example.com/{wine_id}.jpg",
                        f"https://images.example.com/{wine_id}.jpg",
                    )
                raise FakeFetchRangeError(last_id=410, min_count=min_count, max_count=max_count)

            for wine_id in (203, 204):
                yield (
                    wine_id,
                    object(),
                    f"https://example.com/{wine_id}.jpg",
                    f"https://images.example.com/{wine_id}.jpg",
                )
            raise FakeFetchRangeError(last_id=420, min_count=min_count, max_count=max_count)

        with patch.object(main, "get_fetch_tools", return_value=(fake_fetch_data, FakeFetchRangeError)), patch.object(
            main, "get_crop_tools", return_value=(fake_crop, thread_local)
        ), patch.object(main, "MAX_TASK_RETRIES", 1), patch.object(main, "RETRY_BACKOFF_SECONDS", (0.01,)):
            results = main.run_workers([main.RangeTask(min_count=190, max_count=199)], limit=10, max_workers=2)

        self.assertEqual(start_ids, [400, 410])
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result["aborted"])
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(result["fetched"], 4)
        self.assertEqual(result["success"], 4)
        self.assertEqual(result["fail"], 0)


class CropLabelCachingTest(unittest.TestCase):
    def load_crop_labels_module(self):
        module_name = "crop_labels_test_module"
        module_path = "/Users/changmook/work/label_crop/src/crop_labels.py"

        fake_cv2 = types.ModuleType("cv2")
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.ndarray = object
        fake_boto3 = types.ModuleType("boto3")
        fake_boto3.client = lambda *args, **kwargs: object()
        fake_pymysql = types.ModuleType("pymysql")
        fake_pymysql.connect = lambda *args, **kwargs: FakeConnection()
        fake_pymysql.MySQLError = RuntimeError
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_botocore = types.ModuleType("botocore")
        fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
        fake_botocore_exceptions.BotoCoreError = RuntimeError
        fake_botocore_exceptions.ClientError = RuntimeError
        fake_botocore.exceptions = fake_botocore_exceptions
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        fake_google = types.ModuleType("google")
        fake_google_genai = types.ModuleType("google.genai")

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

        fake_google_genai.Client = FakeClient
        fake_google_genai.types = types.SimpleNamespace()
        fake_google.genai = fake_google_genai

        sys.modules.pop(module_name, None)

        with patch.dict(
            sys.modules,
            {
                "cv2": fake_cv2,
                "numpy": fake_numpy,
                "boto3": fake_boto3,
                "pymysql": fake_pymysql,
                "torch": fake_torch,
                "botocore": fake_botocore,
                "botocore.exceptions": fake_botocore_exceptions,
                "dotenv": fake_dotenv,
                "google": fake_google,
                "google.genai": fake_google_genai,
            },
        ):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

        return module

    def test_s3_client_and_db_connection_are_reused_per_thread(self):
        crop_labels = self.load_crop_labels_module()
        created_s3_clients = []
        created_connections = []

        def fake_s3_client(*args, **kwargs):
            client = object()
            created_s3_clients.append(client)
            return client

        def fake_connect(*args, **kwargs):
            connection = FakeConnection()
            created_connections.append(connection)
            return connection

        def run_twice_and_reset():
            crop_labels.get_s3_client()
            crop_labels.get_s3_client()
            crop_labels.update_winelabel_crop(1, '{"a":"b"}')
            crop_labels.update_winelabel_crop(2, '{"c":"d"}')
            crop_labels.reset_thread_local_resources()

        with patch.object(crop_labels.boto3, "client", side_effect=fake_s3_client), patch.object(
            crop_labels.pymysql, "connect", side_effect=fake_connect
        ):
            run_twice_and_reset()

            worker = threading.Thread(target=run_twice_and_reset)
            worker.start()
            worker.join()

        self.assertEqual(len(created_s3_clients), 2)
        self.assertEqual(len(created_connections), 2)
        self.assertEqual(created_connections[0].commit_calls, 2)
        self.assertEqual(created_connections[1].commit_calls, 2)
        self.assertGreaterEqual(created_connections[0].ping_calls, 1)
        self.assertGreaterEqual(created_connections[1].ping_calls, 1)


if __name__ == "__main__":
    unittest.main()
