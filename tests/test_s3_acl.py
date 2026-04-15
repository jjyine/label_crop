import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path("/Users/changmook/work/label_crop")


class FakeS3Client:
    def __init__(self):
        self.head_bucket_calls = []
        self.put_object_calls = []

    def head_bucket(self, **kwargs):
        self.head_bucket_calls.append(kwargs)

    def put_object(self, **kwargs):
        self.put_object_calls.append(kwargs)


def load_crop_labels_module(module_name: str):
    module_path = REPO_ROOT / "src" / "crop_labels.py"

    fake_cv2 = types.ModuleType("cv2")
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *args, **kwargs: object()
    fake_pymysql = types.ModuleType("pymysql")
    fake_pymysql.connect = lambda *args, **kwargs: object()
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


def load_fetch_data_module(module_name: str):
    module_path = REPO_ROOT / "src" / "fetch_data.py"

    fake_cv2 = types.ModuleType("cv2")
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_requests = types.ModuleType("requests")
    fake_pymysql = types.ModuleType("pymysql")
    fake_pymysql.connect = lambda *args, **kwargs: object()
    fake_pymysql.MySQLError = RuntimeError
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *args, **kwargs: object()
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
    fake_botocore_exceptions.BotoCoreError = RuntimeError
    fake_botocore_exceptions.ClientError = RuntimeError
    fake_botocore.exceptions = fake_botocore_exceptions
    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil.Image = fake_pil_image
    fake_google = types.ModuleType("google")
    fake_google_genai = types.ModuleType("google.genai")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

    fake_google_genai.Client = FakeClient
    fake_google_genai.types = types.SimpleNamespace()
    fake_google.genai = fake_google_genai

    fake_crop_labels = types.ModuleType("src.crop_labels")
    fake_crop_labels._thread_local = threading.local()

    sys.modules.pop(module_name, None)

    with patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "requests": fake_requests,
            "pymysql": fake_pymysql,
            "torch": fake_torch,
            "dotenv": fake_dotenv,
            "boto3": fake_boto3,
            "botocore": fake_botocore,
            "botocore.exceptions": fake_botocore_exceptions,
            "PIL": fake_pil,
            "PIL.Image": fake_pil_image,
            "google": fake_google,
            "google.genai": fake_google_genai,
            "src.crop_labels": fake_crop_labels,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return module


class CropLabelsS3AclTest(unittest.TestCase):
    def setUp(self):
        self.crop_labels = load_crop_labels_module("crop_labels_s3_acl_test_module")

    def test_upload_bytes_to_s3_uses_public_read_acl(self):
        fake_s3 = FakeS3Client()

        with patch.object(self.crop_labels, "get_s3_client", return_value=fake_s3):
            result = self.crop_labels.upload_bytes_to_s3(b"png-bytes", "labels/sample.png")

        self.assertEqual(
            result,
            f"https://{self.crop_labels.S3_BUCKET_NAME}.s3.{self.crop_labels.AWS_REGION}.amazonaws.com/labels/sample.png",
        )
        self.assertEqual(len(fake_s3.put_object_calls), 1)
        self.assertEqual(fake_s3.put_object_calls[0]["ACL"], "public-read")
        self.assertEqual(fake_s3.put_object_calls[0]["ContentType"], "image/png")

    def test_check_s3_permissions_uses_public_read_acl(self):
        fake_s3 = FakeS3Client()

        with patch.object(self.crop_labels, "get_s3_client", return_value=fake_s3):
            result = self.crop_labels.check_s3_permissions()

        self.assertTrue(result)
        self.assertEqual(len(fake_s3.head_bucket_calls), 1)
        self.assertEqual(len(fake_s3.put_object_calls), 1)
        self.assertEqual(fake_s3.put_object_calls[0]["ACL"], "public-read")
        self.assertEqual(fake_s3.put_object_calls[0]["ContentType"], "text/plain")

    def test_test_s3_upload_uses_public_read_acl(self):
        fake_s3 = FakeS3Client()

        with patch.object(self.crop_labels, "get_s3_client", return_value=fake_s3):
            result = self.crop_labels.test_s3_upload()

        self.assertTrue(result)
        self.assertEqual(len(fake_s3.put_object_calls), 1)
        self.assertEqual(fake_s3.put_object_calls[0]["ACL"], "public-read")
        self.assertEqual(fake_s3.put_object_calls[0]["ContentType"], "text/plain")


class FetchDataS3AclTest(unittest.TestCase):
    def setUp(self):
        self.fetch_data = load_fetch_data_module("fetch_data_s3_acl_test_module")

    def test_upload_bytes_to_s3_uses_public_read_acl(self):
        fake_s3 = FakeS3Client()

        with patch.object(self.fetch_data, "get_s3_client", return_value=fake_s3):
            result = self.fetch_data.upload_bytes_to_s3(b"png-bytes", "labels/sample.png")

        self.assertEqual(
            result,
            f"https://{self.fetch_data.S3_BUCKET_NAME}.s3.{self.fetch_data.AWS_REGION}.amazonaws.com/labels/sample.png",
        )
        self.assertEqual(len(fake_s3.put_object_calls), 1)
        self.assertEqual(fake_s3.put_object_calls[0]["ACL"], "public-read")
        self.assertEqual(fake_s3.put_object_calls[0]["ContentType"], "image/png")

    def test_check_s3_permissions_uses_public_read_acl(self):
        fake_s3 = FakeS3Client()

        with patch.object(self.fetch_data, "get_s3_client", return_value=fake_s3):
            result = self.fetch_data.check_s3_permissions()

        self.assertTrue(result)
        self.assertEqual(len(fake_s3.head_bucket_calls), 1)
        self.assertEqual(len(fake_s3.put_object_calls), 1)
        self.assertEqual(fake_s3.put_object_calls[0]["ACL"], "public-read")
        self.assertEqual(fake_s3.put_object_calls[0]["ContentType"], "text/plain")


if __name__ == "__main__":
    unittest.main()
