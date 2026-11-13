import io
import logging
import mimetypes
import os
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import httpx
from app.config import settings

logger = logging.getLogger("NyayaSathi.Supabase")


class SupabaseStorageService:
    """
    Manages direct REST interactions with Supabase Storage API.
    Uses SUPABASE_URL, SUPABASE_SECRET_KEY, and SUPABASE_BUCKET.
    Falls back gracefully when Supabase credentials are not configured.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        timeout: float = 30.0,
    ):
        raw_url = supabase_url if supabase_url is not None else settings.SUPABASE_URL
        self.supabase_url = self._clean_supabase_url(raw_url)
        self.secret_key = (secret_key if secret_key is not None else settings.SUPABASE_SECRET_KEY or "").strip()
        raw_bucket = bucket_name if bucket_name is not None else settings.SUPABASE_BUCKET
        self.bucket = (raw_bucket or "nyayasathi-documents").strip()
        self.timeout = timeout
        self._is_configured_override: Optional[bool] = None

    @staticmethod
    def _clean_supabase_url(url: Optional[str]) -> str:
        """
        Sanitize and normalize SUPABASE_URL to strictly:
        https://<PROJECT_ID>.supabase.co
        Strips trailing slashes and any path segments like /rest/v1, /storage/v1, /api/v1, etc.
        """
        if not url or not url.strip():
            return ""
        trimmed = url.strip()
        parsed = urllib.parse.urlparse(trimmed)
        if parsed.scheme and parsed.netloc:
            # Reconstruct origin only (scheme + host:port)
            return f"{parsed.scheme}://{parsed.netloc}"
        return trimmed.rstrip("/")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase Storage is configured."""
        if self._is_configured_override is not None:
            return self._is_configured_override
        return bool(self.supabase_url and self.secret_key and self.bucket)

    @is_configured.setter
    def is_configured(self, val: bool):
        self._is_configured_override = val

    @is_configured.deleter
    def is_configured(self):
        self._is_configured_override = None

    @property
    def encoded_bucket(self) -> str:
        """Return URL-encoded bucket name."""
        return urllib.parse.quote(self.bucket, safe="")

    def ensure_bucket(self, public: bool = False) -> bool:
        """
        Check if the storage bucket exists, and if not (HTTP 404), create it.
        Returns True if the bucket exists or was successfully created.
        """
        if not self.is_configured:
            return False

        check_url = f"{self.supabase_url}/storage/v1/bucket/{self.encoded_bucket}"
        headers = self._get_headers()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(check_url, headers=headers)
                if res.status_code == 200:
                    return True
                elif res.status_code == 404:
                    # Bucket does not exist, create it
                    logger.info(f"Supabase bucket '{self.bucket}' does not exist. Creating private bucket...")
                    create_url = f"{self.supabase_url}/storage/v1/bucket"
                    create_body = {
                        "id": self.bucket,
                        "name": self.bucket,
                        "public": public,
                    }
                    create_headers = dict(headers)
                    create_headers["Content-Type"] = "application/json"
                    c_res = client.post(create_url, headers=create_headers, json=create_body)
                    if c_res.status_code in (200, 201):
                        logger.info(f"Successfully created Supabase bucket '{self.bucket}'.")
                        return True
                    else:
                        logger.error(
                            f"Failed to create Supabase bucket '{self.bucket}': "
                            f"HTTP {c_res.status_code} - {c_res.text}"
                        )
                        return False
                else:
                    logger.warning(
                        f"Supabase bucket check for '{self.bucket}' returned HTTP {res.status_code} - {res.text}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Network error checking/creating Supabase bucket '{self.bucket}': {e}")
            return False

    def _get_headers(self) -> Dict[str, str]:
        """
        Build authentication headers for Supabase Storage REST API.
        For new Supabase sb_secret_* keys, authentication must be passed using
        the 'apikey' header, NOT as 'Authorization: Bearer'.
        For legacy JWT keys, both are supported.
        Never log or expose the secret key.
        """
        headers: Dict[str, str] = {
            "apikey": self.secret_key,
        }
        # Only attach Authorization: Bearer if using a legacy JWT (not sb_secret_*)
        if not self.secret_key.startswith("sb_secret_") and self.secret_key:
            headers["Authorization"] = f"Bearer {self.secret_key}"
        return headers

    def _object_url(self, path: str) -> str:
        # Quote path parts except slashes
        clean_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
        return f"{self.supabase_url}/storage/v1/object/{self.encoded_bucket}/{clean_path}"

    def upload_file(
        self,
        file_path: Path,
        storage_path: str,
        content_type: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload a file to Supabase storage.
        Returns (success: bool, error_detail: Optional[str]).
        """
        if not self.is_configured:
            logger.warning("Supabase storage is not configured. Skipping upload.")
            return False, "Supabase storage is not configured"

        if not file_path.exists():
            raise FileNotFoundError(f"Source file {file_path} not found.")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return self.upload_bytes(file_bytes, storage_path, content_type or "application/pdf")

    def upload_bytes(
        self,
        file_bytes: bytes,
        storage_path: str,
        content_type: str = "application/pdf",
        upsert: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload raw bytes to Supabase storage.
        If upload fails with 404 (Bucket not found), attempts to automatically
        provision the bucket and retries the upload once.
        Returns (success: bool, error_detail: Optional[str]).
        """
        if not self.is_configured:
            logger.warning("Supabase storage is not configured. Skipping upload.")
            return False, "Supabase storage is not configured"

        url = self._object_url(storage_path)
        headers = self._get_headers()
        headers["Content-Type"] = content_type
        if upsert:
            headers["x-upsert"] = "true"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, headers=headers, content=file_bytes)
                if res.status_code in (200, 201):
                    logger.info(f"Successfully uploaded {storage_path} to Supabase bucket '{self.bucket}'.")
                    return True, None

                # If 404 (Bucket not found), attempt auto-provisioning the bucket and retry once
                if res.status_code == 404 and ("Bucket not found" in res.text or "NoSuchBucket" in res.text):
                    logger.info(f"Bucket '{self.bucket}' not found during upload. Attempting auto-creation...")
                    if self.ensure_bucket():
                        retry_res = client.post(url, headers=headers, content=file_bytes)
                        if retry_res.status_code in (200, 201):
                            logger.info(f"Successfully uploaded {storage_path} after auto-creating bucket '{self.bucket}'.")
                            return True, None
                        else:
                            error_msg = f"HTTP {retry_res.status_code} - {retry_res.text}"
                            logger.error(f"Supabase upload retry failed for {storage_path}: {error_msg}")
                            return False, error_msg

                error_msg = f"HTTP {res.status_code} - {res.text}"
                logger.error(f"Supabase upload failed for {storage_path}: {error_msg}")
                return False, error_msg
        except Exception as e:
            logger.error(f"Network error uploading {storage_path} to Supabase: {e}")
            raise

    def download_file(self, storage_path: str) -> Optional[bytes]:
        """
        Download a file's bytes from Supabase storage.
        """
        if not self.is_configured:
            return None

        url = self._object_url(storage_path)
        headers = self._get_headers()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url, headers=headers)
                if res.status_code == 200:
                    return res.content
                elif res.status_code == 404:
                    logger.warning(f"File {storage_path} not found in Supabase bucket '{self.bucket}'.")
                    return None
                else:
                    logger.error(
                        f"Supabase download failed for {storage_path}: HTTP {res.status_code} - {res.text}"
                    )
                    return None
        except Exception as e:
            logger.error(f"Network error downloading {storage_path} from Supabase: {e}")
            raise

    def delete_file(self, storage_path: str) -> bool:
        """
        Delete an exact file from Supabase storage.
        Passes the exact object storage path in prefixes, which deletes only
        the exact target file and does not match or delete similarly named files.
        """
        if not self.is_configured:
            return False

        if not storage_path or not storage_path.strip():
            logger.warning("Empty storage path passed to delete_file. Aborting delete.")
            return False

        exact_path = storage_path.strip().lstrip("/")
        url = f"{self.supabase_url}/storage/v1/object/{self.encoded_bucket}"
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        body = {"prefixes": [exact_path]}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.request("DELETE", url, headers=headers, json=body)
                if res.status_code == 200:
                    logger.info(f"Successfully deleted exact object '{exact_path}' from Supabase bucket '{self.bucket}'.")
                    return True
                else:
                    logger.error(f"Supabase delete failed for '{exact_path}': HTTP {res.status_code} - {res.text}")
                    return False
        except Exception as e:
            logger.error(f"Network error deleting '{exact_path}' from Supabase: {e}")
            return False

    def list_files(
        self,
        prefix: str = "",
        limit: int = 100,
        paginate: bool = True,
        max_total_files: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        List files in the Supabase bucket matching a prefix.
        Supports pagination to retrieve more than 100 files across batches.
        """
        if not self.is_configured:
            return []

        url = f"{self.supabase_url}/storage/v1/object/list/{self.encoded_bucket}"
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        all_files: List[Dict[str, Any]] = []
        offset = 0
        batch_limit = min(limit, 100) if paginate else limit

        try:
            with httpx.Client(timeout=self.timeout) as client:
                while True:
                    body = {
                        "prefix": prefix,
                        "limit": batch_limit,
                        "offset": offset,
                        "sortBy": {"column": "name", "order": "asc"},
                    }
                    res = client.post(url, headers=headers, json=body)
                    if res.status_code != 200:
                        logger.error(f"Supabase list failed: HTTP {res.status_code} - {res.text}")
                        break

                    batch = res.json()
                    if not isinstance(batch, list) or not batch:
                        break

                    all_files.extend(batch)

                    # Stop if non-paginated, reached end of list, or reached max total files
                    if not paginate or len(batch) < batch_limit or len(all_files) >= max_total_files:
                        break

                    offset += len(batch)

                return all_files
        except Exception as e:
            logger.error(f"Network error listing files from Supabase: {e}")
            return all_files


# Global singleton instance
supabase_storage = SupabaseStorageService()
