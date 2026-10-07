import io
import json
import os
import threading
import time
from base64 import b64encode
from hashlib import md5
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase

load_dotenv()


API_URL = "https://api.openai.com/v1/chat/completions"

# HTTP status codes worth one retry:
# 429 = rate limited, 5xx = temporary server problems.
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class VlmRequestError(RuntimeError):
    """
    Raised when the VLM API call fails (timeout, connection
    problem, HTTP error, or an unusable reply).

    The AWARE pipeline catches this, treats the person as
    UNCERTAIN for this frame, and does NOT cache the result. A permanent
    quota error disables verification and reports an explicit status instead.
    """


NO_CREDITS_MESSAGE = "OpenAI API has no credits: GPT verification is OFF"


class VlmInsufficientQuotaError(VlmRequestError):
    """Permanent API quota failure; retrying cannot restore credits."""


class GPTFourWrapper(WrapperBase):
    def __init__(
        self,
        model="gpt-4o",
        enable_caching=True,
        simulate_time_delay=False,
        system_prompt=None,
        cache_file_name="cache_db.json",
        connect_timeout_s=5.0,
        read_timeout_s=15.0,
        max_retries=1,
        max_retry_wait_s=3.0,
        image_detail="auto",
        jpeg_quality=90,
    ):
        """
        Args:
            model: The GPT model to use.
            enable_caching: Cache replies by an MD5 hash of the
                request, and skip the API call if already cached.
                (AWARE keeps this off: live crops are never
                byte-identical, and the pipeline has its own
                per-person cache.)
            simulate_time_delay: When a cached reply is used,
                sleep for the original API delay.
            system_prompt: Optional system prompt.
            cache_file_name: JSON file used when caching is on.
            connect_timeout_s: Give up if connecting to the API
                takes longer than this.
            read_timeout_s: Give up if the API takes longer than
                this to answer.
            max_retries: Extra attempts after a rate-limit (429)
                or temporary server error (5xx).
            max_retry_wait_s: Longest wait before a retry.
            image_detail: "auto", "low" or "high". "low" is much
                cheaper and faster but sees less detail.
            jpeg_quality: JPEG quality used to encode crops.

        Raises:
            RuntimeError: When the API key is not set.
        """

        self.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key is None:
            raise RuntimeError(
                "Could not find API Key in environment variables. "
                "Please set the OPENAI_API_KEY environment variable "
                'in your .bashrc or .profile like: '
                'export OPENAI_API_KEY="XYZXZY...".'
            )

        self.system_prompt = system_prompt
        logger.info(f"Using system prompt: {self.system_prompt}")

        self.model = model
        self.enable_caching = enable_caching
        self.simulate_time_delay = simulate_time_delay

        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.max_retries = max(0, int(max_retries))
        self.max_retry_wait_s = max_retry_wait_s

        self.image_detail = image_detail
        self.jpeg_quality = jpeg_quality

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # ----------------------------------------------------
        # Optional on-disk cache.
        #
        # The lock makes it safe when several VLM calls run
        # in parallel threads.
        # ----------------------------------------------------

        self.cache_db_file_path = Path(__file__).parent / cache_file_name
        self.cache_db = {}
        self._cache_lock = threading.Lock()

        if self.enable_caching and self.cache_db_file_path.exists():

            try:
                with open(self.cache_db_file_path, "r") as f:
                    self.cache_db = json.load(f)

            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"Could not read cache file "
                    f"{self.cache_db_file_path}: {e}. "
                    "Starting with an empty cache."
                )
                self.cache_db = {}

    # ========================================================
    # REQUEST BUILDING
    # ========================================================

    def encode_image(self, image: Image):
        """
        Encode the image as base64 JPEG (much smaller than PNG
        for photos, so uploads are faster).
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_byte_arr = io.BytesIO()
        image.save(
            img_byte_arr,
            format="JPEG",
            quality=self.jpeg_quality,
        )

        return b64encode(img_byte_arr.getvalue()).decode("utf-8")

    def _build_payload(self, prompt: str, image: Image):

        base64_image = self.encode_image(image)

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": self.image_detail,
                    },
                },
            ],
        }

        messages = []

        if self.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )

        messages.append(user_message)

        return {
            "model": f"{self.model}",
            "messages": messages,
            "max_tokens": 300,
        }

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def run_inference(self, prompt: str, image: Image):
        """
        Send the prompt + image to the API and return the reply
        text.

        Raises:
            VlmRequestError: on timeout, connection problems,
                HTTP errors, or unusable replies. Failures are
                never returned as fake answers.
        """

        payload = self._build_payload(prompt, image)

        payload_hash = None

        if self.enable_caching:

            payload_hash = md5(
                json.dumps(payload).encode("utf-8")
            ).hexdigest()

            cached = self._cache_lookup(payload_hash)

            if cached is not None:
                return cached

        content, api_delay = self._post_with_retries(payload)

        if self.enable_caching:
            self._cache_store(payload_hash, content, api_delay)

        return content

    # ========================================================
    # HTTP
    # ========================================================

    def _post_with_retries(self, payload):
        """
        POST the request. Retries only on temporary rate limits (429)
        and temporary server errors (5xx). Timeouts and
        connection errors are raised immediately, so the
        pipeline can move on.

        Returns:
            (reply_text, api_delay_seconds)
        """

        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):

            try:

                response = requests.post(
                    API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=(
                        self.connect_timeout_s,
                        self.read_timeout_s,
                    ),
                )

            except requests.exceptions.Timeout as e:

                raise VlmRequestError(
                    f"VLM request timed out: {e}"
                ) from e

            except requests.exceptions.RequestException as e:

                raise VlmRequestError(
                    f"VLM request failed: {e}"
                ) from e

            # A 429 can mean exhausted credits rather than a temporary rate limit.
            if response.status_code == 429:
                try:
                    error = response.json().get("error", {})
                    no_credits = (
                        error.get("type") == "insufficient_quota"
                        or error.get("code") == "insufficient_quota"
                    )
                except (ValueError, AttributeError, TypeError):
                    no_credits = False
                if no_credits or "insufficient_quota" in response.text:
                    raise VlmInsufficientQuotaError(NO_CREDITS_MESSAGE)

            # ------------------------------------------------
            # Temporary problem: wait briefly and retry.
            # ------------------------------------------------

            if (
                response.status_code in RETRY_STATUS_CODES
                and attempt < attempts
            ):

                wait = self._retry_wait(response)

                logger.warning(
                    f"VLM API returned HTTP {response.status_code}; "
                    f"retrying in {wait:.1f}s "
                    f"(attempt {attempt}/{attempts})."
                )

                time.sleep(wait)
                continue

            # ------------------------------------------------
            # Any other non-success status is a real failure.
            # ------------------------------------------------

            if response.status_code != 200:

                raise VlmRequestError(
                    f"VLM API returned HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:300]}"
                )

            # ------------------------------------------------
            # Extract the reply text.
            # ------------------------------------------------

            try:

                data = response.json()
                content = data["choices"][0]["message"]["content"]

            except (ValueError, KeyError, IndexError, TypeError) as e:

                raise VlmRequestError(
                    f"Unexpected VLM reply: {response.text[:300]}"
                ) from e

            if not isinstance(content, str) or not content.strip():

                raise VlmRequestError(
                    "VLM reply was empty (possibly refused)."
                )

            return content, response.elapsed.total_seconds()

        # Only reached if every attempt was a retryable error.
        raise VlmRequestError(
            "VLM request failed after retries."
        )

    def _retry_wait(self, response):
        """
        How long to wait before retrying. Uses the API's
        Retry-After header if present, capped at
        max_retry_wait_s.
        """

        header = response.headers.get("retry-after")

        try:
            wait = float(header) if header is not None else 1.0

        except ValueError:
            wait = 1.0

        return min(max(wait, 0.0), self.max_retry_wait_s)

    # ========================================================
    # OPTIONAL ON-DISK CACHE
    # ========================================================

    def _cache_lookup(self, payload_hash):

        lookup_start_time = time.time()

        with self._cache_lock:
            entry = self.cache_db.get(payload_hash)

        if entry is None:
            return None

        content = entry.get("content")

        # Older cache files stored the full API response.
        if content is None:

            try:
                content = entry["response"]["choices"][0]["message"]["content"]

            except (KeyError, IndexError, TypeError):

                logger.warning(
                    "Unusable cached response; "
                    "re-running the query."
                )

                return None

        if self.simulate_time_delay:

            lookup_time = time.time() - lookup_start_time
            delta_t = max(0.0, entry.get("api_delay", 0.0) - lookup_time)
            time.sleep(delta_t)

        return content

    def _cache_store(self, payload_hash, content, api_delay):

        with self._cache_lock:

            self.cache_db[payload_hash] = {
                "content": content,
                "api_delay": api_delay,
            }

            self._save_cache_locked()

    def save_cache(self):

        with self._cache_lock:
            self._save_cache_locked()

    def _save_cache_locked(self):
        """
        Write the cache to a temporary file first, then replace
        the real file, so a crash mid-write can't corrupt it.
        Caller must hold self._cache_lock.
        """

        logger.info(f"Saving cache to {self.cache_db_file_path}")

        tmp_path = self.cache_db_file_path.with_suffix(".tmp")

        with open(tmp_path, "w") as f:
            json.dump(self.cache_db, f, indent=4)

        os.replace(tmp_path, self.cache_db_file_path)

    # ========================================================
    # LEGACY
    # ========================================================

    def get_failed_message(self, response):
        """
        Kept for compatibility. run_inference() now raises
        VlmRequestError instead of returning this.
        """

        return f"""
Decision: UNCERTAIN
Justification: FAILED - {response}
"""


if __name__ == "__main__":

    prompt = (
        "We are on a search and rescue mission. Is there an injured "
        "person with a gray shirt in the image? If so, please answer "
        "yes or no if you would recommend sending a rescue team."
    )

    image_folder = Path(__file__).parent / "images"
    model = GPTFourWrapper(enable_caching=False)

    for image_path in image_folder.glob("*.png"):

        image = Image.open(image_path)

        try:
            print(model.run_inference(prompt, image))

        except VlmRequestError as e:
            print(f"Failed for {image_path.name}: {e}")