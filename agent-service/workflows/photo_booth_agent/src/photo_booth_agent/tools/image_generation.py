# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import aiohttp
import qrcode
from minio import Minio, S3Error
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import TypedBaseModel
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field
from urllib3.exceptions import MaxRetryError

from photo_booth_agent.configs.minio import POLICY_TEMPLATE, MinioConfig
from photo_booth_agent.constant import SERVICE_NAME
from photo_booth_agent.exception import InvalidImageError
from photo_booth_agent.utils import initialize_minio

logger = logging.getLogger(SERVICE_NAME)


class GenerateImageToolConfig(FunctionBaseConfig, name="generate_image"):
    base_url: str = Field(
        description="The base URL of the image generation service.",
        default="http://image-generator:7070"
        if os.path.exists("/.dockerenv")
        else "http://localhost:7070",
    )
    internal_minio: MinioConfig = Field(
        description="The configuration for the internal MinIO service.",
        default=MinioConfig(),
    )
    external_minio: MinioConfig = Field(
        description="The configuration for the external MinIO service.",
        default=MinioConfig(),
    )
    steps: int = Field(
        description="The number of steps to use to generate the image.",
        default=30,
    )
    disable_safety_checker: bool = Field(
        description="Whether to disable the safety checker.",
        default=False,
    )
    max_image_age: int = Field(
        description="The maximum age of the image in seconds.",
        default=30,
    )
    qrcode_expires: str = Field(
        description="The duration the QR code will be valid.",
        default="00:05:00",
    )
    through_html: bool = Field(
        description="Whether to generate the QR code through an HTML page.",
        default=True,
    )


class GenerateImageInput(TypedBaseModel, name="generate_image"):
    """
    This is the input for the generate_image function.
    """

    image_url_or_path: str = Field(
        description="The URL of the image to edit. Must be a valid image URL",
    )
    prompt: str = Field(
        description="Prompt used to edit the image.",
    )
    action_uuid: str = Field(
        default="",
        description="""
The action UUID of the tool call.
This field will be set by the framework.
You don't need to set it, just leave it empty.
""",
    )
    comments_to_steps: list[str] = Field(
        description="""Comments that are related to the image to be generated.
The comments should be short and entertaining.
Example:
 - 'Uhm... Let me find my water color palette?',
 - 'I'll add a bit of shading to the wheels so it looks more realistic.',
 - 'Finally, I'll add some more details on the car to finish the masterpiece.''""",
    )


class GenerateImageOutput(TypedBaseModel, name="generate_image_output"):
    image_url: str = Field(
        description="The URL of the generated image.",
    )
    description: str = Field(
        description="A short description of the generated image.",
    )
    qrcode_url: str | None = Field(
        description="The URL of the QR code for the generated image. "
        "If None, the QR code is not available.",
    )


def _parse_duration(duration_str: str) -> timedelta:
    """Parse duration string in format HH:MM:SS to timedelta."""
    hours, minutes, seconds = map(int, duration_str.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _upload_to_minio(
    minio_client: Minio,
    bucket: str,
    object_key: str,
    data: bytes,
    content_type: str,
) -> str | None:
    """Upload data to MinIO and return the object URL, or None on failure."""
    logger.debug(
        f"Attempting to upload to MinIO: bucket={bucket}, "
        f"object_key={object_key}, size={len(data)} bytes, "
        f"content_type={content_type}"
    )
    try:
        minio_client.put_object(
            bucket,
            object_key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        logger.debug(f"Successfully uploaded {object_key} to MinIO bucket {bucket}")
        return object_key
    except (S3Error, MaxRetryError, ConnectionError, OSError):
        logger.exception(f"Failed to upload {object_key} to MinIO")
        return None


def _generate_qr_code(target_url: str) -> bytes:
    """Generate a QR code image for the given URL."""
    logger.debug(f"Generating QR code for URL: {target_url[:100]}...")
    qrcode_image = qrcode.make(target_url).get_image()
    qrcode_buffer = io.BytesIO()
    qrcode_image.save(qrcode_buffer, "PNG")
    qr_bytes = qrcode_buffer.getvalue()
    logger.debug(f"Generated QR code, size: {len(qr_bytes)} bytes")
    return qr_bytes


async def _fetch_and_validate_image(url: str, max_age_seconds: float) -> bytes:
    """Fetch image from URL and validate its age."""
    async with aiohttp.ClientSession() as session:
        # Check image age using HEAD request
        async with session.head(url) as head_response:
            try:
                head_response.raise_for_status()
            except Exception as e:
                raise InvalidImageError(f"failed to check image age: {e}") from e

            last_modified_str = head_response.headers.get("Last-Modified")
            if last_modified_str:
                last_modified = parsedate_to_datetime(last_modified_str)
                now = datetime.now(UTC)
                age = now - last_modified

                if age.total_seconds() > max_age_seconds:
                    raise InvalidImageError(
                        f"image is too old ({age.total_seconds():.0f} seconds)."
                        f" Maximum age is {max_age_seconds} seconds."
                    )
                logger.info(f"Image age: {age.total_seconds():.1f} seconds")
            else:
                logger.warning("No Last-Modified header found, skipping age check")

        # Get the actual image data
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.content.read()


def _create_external_sharing_urls(
    external_minio: Minio,
    bucket: str,
    image_bytes: bytes,
    object_key: str,
    expires: timedelta,
    through_html: bool,
) -> tuple[str | None, str | None]:
    """
    Upload image to external MinIO and create sharing URLs.
    """
    logger.debug(
        f"Creating external sharing URLs: bucket={bucket}, "
        f"object_key={object_key}, image_size={len(image_bytes)} bytes, "
        f"expires={expires}, through_html={through_html}"
    )

    try:
        logger.debug(f"Uploading image to external MinIO: {object_key}")
        external_minio.put_object(
            bucket,
            object_key,
            io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/png",
        )
        logger.debug(f"Successfully uploaded image to external MinIO: {object_key}")

        image_download_url = external_minio.presigned_get_object(
            bucket,
            object_key,
            expires=expires,
        )
        logger.debug(f"Generated presigned URL (length: {len(image_download_url)})")
    except Exception:
        logger.exception("Failed to upload image to external MinIO")
        return None, None

    # Optionally create HTML download page
    download_page_url = None
    if through_html:
        logger.debug("Generating HTML download page")
        download_page_html = _generate_download_page_html(image_download_url)
        download_page_html_bytes = download_page_html.encode("utf-8")
        download_page_storage_key = f"html/download_page_{uuid.uuid4()}.html"
        logger.debug(
            f"HTML page size: {len(download_page_html_bytes)} bytes, "
            f"storage_key: {download_page_storage_key}"
        )

        try:
            logger.debug(
                f"Uploading HTML page to external MinIO: {download_page_storage_key}"
            )
            external_minio.put_object(
                bucket,
                download_page_storage_key,
                io.BytesIO(download_page_html_bytes),
                length=len(download_page_html_bytes),
                content_type="text/html",
            )
            logger.debug(
                f"Successfully uploaded HTML page: {download_page_storage_key}"
            )

            download_page_url = external_minio.presigned_get_object(
                bucket,
                download_page_storage_key,
                expires=expires,
                response_headers={
                    "response-content-type": "text/html",
                    "response-content-disposition": "inline",
                },
            )
            logger.debug(
                f"Generated presigned URL for HTML page "
                f"(length: {len(download_page_url)})"
            )
        except (S3Error, MaxRetryError, ConnectionError, OSError):
            logger.exception("Failed to upload HTML page to external MinIO")

    # Return QR target (HTML page or direct image) and download page URL
    qr_target_url = download_page_url or image_download_url
    return qr_target_url, download_page_url


def _generate_download_page_html(image_download_url: str) -> str:
    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}

                .container {{
                    max-width: 500px;
                    width: 100%;
                    background: white;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}

                .image-container {{
                    width: 100%;
                    background: #f8f9fa;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 30px;
                }}

                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}

                .actions {{
                    padding: 20px;
                    background: white;
                }}

                .button {{
                    width: 100%;
                    padding: 16px;
                    margin-bottom: 12px;
                    border: none;
                    border-radius: 12px;
                    font-size: 17px;
                    font-weight: 600;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                    transition: all 0.2s ease;
                    text-decoration: none;
                    color: white;
                }}

                .button:active {{
                    transform: scale(0.98);
                }}

                .button-share {{
                    background: #007AFF;
                }}

                .button-share:hover {{
                    background: #0051D5;
                }}

                .button-exit {{
                    background: #FF3B30;
                    margin-bottom: 0;
                }}

                .button-exit:hover {{
                    background: #D62E24;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="image-container">
                    <img src="{image_download_url}" alt="Generated Image" id="image" />
                </div>
                <div class="actions">
                    <button class="button button-share" onclick="shareImage()">
                        Share
                    </button>
                    <button class="button button-exit" onclick="closeWindow()">
                        Exit
                    </button>
                </div>
            </div>

            <script>
                async function shareImage() {{
                    const imageUrl = "{image_download_url}";

                    // Check if Web Share API is supported
                    if (navigator.share) {{
                        try {{
                            // Fetch the image as blob
                            const response = await fetch(imageUrl);
                            const blob = await response.blob();
                            const file = new File([blob], 'image.png', {{ type: blob.type }});

                            await navigator.share({{
                                title: 'Check out this image!',
                                text: 'Check out this image I generated with Spark & Reachy Photo Booth!',
                                files: [file]
                            }});
                        }} catch (error) {{
                            // Fallback to sharing just the URL
                            try {{
                                await navigator.share({{
                                    title: 'Check out this image!',
                                    text: 'Check out this image I generated with Spark & Reachy Photo Booth!',
                                    url: imageUrl
                                }});
                            }} catch (err) {{
                                window.open("{image_download_url}", "_blank");
                            }}
                        }}
                    }} else {{
                        window.open("{image_download_url}", "_blank");
                    }}
                }}


                function closeWindow() {{
                    // Try to close the window/tab
                    if (window.opener) {{
                        window.close();
                    }} else {{
                        // If can't close, navigate back or show message
                        if (window.history.length > 1) {{
                            window.history.back();
                        }} else {{
                            alert('You can now close this tab');
                        }}
                    }}
                }}
            </script>
        </body>
        </html>
        """  # noqa: E501


@register_function(config_type=GenerateImageToolConfig)
async def generate_image_tool(config: GenerateImageToolConfig, _: Builder):
    # Initialize MinIO clients
    logger.debug(
        f"Initializing internal MinIO: url={config.internal_minio.base_url}, "
        f"bucket={config.internal_minio.bucket}, secure={config.internal_minio.secure}"
    )
    try:
        internal_minio = initialize_minio(
            config.internal_minio.base_url,
            config.internal_minio.access_key,
            config.internal_minio.secret_key.get_secret_value(),
            config.internal_minio.bucket,
            policy=POLICY_TEMPLATE.format(bucket=config.internal_minio.bucket),
            secure=config.internal_minio.secure,
            create_bucket=config.internal_minio.create_bucket,
            timeout=config.internal_minio.timeout,
            num_retries=config.internal_minio.num_retries,
            backoff_factor=config.internal_minio.backoff_factor,
        )
        logger.debug("Successfully initialized internal MinIO")
    except Exception:
        logger.exception("Failed to initialize MinIO")
        raise

    logger.debug(
        f"Initializing external MinIO: url={config.external_minio.base_url}, "
        f"bucket={config.external_minio.bucket}, secure={config.external_minio.secure}"
    )
    try:
        external_minio = initialize_minio(
            config.external_minio.base_url,
            config.external_minio.access_key,
            config.external_minio.secret_key.get_secret_value(),
            config.external_minio.bucket,
            policy=POLICY_TEMPLATE.format(bucket=config.external_minio.bucket),
            secure=config.external_minio.secure,
            create_bucket=config.external_minio.create_bucket,
            timeout=config.external_minio.timeout,
            num_retries=config.external_minio.num_retries,
            backoff_factor=config.external_minio.backoff_factor,
        )
        logger.debug("Successfully initialized external MinIO")
    except Exception:
        logger.exception(
            "Failed to initialize external minio. Image sharing will not be available"
        )
        external_minio = None

    async def _inner(input: GenerateImageInput) -> GenerateImageOutput:
        logger.debug(
            f"generate_image_tool called: action_uuid={input.action_uuid}, "
            f"prompt='{input.prompt[:50]}...', image_url={input.image_url_or_path}"
        )

        if not input.image_url_or_path:
            raise InvalidImageError("image_url_or_path is required")

        logger.info(f"Reading image from URL: {input.image_url_or_path}")

        image_data = await _fetch_and_validate_image(
            input.image_url_or_path, config.max_image_age
        )

        if not image_data:
            raise InvalidImageError("Failed to get image data")

        logger.debug(f"Fetched and validated input image: {len(image_data)} bytes")
        image_base64_str = base64.b64encode(image_data).decode("utf-8")
        request_data = {
            "prompt": input.prompt,
            "image": f"data:image/png;base64,{image_base64_str}",
            "steps": config.steps,
            "disable_safety_checker": config.disable_safety_checker,
        }
        logger.debug(
            f"Sending image generation request: url={config.base_url}, "
            f"steps={config.steps}, "
            f"disable_safety_checker={config.disable_safety_checker}"
        )

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                config.base_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
            ) as response,
        ):
            response.raise_for_status()
            result = await response.json()
            logger.debug("Received response from image generation service")

        image_base64 = result["artifacts"][0]["base64"]
        image_bytes = base64.b64decode(image_base64)
        logger.debug(f"Decoded generated image: {len(image_bytes)} bytes")

        image_object_key = _upload_to_minio(
            internal_minio,
            config.internal_minio.bucket,
            f"images/generated_{input.action_uuid}.jpg",
            image_bytes,
            "image/jpeg",
        )
        if not image_object_key:
            raise RuntimeError("Failed to upload image to internal MinIO")
        internal_url = f"http://{config.internal_minio.base_url}/{config.internal_minio.bucket}/{image_object_key}"
        logger.debug(f"Image uploaded to internal MinIO: {internal_url}")

        if not external_minio:
            logger.debug("External MinIO not available, skipping QR code generation")
            return GenerateImageOutput(
                image_url=internal_url,
                description=f"The generated image of `{input.prompt}`."
                " The QR code is not available.",
                qrcode_url=None,
            )

        # Create external sharing URLs
        logger.debug("Creating external sharing URLs for QR code")
        qr_target_url, _ = _create_external_sharing_urls(
            external_minio,
            config.external_minio.bucket,
            image_bytes,
            image_object_key,
            _parse_duration(config.qrcode_expires),
            config.through_html,
        )

        # If external upload failed, return without QR code
        if not qr_target_url:
            logger.warning(
                "Failed to create external sharing URLs, QR code unavailable"
            )
            return GenerateImageOutput(
                image_url=internal_url,
                description=f"The generated image of `{input.prompt}`."
                " The QR code is not available.",
                qrcode_url=None,
            )

        logger.debug(f"QR target URL created: {qr_target_url[:100]}...")
        qrcode_bytes = _generate_qr_code(qr_target_url)

        logger.debug("Uploading QR code to internal MinIO")
        qrcode_object_key = _upload_to_minio(
            internal_minio,
            config.internal_minio.bucket,
            f"qrcode/qrcode_{uuid.uuid4()}.png",
            qrcode_bytes,
            "image/png",
        )

        qrcode_url = (
            f"http://{config.internal_minio.base_url}/{config.internal_minio.bucket}/{qrcode_object_key}"
            if qrcode_object_key
            else None
        )

        if qrcode_url:
            logger.debug(f"QR code uploaded successfully: {qrcode_url}")
        else:
            logger.warning("Failed to upload QR code to internal MinIO")

        return GenerateImageOutput(
            image_url=internal_url,
            description=f"The generated image of `{input.prompt}`.",
            qrcode_url=qrcode_url,
        )

    yield FunctionInfo.create(
        single_fn=_inner,
        input_schema=GenerateImageInput,
        single_output_schema=GenerateImageOutput,
        description=f"""
Generate an image given a prompt and an image.
The image must not be older than {config.max_image_age} seconds.
""",
    )
