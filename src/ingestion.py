"""
Módulo de ingestión de datos desde AWS S3 con versionado habilitado.

Responsabilidades:
- Descargar el dataset desde S3 al entorno local del pipeline.
- Registrar la versión exacta del objeto S3 para trazabilidad completa.
- Subir artefactos procesados de vuelta a S3 con versionado.

Configuración esperada (variables de entorno o config.yaml):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    S3_BUCKET_RAW      — bucket con datos crudos versionados
    S3_BUCKET_PROCESSED — bucket para datos procesados
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _get_s3_client(region: str | None = None) -> boto3.client:
    """Construye el cliente S3 usando credenciales del entorno.
    Si MLFLOW_S3_ENDPOINT_URL está definido (p.ej. MinIO local), lo usa como endpoint.
    """
    endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client(
        "s3",
        region_name=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=endpoint_url or None,
    )


def download_dataset(
    bucket: str,
    s3_key: str,
    local_path: str | Path,
    version_id: str | None = None,
    region: str | None = None,
) -> dict:
    """
    Descarga el dataset desde S3 al filesystem local.

    Args:
        bucket:      Nombre del bucket S3 (debe tener versionado habilitado).
        s3_key:      Clave del objeto en S3, p.ej. 'raw/Data_CU_venta.csv'.
        local_path:  Ruta local donde guardar el archivo.
        version_id:  VersionId específico de S3 para pinear una versión exacta.
                     Si es None, descarga la versión más reciente (latest).
        region:      Región AWS (opcional, toma AWS_DEFAULT_REGION del entorno).

    Returns:
        dict con metadata de la descarga: bucket, key, version_id, etag, size.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_s3_client(region)
    kwargs: dict = {"Bucket": bucket, "Key": s3_key}
    if version_id:
        kwargs["VersionId"] = version_id

    logger.info("Descargando s3://%s/%s (version=%s)", bucket, s3_key, version_id or "latest")
    try:
        response = client.get_object(**kwargs)
        actual_version = response.get("VersionId", "N/A")
        etag = response.get("ETag", "").strip('"')
        content_length = response.get("ContentLength", 0)

        with open(local_path, "wb") as f:
            f.write(response["Body"].read())

        logger.info(
            "Descargado → %s  [VersionId=%s | ETag=%s | %.1f MB]",
            local_path,
            actual_version,
            etag,
            content_length / 1_048_576,
        )
        return {
            "bucket": bucket,
            "key": s3_key,
            "version_id": actual_version,
            "etag": etag,
            "size_bytes": content_length,
            "local_path": str(local_path),
        }
    except ClientError as exc:
        logger.error("Error al descargar s3://%s/%s: %s", bucket, s3_key, exc)
        raise


def upload_artifact(
    local_path: str | Path,
    bucket: str,
    s3_key: str,
    metadata: dict | None = None,
    region: str | None = None,
) -> dict:
    """
    Sube un artefacto local a S3 y retorna la versión asignada.

    Args:
        local_path:  Ruta local del archivo a subir.
        bucket:      Bucket S3 destino (versionado habilitado).
        s3_key:      Clave destino en S3.
        metadata:    Metadatos adicionales a adjuntar al objeto S3.
        region:      Región AWS.

    Returns:
        dict con version_id y etag del objeto subido.
    """
    local_path = Path(local_path)
    client = _get_s3_client(region)

    extra_args: dict = {}
    if metadata:
        # S3 metadata: valores deben ser strings
        extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

    logger.info("Subiendo %s → s3://%s/%s", local_path.name, bucket, s3_key)
    try:
        with open(local_path, "rb") as f:
            response = client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=f,
                **extra_args,
            )
        version_id = response.get("VersionId", "N/A")
        etag = response.get("ETag", "").strip('"')
        logger.info("Subido → VersionId=%s | ETag=%s", version_id, etag)
        return {"version_id": version_id, "etag": etag, "s3_uri": f"s3://{bucket}/{s3_key}"}
    except ClientError as exc:
        logger.error("Error al subir %s: %s", local_path, exc)
        raise


def get_latest_version_info(bucket: str, s3_key: str, region: str | None = None) -> dict:
    """
    Recupera metadata de la versión más reciente de un objeto S3.
    Útil para registrar en MLflow sin necesidad de descargar el archivo.
    """
    client = _get_s3_client(region)
    try:
        response = client.head_object(Bucket=bucket, Key=s3_key)
        return {
            "bucket": bucket,
            "key": s3_key,
            "version_id": response.get("VersionId", "N/A"),
            "etag": response.get("ETag", "").strip('"'),
            "last_modified": str(response.get("LastModified")),
            "size_bytes": response.get("ContentLength", 0),
        }
    except ClientError as exc:
        logger.error("Error consultando s3://%s/%s: %s", bucket, s3_key, exc)
        raise


def ensure_bucket_versioning(bucket: str, region: str | None = None) -> None:
    """
    Verifica que el bucket tenga versionado habilitado.
    Si no está activo, lo habilita automáticamente.
    """
    client = _get_s3_client(region)
    try:
        response = client.get_bucket_versioning(Bucket=bucket)
        status = response.get("Status", "")
        if status == "Enabled":
            logger.info("Bucket '%s' — versionado: %s", bucket, status)
        else:
            logger.warning(
                "Bucket '%s' — versionado '%s'. Habilitando automáticamente...", bucket, status
            )
            client.put_bucket_versioning(
                Bucket=bucket,
                VersioningConfiguration={"Status": "Enabled"},
            )
            logger.info("Bucket '%s' — versionado habilitado correctamente.", bucket)
    except ClientError as exc:
        logger.error("Error configurando versionado del bucket '%s': %s", bucket, exc)
        raise
