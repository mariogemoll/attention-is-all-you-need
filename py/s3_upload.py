"""S3 upload utilities for model checkpoints and files."""

import multiprocessing as mp
import os
import time
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from per_process_logs import redirect_stdio


def upload_file_to_s3(
    file_path: str,
    bucket_name: str,
    object_key: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: str = "eu-north-1",
) -> bool:
    """
    Upload a single file to S3.

    Args:
        file_path: Local path to the file to upload
        bucket_name: S3 bucket name
        object_key: S3 object key (path in bucket)
        aws_access_key_id: AWS access key (optional, can use env vars or IAM)
        aws_secret_access_key: AWS secret key (optional, can use env vars or IAM)
        aws_region: AWS region

    Returns:
        True if upload successful, False otherwise
    """
    try:
        # Create S3 client
        if aws_access_key_id and aws_secret_access_key:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region,
            )
        else:
            # Use default credentials (environment variables, IAM role, etc.)
            s3_client = boto3.client("s3", region_name=aws_region)

        # Upload file
        s3_client.upload_file(file_path, bucket_name, object_key)
        return True

    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return False
    except NoCredentialsError:
        print("ERROR: AWS credentials not found")
        return False
    except ClientError as e:
        print(f"ERROR: AWS client error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error uploading {file_path}: {e}")
        return False


def upload_files_to_s3_worker(
    file_paths: List[str],
    bucket_name: str,
    s3_prefix: str,
    log_dir: str,
    epoch: int,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: str = "eu-north-1",
) -> None:
    """
    Worker function to upload multiple files to S3 in a background process.

    Args:
        file_paths: List of local file paths to upload
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix for all uploads (e.g., "runs/ddp_20250912_093645")
        aws_access_key_id: AWS access key (optional)
        aws_secret_access_key: AWS secret key (optional)
        aws_region: AWS region
        log_dir: Directory for log files
        epoch: Current epoch number for log file naming
    """
    # Redirect stdout and stderr to log files like other processes
    console_out, console_err = redirect_stdio(
        os.path.join(log_dir, f"s3_upload_epoch_{epoch}.log"), also_console=False
    )

    print(
        f"S3 upload worker started. Uploading {len(file_paths)} files to "
        f"s3://{bucket_name}/{s3_prefix}"
    )

    successful_uploads = 0
    failed_uploads = 0

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"WARNING: File does not exist, skipping: {file_path}")
            failed_uploads += 1
            continue

        # Generate S3 object key
        file_name = os.path.basename(file_path)
        object_key = f"{s3_prefix.rstrip('/')}/{file_name}"

        print(f"Uploading {file_name} to s3://{bucket_name}/{object_key}")

        # Upload file
        success = upload_file_to_s3(
            file_path, bucket_name, object_key, aws_access_key_id, aws_secret_access_key, aws_region
        )

        if success:
            successful_uploads += 1
            print(f"✓ Successfully uploaded {file_name}")
        else:
            failed_uploads += 1
            print(f"✗ Failed to upload {file_name}")

        # Small delay between uploads to avoid rate limiting
        time.sleep(0.1)

    print(f"S3 upload worker completed. Success: {successful_uploads}, Failed: {failed_uploads}")


def launch_s3_upload_background(
    file_paths: List[str],
    bucket_name: str,
    s3_prefix: str,
    log_dir: str,
    epoch: int,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: str = "eu-north-1",
) -> mp.Process:
    """
    Launch S3 upload in a background process.

    Args:
        file_paths: List of local file paths to upload
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix for all uploads (e.g., "runs/ddp_20250912_093645")
        log_dir: Directory for log files
        epoch: Current epoch number for log file naming
        aws_access_key_id: AWS access key (optional)
        aws_secret_access_key: AWS secret key (optional)
        aws_region: AWS region

    Returns:
        Process object for the background upload
    """
    # Create and start the background process
    process = mp.Process(
        target=upload_files_to_s3_worker,
        args=(
            file_paths,
            bucket_name,
            s3_prefix,
            log_dir,
            epoch,
            aws_access_key_id,
            aws_secret_access_key,
            aws_region,
        ),
    )

    process.start()
    print(f"Started S3 upload background process (PID: {process.pid})")

    return process


def get_checkpoint_files(checkpoint_dir: str, epoch: int) -> List[str]:
    """
    Get list of checkpoint files for a specific epoch.

    Args:
        checkpoint_dir: Directory containing checkpoints
        epoch: Epoch number

    Returns:
        List of file paths for the epoch's checkpoints
    """
    files = []

    # Epoch-specific files only
    epoch_str = f"{epoch:04d}"
    checkpoint_candidates = [
        os.path.join(checkpoint_dir, f"checkpoint_{epoch_str}.pt"),
        os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt"),
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_str}.pt"),
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
    ]
    model_candidates = [
        os.path.join(checkpoint_dir, f"model_{epoch_str}.pt"),
        os.path.join(checkpoint_dir, f"model_{epoch}.pt"),
        os.path.join(checkpoint_dir, f"model_epoch_{epoch_str}.pt"),
        os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"),
    ]

    for candidates in (checkpoint_candidates, model_candidates):
        for file_path in candidates:
            if os.path.exists(file_path):
                files.append(file_path)
                break

    return files


def validate_s3_config() -> dict[str, Optional[str]]:
    """
    Validate S3 configuration and fail early if required settings are missing.

    Returns:
        Dictionary with validated S3 configuration

    Raises:
        ValueError: If required S3 settings are missing or invalid
        RuntimeError: If boto3 cannot be imported or AWS credentials are invalid
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError as e:
        raise RuntimeError(f"boto3 is required for S3 uploads but not installed: {e}")

    # Get configuration from environment
    config = get_s3_config_from_env()

    # Check required settings
    if not config["bucket_name"]:
        raise ValueError(
            "S3_BUCKET_NAME environment variable is required for S3 uploads. "
            "Set it to your S3 bucket name or disable S3 uploads."
        )

    # Validate bucket name format (basic check)
    bucket_name = config["bucket_name"]
    if not bucket_name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"Invalid S3 bucket name format: {bucket_name}")

    # Test AWS credentials by attempting to create an S3 client
    try:
        if config["aws_access_key_id"] and config["aws_secret_access_key"]:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=config["aws_access_key_id"],
                aws_secret_access_key=config["aws_secret_access_key"],
                region_name=config["aws_region"],
            )
        else:
            # Use default credentials (environment variables, IAM role, ~/.aws/credentials, etc.)
            s3_client = boto3.client("s3", region_name=config["aws_region"])

        # Test credentials by listing buckets (minimal permission check)
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ S3 configuration validated: bucket '{bucket_name}' is accessible")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                raise ValueError(f"S3 bucket '{bucket_name}' does not exist or is not accessible")
            elif error_code == "403":
                raise ValueError(
                    f"Access denied to S3 bucket '{bucket_name}'. Check your AWS permissions."
                )
            else:
                raise ValueError(f"Failed to access S3 bucket '{bucket_name}': {e}")

    except NoCredentialsError:
        raise ValueError(
            "AWS credentials not found. Please configure them using one of:\n"
            "  1. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
            "  2. AWS CLI: run 'aws configure'\n"
            "  3. IAM role (when running on EC2)\n"
            "  4. ~/.aws/credentials file"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to validate AWS credentials: {e}")

    return config


# Example usage and configuration
def get_s3_config_from_env() -> dict[str, Optional[str]]:
    """
    Get S3 configuration from environment variables.

    Returns:
        Dictionary with S3 configuration
    """
    return {
        "bucket_name": os.getenv("S3_BUCKET_NAME"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION", "eu-north-1"),
    }


def create_s3_prefix_from_run_id(run_id: str) -> str:
    """
    Create S3 prefix from run ID.

    Args:
        run_id: Run identifier (e.g., "ddp_20250912_093645")

    Returns:
        S3 prefix path (e.g., "runs/ddp_20250912_093645")
    """
    return f"runs/{run_id}"
