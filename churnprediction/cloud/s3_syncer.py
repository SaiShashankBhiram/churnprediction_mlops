
import os
import subprocess
from churnprediction.logging.logger import logging


class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        try:
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            logging.info(f"Running command: {command}")
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            logging.info(f"S3 Sync Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error syncing to S3: {e.stderr}")
            raise

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        try:
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            logging.info(f"Running command: {command}")
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            logging.info(f"S3 Sync Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error syncing from S3: {e.stderr}")
            raise