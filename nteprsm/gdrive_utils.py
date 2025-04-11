import logging
import os
import io
import json
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from settings import ROOT_DIR

# Configure logging
logger = logging.getLogger(__name__)

def setup_google_drive(client_secrets_path: Optional[Path | str] = ROOT_DIR / "google_client_secret.json") -> Optional[GoogleDrive]:
    """
    Authenticate and set up a Google Drive instance using the provided client secrets file.

    Args:
        client_secrets_path (Optional[Path | str]): Path to the client secrets JSON file required for authentication.

    Returns:
        Optional[GoogleDrive]: An authenticated Google Drive instance if successful, or None if an error occurs.

    Raises:
        RuntimeError: If authentication fails due to an error.
    """
    try:
        gauth = GoogleAuth()
        gauth.LoadClientConfigFile(str(client_secrets_path))
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        logger.error(f"Error during Google Drive authentication: {e}")
        raise RuntimeError("Failed to authenticate with Google Drive.") from e


def get_folder_id(drive, folder_name, parent_id='root'):
    """
    Retrieve the folder ID by its name and parent ID.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        folder_name (str): Name of the folder to search for.
        parent_id (str): ID of the parent folder. Defaults to 'root'.

    Returns:
        str: The folder ID if found, otherwise None.
    """
    try:
        query = (
            f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' "
            f"and '{parent_id}' in parents and trashed=false"
        )
        folder_list = drive.ListFile({'q': query, 'maxResults': 1}).GetList()
        if folder_list:
            return folder_list[0]['id']
        else:
            logger.warning(f"Folder '{folder_name}' not found in parent '{parent_id}'.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving folder ID for '{folder_name}': {e}")
        return None


def get_file_id_from_fullpath(drive, fullpath):
    """
    Retrieve the file ID by its full path from the root.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        fullpath (str): Full path to the file in the format "root/folder1/folder2/file_name".

    Returns:
        Optional[str]: The file ID if found, otherwise None.
    """
    try:
        path_parts = fullpath.split('/')
        current_parent_id = 'root'

        # Traverse the path to find the file
        for part in path_parts[:-1]:
            current_parent_id = get_folder_id(drive, part, current_parent_id)
            if not current_parent_id:
                logger.error(f"Folder '{part}' not found in the path '{fullpath}'.")
                return None

        file_name = path_parts[-1]
        query = (
            f"title='{file_name}' and '{current_parent_id}' in parents and trashed=false"
        )
        file_list = drive.ListFile({'q': query, 'maxResults': 1}).GetList()
        if file_list:
            return file_list[0]['id']
        else:
            logger.warning(f"File '{file_name}' not found in path '{fullpath}'.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving file ID for '{fullpath}': {e}")
        return None


def read_file_from_drive_fullpath(drive, fullpath):
    """
    Read a file from Google Drive using its full path from the root.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        fullpath (str): Full path to the file in the format "root/folder1/folder2/file_name".

    Returns:
        Union[pd.DataFrame, dict, str, None]: Parsed content (DataFrame, dict, or string) or None if an error occurs.
    """
    try:
        file_id = get_file_id_from_fullpath(drive, fullpath)
        if not file_id:
            logger.error(f"File '{fullpath}' not found.")
            return None

        file = drive.CreateFile({'id': file_id})
        content = file.GetContentString()
        mime_type = file.get('mimeType', '')

        if 'csv' in mime_type:
            logger.info(f"Processing '{fullpath}' as CSV.")
            return pd.read_csv(io.StringIO(content))
        elif 'json' in mime_type:
            logger.info(f"Processing '{fullpath}' as JSON.")
            return json.loads(content)
        elif 'plain' in mime_type:
            logger.info(f"Processing '{fullpath}' as plain text.")
            return content
        else:
            logger.warning(f"Unsupported MIME type: {mime_type}. Returning raw content.")
            return content
    except Exception as e:
        logger.error(f"Failed to read or parse file '{fullpath}': {e}")
        return None


def upload_file_to_drive(drive, local_file_path, parent_folder_id='root'):
    """
    Upload a local file to Google Drive.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        local_file_path (str): Local path to the file to upload.
        parent_folder_id (str): ID of the parent folder in Google Drive. Defaults to 'root'.

    Returns:
        Optional[str]: The ID of the uploaded file if successful, otherwise None.
    """
    try:
        file_metadata = {
            'title': os.path.basename(local_file_path),
            'parents': [{'id': parent_folder_id}]
        }
        file = drive.CreateFile(file_metadata)
        file.SetContentFile(local_file_path)
        file.Upload()
        logger.info(f"Uploaded '{local_file_path}' to Google Drive.")
        return file['id']
    except Exception as e:
        logger.error(f"Failed to upload file '{local_file_path}': {e}")
        return None


def upload_object_to_drive(drive, content, title, parent_folder_id='root', mime_type='text/plain', file_format='text'):
    """
    Upload an object (string, CSV, JSON, or pickle content) to Google Drive.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        content (Union[str, pd.DataFrame, dict, Any]): The content to upload. Can be a string, DataFrame, dict, or other serializable object.
        title (str): The title of the file in Google Drive.
        parent_folder_id (str): ID of the parent folder in Google Drive. Defaults to 'root'.
        mime_type (str): MIME type of the content. Defaults to 'text/plain'.
        file_format (str): Format of the content ('text', 'csv', 'json', 'pickle'). Defaults to 'text'.

    Returns:
        Optional[str]: The ID of the uploaded file if successful, otherwise None.
    """
    try:
        file_metadata = {
            'title': title,
            'parents': [{'id': parent_folder_id}],
            'mimeType': mime_type
        }
        file = drive.CreateFile(file_metadata)

        if file_format == 'text':
            file.SetContentString(content)
        elif file_format == 'csv':
            if isinstance(content, pd.DataFrame):
                csv_buffer = io.StringIO()
                content.to_csv(csv_buffer, index=False)
                file.SetContentString(csv_buffer.getvalue())
            else:
                raise ValueError("Content must be a pandas DataFrame for CSV format.")
        elif file_format == 'json':
            if isinstance(content, (dict, list)):
                file.SetContentString(json.dumps(content))
            else:
                raise ValueError("Content must be a dictionary or list for JSON format.")
        elif file_format == 'pickle':
            pickle_buffer = io.BytesIO()
            pickle.dump(content, pickle_buffer)
            pickle_buffer.seek(0)
            file.SetContentFile(pickle_buffer)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        file.Upload()
        logger.info(f"Uploaded object with title '{title}' to Google Drive.")
        return file['id']
    except Exception as e:
        logger.error(f"Failed to upload object with title '{title}': {e}")
        return None
  