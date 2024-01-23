from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


google_auth = GoogleAuth()

drive_app = GoogleDrive(google_auth)


# Make sure files are in your root directory

upload_list = ['test.png', 'test.jpg']


for file_to_upload in upload_list:

  # Navigate to your desired Google Drive folder and grab custom ID from URL path

file = drive_app.CreateFile({'parents': [{'id': 'FOLDER_ID_FROM_GOOGLE_DRIVE'}]})

file.SetContentFile(file_to_upload)

file.Upload()