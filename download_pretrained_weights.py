# download_pretrained_weights.py
"""
Run gdown to download a Google Drive folder to ./pretrained/
"""

import os

os.system('gdown --folder "https://drive.google.com/drive/folders/1zOy_zIxkrvmHBIPU72PB_o0Da-h0h5JA" -O ./pretrained/')