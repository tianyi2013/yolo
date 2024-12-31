:: filepath: /D:/MISC/yolo/build_local.bat
@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Virtual environment setup complete.
echo To activate the virtual environment, run:
echo call venv\Scripts\activate