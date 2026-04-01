call .venv\Scripts\python.exe -m ensurepip --upgrade
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip uninstall -y fitz
call .venv\Scripts\python.exe -m pip install -r requirements.txt

pause