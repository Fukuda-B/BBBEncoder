rem py -m pip install nuitka zstandard

rem py -m nuitka --mingw64 --follow-imports --onefile --standalone --remove-output --windows-disable-console --windows-product-name="BBBEncoder" --windows-file-description="BBB image encoder BBB" bbbencoder.py

py -m nuitka --follow-imports --onefile --standalone --windows-disable-console bbbencoder.py