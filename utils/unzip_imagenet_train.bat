@echo off
setlocal enabledelayedexpansion

REM This file simply extracts the training folder.
REM Note: for this .bat file to work, the original .tar folder has to have already been extracted
REM Place this script in the same directory as the directory that contains the training image folder
REM if you're using a linux based OS, just use ChatGPT to convert this to a bash file to run

REM Navigate to the ILSVRC2012_img_train folder
cd /d "%~dp0ILSVRC2012_img_train"

REM Loop through each .tar file and extract it
for %%f in (*.tar) do (
    echo Extracting %%f...
    mkdir "%%~nf"
    tar -xvf "%%f" -C "%%~nf"
    echo Deleting %%f...
    del "%%f"
)

echo Extraction complete.
pause
