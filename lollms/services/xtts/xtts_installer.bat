@echo off
setlocal

:: Check if portable Conda is installed
where installer_files\miniconda3\Scripts\activate.bat >nul 2>&1
if %errorlevel% == 0 (
    echo "Using portable Conda installation."
    set CONDA_BASE=installer_files\miniconda3
    call "%CONDA_BASE%\Scripts\activate.bat"
    call "%CONDA_BASE%\condabin\conda.bat" deactivate
    call "%CONDA_BASE%\condabin\conda.bat" info --envs | findstr /B /C:"xtts" >nul 2>&1
    if %errorlevel% == 0 (
        echo "Conda environment 'xtts' already exists. Deleting it."
        call "%CONDA_BASE%\condabin\conda.bat" env remove --name xtts --yes
    )
    call "%CONDA_BASE%\condabin\conda.bat" create --name xtts --yes
    call "%CONDA_BASE%\Scripts\activate.bat" xtts
    call "%CONDA_BASE%\Scripts\pip.exe" install xtts-api-server --user
) else (
    echo "No portable Conda found. Checking for system-wide Conda installation."
    where conda >nul 2>&1
    if %errorlevel% == 0 (
        echo "Using system-wide Conda installation."
        call conda deactivate
        call conda info --envs | findstr /B /C:"xtts" >nul 2>&1
        if %errorlevel% == 0 (
            echo "Conda environment 'xtts' already exists. Deleting it."
            call conda env remove --name xtts --yes
        )
        call conda create --name xtts --yes
        call conda activate xtts
        call pip install xtts-api-server --user
    ) else (
        echo "No Conda installation found. Please install Conda."
        exit /b 1
    )
)
exit /b
