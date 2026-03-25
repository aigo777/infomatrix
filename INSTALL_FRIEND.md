# Быстрая установка

Проект запускается из корня репозитория `C:\Users\User\infomatrix`.

## 1. Открыть PowerShell в папке проекта

```powershell
cd C:\Users\User\infomatrix
```

## 2. Создать виртуальную среду

Если Python уже установлен:

```powershell
python -m venv venv
```

## 3. Активировать среду

```powershell
.\venv\Scripts\Activate.ps1
```

Если PowerShell ругается на политику:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

## 4. Поставить зависимости

```powershell
pip install -r requirements.txt
```

## 5. Проверить, что всё установилось

```powershell
python -c "import cv2, mediapipe, numpy; print('deps ok')"
```

## 6. Запуск проекта

Без assist:

```powershell
python src\main.py --assist off --os-click off --drift off
```

С assist:

```powershell
python src\main.py --assist on --os-click off --drift off
```

## 7. Полезные проверки

Тесты:

```powershell
$env:PYTHONPATH='C:\Users\User\infomatrix\src'
python -m unittest discover -s src -p "test_*.py"
```

Проверить, какой Python реально используется:

```powershell
python -c "import sys; print(sys.executable)"
```

Нужно, чтобы путь был внутри `C:\Users\User\infomatrix\venv`.
