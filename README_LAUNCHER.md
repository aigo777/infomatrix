# EyeAssist OS Launcher

Короткая инструкция для финального запуска продукта.

## Быстрый старт

Из корня проекта:

```powershell
cd C:\Users\User\infomatrix
.\venv\Scripts\python.exe src\launcher.py
```

## Что умеет launcher

- `Run Calibration`
  - запускает demo mode и сразу стартует 9-point calibration
- `Launch Demo Mode`
  - открывает презентационный sandbox
- `Launch Field Conditions (OS Mode)`
  - запускает реальное управление Windows курсором

## Настройки

- `Smart Magnetism Assist`
  - включает assist/magnetism
- `Dwell Click`
  - включает dwell-based click path

## Безопасность

В `OS Mode`:

- `F12` = мгновенный kill-switch
- `ESC` = мгновенный kill-switch

## Рекомендуемый порядок для финалов

1. Открыть launcher
2. Нажать `Run Calibration`
3. Проверить `Launch Demo Mode`
4. После проверки запускать `Launch Field Conditions (OS Mode)`

## Если launcher не открывается

Проверить, что запускается Python из виртуальной среды:

```powershell
.\venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

Проверить Qt:

```powershell
.\venv\Scripts\python.exe -c "import launcher; print(launcher.QT_API)"
```

Ожидается `PyQt5` или `PyQt6`.
