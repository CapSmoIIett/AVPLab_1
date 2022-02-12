## Первая лабораторная работа по АВП. Тема "Векторизация"

#Задание
Перемножить 2 матрицы следующими способами:
 - результат – матрица C1:
  - с включенной векторизацией
  - с выключенной векторизацией
 - результат – матрица C2:
  - ручная векторизация с использованием SSE2-инструкций (либоновее) С помощью intrinsics (альтернатива ассемблеру)

- элементы матрицы
  - своя матрица меньшего размера
- размер внешней матрицы подбирается самостоятельно
  - время получения матрица C1 от нескольких секунд
- размер внутреней матрицы 8x8, тип входных данных float
