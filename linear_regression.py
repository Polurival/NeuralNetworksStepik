from urllib import request
import numpy as np

'''
Задача 1

У нас есть набор данных: 
знания о длине тормозного пути и скорости для трёх автомобилей.

D   V
10  60
7   50
12  75

Напишите через запятую оценки коэффициентов линейной регрессии D на V, 
т.е. β^0, β^1 для модели D=β0+β1V+ε с точностью до трёх знаков после точки.
'''

'''V = np.array([[1, 60],[1, 50],[1, 75]]) # 1-чки - β^0 1-чный вектор добавленный первым столбцом в матрицу V
D = np.array([[10], [7], [12]])
print(V)
print(D)

# β^ = (XT*X)^-1 * XT * Y   (где X - это V, Y - это D)
step1 = V.T.dot(V)
step2 = np.linalg.inv(step1)
step3 = step2.dot(V.T)
betta_cap = step3.dot(D)
print(betta_cap)'''

'''
Задача 2

Найдите оптимальные коэффициенты для построения линейной регрессии.
На вход вашему решению будет подано название csv-файла, из которого нужно считать данные.
(https://stepic.org/media/attachments/lesson/16462/boston_houses.csv)

Ваша задача — подсчитать вектор коэффициентов линейной регрессии 
для предсказания первой переменной (первого столбца данных) по всем остальным. 
Напомним, что модель линейной регрессии — это y=β0+β1x1+⋯+βnxn.

Напечатайте коэффициенты линейной регрессии, начиная с β0, через пробел.
'''

fname = input()  # read file name from stdin
f = request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with

Y = data[:, :1]  # первый столбец-переменная - предсказываемая

first_column_beta0 = np.ones_like(Y)
X = data[:, 1:]  # последующие столбцы-переменные - предикторы
X_beta0 = np.hstack((first_column_beta0, X)) # впихиваем β0 поправку первым столбцом

step1 = X_beta0.T.dot(X_beta0)
step2 = np.linalg.inv(step1)
step3 = step2.dot(X_beta0.T)
betta_cap = step3.dot(Y)

result_str = ' '.join(map(str, betta_cap.flatten()))
print(result_str)
