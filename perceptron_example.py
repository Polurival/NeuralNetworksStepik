"""
Мы погуляли в фруктовом саду и собрали следующие данные про несколько обнаруженных там фруктов:

     желтизна   симметричность  это груша?
1       1             0.3           да
2       0.4           0.5           да
3       0.7           0.8           нет

Пусть у нас есть перцептрон с весами (0, 0) и смещением 0.

Обучите его на приведённых данных различать груши и не груши (пока не надо обучать до сходимости: просто примените обучающее правило последовательно по одному разу на каждом примере, всего у вас получится 3 последовательных применения обучающего правила).

Напишите в ответе через запятую получившиеся у вас смещение, вес для желтизны фрукта и вес для его симметричности (только итоговые, не надо писать результаты каждого шага).


Псевдокод алгоритма обучения:

perfect = False
while NOT perfect do
    perfect = True
    for all e in examples do
        if Predict(e) != Target(e) then
            perfect = False
            if Predict(e) == 0 then
                w = w + e
            end if
            if Predict(e) == 1 then
                w = w - e
            end if
        end if
    end for
end while

"""

import numpy as np


def target(e):
    if e[1] == 1 and e[2] == 0.3:
        return 1
    elif e[1] == 0.4 and e[2] == 0.5:
        return 1
    elif e[1] == 0.7 and e[2] == 0.8:
        return 0


def predict(e):
    _sum = e[0]*w[0] + e[1]*w[1] + e[2]*w[2]
    print("sum ", _sum)

    if _sum > 0:
        return 1
    else:
        return 0


w = np.array([0, 0, 0])
print("w shape ", w.shape)

examples = np.array([[1, 1, 0.3], [1, 0.4, 0.5], [1, 0.7, 0.8]])

perfect = False
while not perfect:
    perfect = True
    for example in examples:
        print("example ", example)
        if predict(example) != target(example):
            perfect = False
            if predict(example) == 0:
                w = w + example
            else:
                w = w - example
            print("w ", w)

print("final weights ", w)
