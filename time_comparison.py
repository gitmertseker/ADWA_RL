import numpy as np
import random
import time



""""
start_time = time.perf_counter()
x = round(random.uniform(0,5),3)
# print(x)
end_time = time.perf_counter()

print("Run time for round = {} msec".format(1000*(end_time-start_time)))


st = time.perf_counter()
y = random.uniform(0,5)
# print(y)
et = time.perf_counter()
print("Run time for random = {} msec".format(1000*(et-st)))
"""

# a = 9.35678945

# start_time = time.perf_counter()
# round(a,3)
# end_time = time.perf_counter()

# print("Run time for round = {} msec".format(1000*(end_time-start_time)))

# st = time.perf_counter()
# int(a*1000)/1000
# et = time.perf_counter()
# print("Run time for int method = {} msec".format(1000*(et-st)))
x = np.ones((1,40))
# x = np.reshape(x,(1,40))
# print(x.shape)
a = np.zeros((100,100))
# x = a[40,:]
# print(x)
# print(x.shape)
cm = np.ones((40,40))
# print(cm)
# print(cm.shape)
# b = a[0:39,20:60]
# print(b.shape)
b = 100
c = 102
cm = a[60:b,62:c]
d = -1
# print(cm.shape)
# print(cm[0])
# print(cm[0].shape)
cm = a[-10:10,20:60]
cm = a[d:19,20:60]
print(cm)
# print(cm)
# if not (cm.shape == (40,40)):
#     if b > a.shape[0]:
#         cm = np.vstack((cm,np.ones((b-a.shape[0],40))))
#     if c > a.shape[1]:
#         cm = np.hstack((cm,np.ones((40,c-a.shape[1]))))

if not (cm.shape == (40,40)):
    if d < 0:
        cm = []
        cm = np.vstack((cm,np.ones((b-a.shape[0],40))))




# print(cm)
print(cm.shape)