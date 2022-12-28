
import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import scipy.interpolate

data = np.genfromtxt('CO2_data.csv', delimiter=',')
t = data[0,:]
co2 = data[1,:]
n = t.size
xplot = np.linspace(0, 62, 1000)
yplot = 2.1 * xplot + 0.9
plt.plot(xplot, yplot, t, co2, 'ko')
# plt.show()

######## Problem 1 #############
coeffs = np.polyfit(t, co2, 1)
A1 = coeffs.reshape(1, 2)
# print("A1 = ", A1)
E = np.sqrt((1 / n) * np.sum((coeffs[0] * t + coeffs[1] - co2) ** 2))
A2 = E
print("A2 = ", A2)

#coeffs2 = np.polyfit(t, co2, 2)
a = 30
r = 0.03
b = 300
RMS_Error = lambda coeffs2: np.sqrt((1 / n) * np.sum((coeffs2[0] * np.exp(coeffs2[1] * t) + coeffs2[2] - co2) ** 2))
coeff_min = scipy.optimize.minimize(RMS_Error, np.array([a, r, b]), method='Nelder-Mead')
A3 = coeff_min.x
A3 = A3.reshape(1,3)
# print(A3.shape)
# print(A3)
A4 = RMS_Error(coeff_min.x)
print(A4)

# print(coeff_min.x[0])
# print(coeff_min.x[1])
# print(coeff_min.x[2])
A = -5
B = 4
C = 0
RMS_Error2 = lambda coeffs: np.sqrt((1 / n) * np.sum((coeffs[0] * np.exp(coeffs[1] * t)\
              + coeffs[2] + coeffs[3] * np.sin(coeffs[4]*(t - coeffs[5])) - co2) ** 2))
coeff_min2 = scipy.optimize.minimize(RMS_Error2, np.array([coeff_min.x[0], coeff_min.x[1], coeff_min.x[2], A, B, C]), method='Nelder-Mead', options={'maxiter': 10000})
# print(coeff_min2.x)
A5 = coeff_min2.x

A5 = A5.reshape(1,6)
A6 = RMS_Error2(coeff_min2.x)
print(A6)

######### Problem 2 ##########
data = np.genfromtxt('lynx.csv', delimiter=',')
t = data[0, :]
pop = data[1, :]
n = t.size
# print(pop[10])
pop[10] = 34
# print(pop[10])
# print(pop[28])
pop[28] = 27
# print(pop[28])

interp_func = scipy.interpolate.interp1d(t, pop, kind='cubic')
#print(interp_func(24.5))
A7 = interp_func(24.5)

coeffs = np.polyfit(t, pop, 1)
A8 = coeffs.reshape(1, 2)
# print("A8 = ", A8)
E = np.sqrt((1 / n) * np.sum((coeffs[0] * t + coeffs[1] - pop) ** 2))
A9 = E
# print("A9 = ", A9)

coeffs2 = np.polyfit(t, pop, 2)
A10 = coeffs2.reshape(1, 3)
# print("A10 = ", A10)
E = np.sqrt((1 / n) * np.sum((coeffs2[0] * (t**2) + coeffs2[1] * t + coeffs2[2] - pop) ** 2))
# E = np.sqrt((1 / n) * np.sum((np.polyval(coeffs2, t) - pop) ** 2))
A11 = E
# print("A11 = ", A11)

coeffs10 = np.polyfit(t, pop, 10)
A12 = coeffs10.reshape(1,11)
# print(coeffs10)
E = np.sqrt((1 / n) * np.sum((np.polyval(coeffs10, t) - pop) ** 2))
A13 = E
# print(E)

m1 = 1
m2 = 0
m3 = -1
b1 = 40
b2 = 5
b3 = 30
t1 = 15.5
t2 = 20.5
coeffs = np.array([m1, m2, m3, b1, b2, b3, t1, t2])
def lynx_function(coeffs, t):
    yArr = np.zeros(t.size)
    m1 = coeffs[0]
    m2 = coeffs[1]
    m3 = coeffs[2]
    b1 = coeffs[3]
    b2 = coeffs[4]
    b3 = coeffs[5]
    t1 = coeffs[6]
    t2 = coeffs[7]

    for i in range(t.size):
        if t[i] <= t1:
            yArr[i] = m1 * t[i] + b1
        elif t[i] <= t2:
            yArr[i] = m2 * t[i] + b2
        else:
            yArr[i] = m3 * t[i] + b3
    return yArr

#print(lynx_function(coeffs, t))
E = np.sqrt((1 / n) * np.sum((np.polyval(coeffs10, t) - pop) ** 2))
RMS_Error = lambda coeffs:np.sqrt((1 / n) * np.sum((lynx_function(coeffs, t) - pop) ** 2))
# np.sqrt((1 / n) * (np.sum(lynx_function(coeffs, t)) - pop) ** 2)
coeff_min = scipy.optimize.minimize(RMS_Error, coeffs, method='Nelder-Mead', options={'maxiter': 10000})
#print("x = ",coeff_min.x)
A14 = coeff_min.x[6:8]

A14 = A14.reshape(1,2)
A15 = RMS_Error(coeff_min.x)









