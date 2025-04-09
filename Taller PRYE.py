"""Programa realizado por: Valentina Prieto - Bryan Pinzon - Camila Pedraza"""
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Primera parte
#Normal
n_muestras = 1000

mu1, sigma1 = 0, 1
mu2, sigma2 = 5, 2
mu3, sigma3 = 10, 5
mu4, sigma4 = 25, 9

medias_normal_1 = []
medias_normal_2 = []
medias_normal_3 = []
medias_normal_4 = []

for i in range(1, n_muestras + 1):
    muestra_1 = np.random.normal(mu1, sigma1, i)
    muestra_2 = np.random.normal(mu2, sigma2, i)
    muestra_3 = np.random.normal(mu3, sigma3, i)
    muestra_4 = np.random.normal(mu4, sigma4, i)

    medias_normal_1.append(np.mean(muestra_1))
    medias_normal_2.append(np.mean(muestra_2))
    medias_normal_3.append(np.mean(muestra_3))
    medias_normal_4.append(np.mean(muestra_4))

plt.figure(figsize=(10, 5))
plt.plot(medias_normal_1, "r", label="Normal(0, 1)")
plt.plot(medias_normal_2, "g", label="Normal(5, 4)")
plt.title("Ley de los Grandes Números con distribuciones normales")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(medias_normal_3, "b", label="Normal(10, 5)")
plt.plot(medias_normal_4, "y", label="Normal(25, 9)")
plt.title("Ley de los Grandes Números con distribuciones normales")
plt.legend()
plt.grid(True)
plt.show()

# Binomial
n_muestras = 1000
p1 = 0.5
n1 = 10
p2 = 0.15
n2 = 100
p3 = 0.2
n3 = 1000
p4 = 0.5
n4 = 1000
# var = n*p*q
medias_binomial_1 = []
medias_binomial_2 = []
medias_binomial_3 = []
medias_binomial_4 = []
for i in range(n_muestras):
  # Agrega la media de una muestra aleatoría de tamaño t_muestra.
  medias_binomial_1.append(np.mean(np.random.binomial(n1, p1, i+1)))
  medias_binomial_2.append(np.mean(np.random.binomial(n2, p2, i+1)))
  medias_binomial_3.append(np.mean(np.random.binomial(n3, p3, i+1)))
  medias_binomial_4.append(np.mean(np.random.binomial(n4, p4, i+1)))

plt.figure(figsize=(10, 5))
plt.plot(medias_binomial_1, "r",  label="Binomial(10, 0.5) var 2.5")
plt.plot(medias_binomial_2, "g", label="Binomial(100, 0.15) var 12.75")
plt.legend()
plt.grid(True)
plt.title("Ley de los Grandes Números con distribuciones binomiales")
plt.show()
plt.figure(figsize=(15, 9))
plt.plot(medias_binomial_3, "r",  label="Binomial(1000, 0.2) var 160")
plt.plot(medias_binomial_4, "b", label="Binomial(1000, 0.5) var 250")
plt.legend()
plt.grid(True)
plt.title("Ley de los Grandes Números con distribuciones binomiales")
plt.show()


# Parte 2
def funcion_normal(mu, sigma):
  x=[i*0.01+mu for i in range(-100*int(sigma+1),100*int(sigma+1),1)]
  y = [np.exp(-((i-mu)/sigma)**2/2)/(sigma*(2*np.pi)**(1/2)) for i in x]
  return x, y

# Binomial parte 2
n_muestras = 1000
t_muestra = 1000
p1 = 0.5
p2 = 0.001
n1 = 30
n2 = 20
muestras_binomial_1 = []
muestras_binomial_2 = []
for i in range(n_muestras):
  # Agrega la media de una muestra aleatoría de tamaño t_muestra.
  muestras_binomial_1.append(np.mean(np.random.binomial(n1, p1, t_muestra)))
  muestras_binomial_2.append(np.mean(np.random.binomial(n2, p2, t_muestra)))

x_normal_1, y_normal_1 = funcion_normal(p1*n1, (n1*p1*(1-p1)/t_muestra)**(1/2))
x_normal_2, y_normal_2 = funcion_normal(p2*n2, (n2*p2*(1-p2)/t_muestra)**(1/2))

plt.figure(figsize=(15, 8))
plt.plot(x_normal_1, y_normal_1, 'r', label="TLC")
plt.title("Teorema del limite central con distribuciones binomiales var > 0.5")
plt.hist(muestras_binomial_1, density=True, color='blue', bins=15, label="Binomial(30, 0.5) var 7.5")
plt.legend()
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(x_normal_2, y_normal_2, 'b', label="TLC")
plt.title("Teorema del limite central con distribuciones binomiales var < 0.05")
plt.hist(muestras_binomial_2, density=True, color='red', bins=20, label="Binomial(20, 0.001) var 0.01998 < 0.05")
plt.legend()
plt.show()


# Exponencial
n_muestras = 10000
t_muestra = 1000
lambda1 = 0.7
lambda2 = 2

medias_exponencial_1 = []
medias_exponencial_2 = []
for i in range(n_muestras):
  # Agrega la media de una muestra aleatoría de tamaño t_muestra.
  medias_exponencial_1.append(np.mean(np.random.exponential(lambda1, t_muestra)))
  medias_exponencial_2.append(np.mean(np.random.exponential(lambda2, t_muestra)))

x_normal_1, y_normal_1 = funcion_normal(lambda1, lambda1/((t_muestra)**(1/2)))
x_normal_2, y_normal_2 = funcion_normal(lambda2, lambda2/((t_muestra)**(1/2)))

plt.figure(figsize=(10, 5))
plt.title("Teorema del limite central con distribuciones exponenciales var > 0.5")
plt.plot(x_normal_2, y_normal_2, 'y', label="TLC")
plt.hist(medias_exponencial_2, density=True, color='green', label="Binomial(2, 1000) var 4 > 0.5")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Teorema del limite central con distribuciones exponenciales var < 0.5")
plt.plot(x_normal_1, y_normal_1, 'r', label="TLC")
plt.hist(medias_exponencial_1, density=True, color='blue', label="Binomial(0.7, 1000) var 0.49 < 0.5")
plt.legend()
plt.show()

