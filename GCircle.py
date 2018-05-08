# Plotting Gershgorin Circles for any square matrix
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

# example matrix, each entity can be complex number
A = np.array([[5, 0, 0, -1],
              [1, 0, -1, 0],
              [-1.5, 1, -2, 1],
              [-1, 1, 1, -3j]
             ],dtype='complex')

# begin plotting figure
fig = plt.figure()
ax = fig.add_subplot(111)

# Circle: |A[i,i]-z| <= sum(|A[i,j]| for j in range(n) and j != i)
for i in range(A.shape[0]):
    real = A[i,i].real    # each complex's real part
    imag = A[i,i].imag    # each complex's image part

    # calculate the radius of each circle
    radius = -sqrt(A[i,i].real**2+A[i,i].imag**2)
    for j in range(A.shape[0]):
        radius += sqrt(A[i,j].real**2+A[i,j].imag**2)

    # add the circle to the  figure and plot the center of the circle
    cir = Circle(xy = (real,imag), radius=radius, alpha=0.5, fill=False)
    ax.add_patch(cir)
    x, y = real, imag
    ax.plot(x, y, 'ro')

# title
plt.title("Gershgorin Circles of Matrix")

# show the figure which can be used for analyse eigenvalues of the matrix
plt.show()
