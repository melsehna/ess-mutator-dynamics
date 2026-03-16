import numpy as np
import matplotlib.pyplot as plt

gamma = 0.4
alpha = 2

def fNext(f):
    return f * (1 - gamma) + alpha * f**2 * (1 - f)

fVals = np.linspace(0, 1, 500)
fNextVals = fNext(fVals)

a = alpha
b = -alpha
c = gamma
discriminant = b**2 - 4*a*c

fixedPoints = [0.0]
if discriminant >= 0:
    root1 = (-b + np.sqrt(discriminant)) / (2*a)
    root2 = (-b - np.sqrt(discriminant)) / (2*a)
    fixedPoints.extend([root1, root2])

plt.figure(figsize=(8, 6))
plt.plot(fVals, fNextVals, label=r'$f(t+1)$', lw=2)
plt.plot(fVals, fVals, 'k--', label=r'$f(t+1) = f(t)$')

for fp in fixedPoints:
    if 0 <= fp <= 1:
        plt.plot(fp, fp, 'rx', markersize=10, label='Equilibrium' if fp == fixedPoints[0] else '')

plt.xlabel(r'$f(t)$')
plt.ylabel(r'$f(t+1)$')
plt.legend()
plt.tight_layout()
plt.savefig('memes.png')
plt.show()
