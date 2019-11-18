import numpy as np
import matplotlib.pyplot as plt

class Diff:
    def __init__(self,f,h=1.0E-5):
        self.h,self.f = (h,f)
    def __call__(self,x):
        raise NotImplementedError

class Forward1(Diff):
    def __call__(self,x):
        f,h = self.f,self.h
        return (f(x+h)-f(x))/h

class Backward1(Diff):
    def __call__(self,x):
        f,h = self.f,self.h
        return (f(x)-f(x-h))/h

class Central2(Diff):
    def __call__(self,x):
        f,h = self.f,self.h
        return (f(x+h)-f(x-h))/(2*h)

class Central4(Diff):
    def __call__(self,x):
        f,h = self.f,self.h
        return (4/3)*(f(x+h) - f(x-h))/(2*h) - \
                (1/3)*(f(x+2*h)-f(x-2*h))/(4*h)

class Central6(Diff):
    def __call__(self,x):
        f,h = self.f,self.h
        return (3/2)*(f(x+h)-f(x-h))/(2*h) - \
               (3/5)*(f(x+2*h)-f(x-2*h))/(4*h) + \
               (1/10)*(f(x+3*h)-f(x-3*h))/(6*h)

class Forward3(Diff):
    def __call__(self,x):
        f,h = self.f,self.h
        return (-(1/6)*f(x+2*h) + f(x+h) - 0.5*f(x) - \
               (1/3)*f(x-h))/h

diffschemeF = Forward1(np.sin, h=1.0E-5)
diffschemeB = Backward1(np.sin, h=1.0E-5)
diffschemeC = Central2(np.sin, h=1.0E-5)
diffschemeC4 = Central4(np.sin, h=1.0E-5)
diffschemeC6 = Central6(np.sin, h=1.0E-5)
diffschemeF3 = Forward3(np.sin, h=1.0E-5)

diffschemeF(np.pi)
diffschemeB(np.pi)
diffschemeC(np.pi)
diffschemeC4(np.pi)
diffschemeC6(np.pi)
diffschemeF3(np.pi)

def test_Central2():
    def f(x):
        return (a*np.cos(x)**2)*(b*np.sin(x))
    def exact(x):
        return np.cos(x)**3 - 2*np.cos(x)*(np.sin(x)**2)

    a,b = 1.0,1.0
    diffscheme = Central2(f, h=1.0E-5)
    vecdiffscheme = np.vectorize(diffscheme)
    i = np.arange(0,10,0.1)
    numdiff = vecdiffscheme(i)
    # numdiff = [diffscheme(j) for j in i]
    exactdiffval = exact(i)
    plt.plot(i,numdiff,'bo',i,exactdiffval,'k-')
    plt.show()

test_Central2()
