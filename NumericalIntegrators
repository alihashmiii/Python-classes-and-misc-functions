import numpy as np

class Integrator:
    def __init__(self,a,b,n):
        self.a,self.b,self.n = a,b,n
        self.points,self.weights = self.construct_method()
    def construct_method(self):
        raise NotImplementedError
    def integrate(self,f):
        s = 0
        for i in range(len(self.weights)):
            s += self.weights[i]*f(self.points[i])
        return s
    def _vectorized_integrate(self,f):
        return np.dot(self.weights,f(self.points))

class Midpoint(Integrator):
    def construct_method(self):
        a,b,n = self.a,self.b,self.n
        h = (b-a)/float(n)
        x = np.linspace(a+0.5*h,b-0.5*h,n)
        w = np.zeros(len(x)) + h
        return x,w

class trapezoidal(Integrator):
    def construct_method(self):
        a,b,n = self.a,self.b,self.n
        x = np.linspace(a,b,n)
        h = (b-a)/float(n-1)
        w = np.zeros(len(x))+h
        w[0] /= 2
        w[-1] /= 2
        return x,w

class Simpson(Integrator):
    def construct_method(self):
        a,b,n = self.a,self.b,self.n
        if n % 2 != 1:
            print('n=%d must be odd, 1 is added' %n)
            n+=1
        x = np.linspace(a,b,n)
        h = (b-a)/float(n-1)*2
        w = np.zeros(len(x))
        w[0:n:2] = h*(1.0/3)
        w[1:n-1:2] = h*(2.0/3)
        w[0] /= 2
        w[-1] /= 2
        return x,w

class GaussLegendre2(Integrator):
    def construct_method(self):
        a,b,n = self.a,self.b,self.n
        if n % 2 != 0:
            print("n=%d must be even, 1 is subtracted" %n)
            n -= 1
        nintervals = int(n /2.0)
        h = (b-a)/float(nintervals)
        x = np.zeros(n)
        sqrt3 = 1.0/np.sqrt(3)
        for i in range(nintervals):
            x[2*i] = a + (i+0.5)*h - 0.5*sqrt3*h
            x[2*i+1] = a + (i+0.5)*h + 0.5*sqrt3*h
        w = np.zeros(len(x))+ h/2.0
        return x,w

if __name__ == '__main__':
    # function to be passed to the integrate method. define any f(x)
    def f(x):
        return x*x

    mid = Midpoint(0,2,101)
    mid.integrate(f)
    mid._vectorized_integrate(f)

    trap = trapezoidal(0,2,101)
    trap.integrate(f)
    trap._vectorized_integrate(f)

    sim = Simpson(0, 2, 100)
    sim.integrate(f)
    sim._vectorized_integrate(f)

    GL = GaussLegendre2(0,2,101)
    GL.integrate(f)
    GL._vectorized_integrate(f)
