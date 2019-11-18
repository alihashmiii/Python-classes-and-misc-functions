import numpy as np

class FuncwithDerivative:
    def __init__(self,h=1.0E-5):
        self.h = h
    def __call__(self,x): # this will be implemented in the inherited class .. df and ddf make a call to __call__
        raise NotImplementedError
    def df(self,x):
        h = self.h
        print("superclass")
        return (self(x+h)-self(x-h))/(2.0*h)
    def ddf(self,x):
        h = self.h
        print("superclass")
        return (self(x+h) - 2*self(x) + self(x-h))/(float(h)**2)

class MyFunc(FuncwithDerivative):
    def __init__(self,a):
        self.a = a
    def __call__(self,x):
        return np.cos(self.a*x) + x**3
    def df(self,x):  # polymorphic behaviour .. method overide. subclass method will not call superclass method
        return -self.a*np.sin(self.a*x) + 3*x**2
    def ddf(self,x): # polymorphic behaviour .. method overide. subclass method will not call superclass method
        return -self.a*self.a*np.cos(self.a*x)+6*x


    class MyComplicatedFunc(FuncwithDerivative):
    def __init__(self,p,q,r,h):
        FuncwithDerivative.__init__(self,h=1.0E-5)
        self.p,self.q,self.r = p,q,r
    def __call__(self,x):
        return np.log(abs(self.p*np.tanh(self.q*x*np.cos(self.r*x))))
    
    
# test: .df and .ddf methods in MyFunc will `not` call superclass
func = MyFunc(3)
func.df(4)
func.ddf(4)

# test: .df and .ddf methods in MyComplicatedFunc will call superclass
f = MyComplicatedFunc(1, 1, 1, h=1.0E-5)
x = np.pi/2;
f(x)
f.df(x)
f.ddf(x)
