import math
from scipy.misc import derivative
import sys

""" 

Newton Raphson Method

A method for finding successively better approximations of roots for a function

 """

def f(x):

    """ Input Function """

    f = x**3 + x**2 + 5
    return(f)

def fprime(x):

    """ Compute Derivative of Function """

    d = derivative(f,x)
    return(d)

def newton_raphson(input):

    print('==== Newton Raphson Method ====')

    x_iter = [input]

    #Comput Error (i.e. plung x_inter into f(x))
    err = f(x_iter[0])

    #Iterate for new x_iter until 
    while abs(err) > 1e-6:
        x_current = x_iter[-1]
        x_next = x_current - (f(x_current) / fprime(x_current))
        x_iter.append(x_next)
        err = f(x_next)
        
    print('Total Iterations: ', len(x_iter))
    print('Error: ',err)
    print('X = ',x_iter[-1])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Error: Not Enough Arguments')
        sys.exit()
    else:
        newton_raphson(float(sys.argv[1]))