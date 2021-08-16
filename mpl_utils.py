import numpy as np
import matplotlib.pyplot as plt

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$-\frac{%s}{%s}$'%(latex,den)
            elif num < 0:
                return r'$-\frac{%s%s}{%s}$'%(-num,latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)

    return _multiple_formatter

if __name__ == "__main__":
    x = np.linspace(-np.pi, 3 * np.pi, 500)
    plt.plot(x, np.cos(x))
    plt.title(r'Multiples of $\pi$')
    ax = plt.gca()
    ax.grid(True)
    ax.set_aspect(1.0)
    ax.axhline(0, color='black', lw=2)
    ax.axvline(0, color='black', lw=2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    plt.show()