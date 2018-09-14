
# FB - 201012057
import math

def dec2bin(f):
    if f >= 1:
        g = int(math.log(f, 2))
    else:
        g = -1
    if f >= 0:
        signstr = "0"
    else:
        signstr = "1"
    h = g + 1
    ig = math.pow(2, g)
    st = ""    
    while f > 0 or ig >= 1: 
        if f < 1:
            if len(st[h:]) >= 18: # 18 fractional digits max
                   break
        if f >= ig:
            st += "1"
            f -= ig
        else:
            st += "0"
        ig /= 2
    st = st[:h] + "." + st[h:]
    return st


x = 0.02
y = 0.01
w = 0.01
v = 0.01
s = x*w + y*v
d = y*w - x*v

print("checkSVD(\"{0}\", -- {1:.10f} = x \n         \"{2}\", -- {3:.10f} = x \n         \"{4}\", -- {5:.10f} = x \n         \"{6}\", -- {7:.10f} = x \n         \"{8}\", -- {9:.10f} = x \n         \"{10}\");-- {11:.10f} = x \n".format(dec2bin(x), x, dec2bin(y), y, dec2bin(w), w, dec2bin(v), v, dec2bin(s), s, dec2bin(d), d))
