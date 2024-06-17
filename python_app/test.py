import numpy as np
import functions


def example(*args, **kwargs):
    if args[0] == "f1":
        f = functions.f1
    elif args[0] == "f2":
        f = functions.f2
    elif args[0] == "f3":
        f = functions.f3
    elif args[0] == "f4":
        f = functions.f5
    elif args[0] == "f5":
        f = functions.f5
    elif args[0] == "f6":
        f = functions.f6
    elif args[0] == "f7":
        f = functions.f7
    elif args[0] == "f8":
        f = functions.f8
    elif args[0] == "f9":
        f = functions.f9
    elif args[0] == "f10":
        f = functions.f10
    elif args[0] == "f11":
        f = functions.f11
    elif args[0] == "f12":
        f = functions.f12
    elif args[0] == "f13":
        f = functions.f13
    elif args[0] == "f14":
        f = functions.f14
    elif args[0] == "f15":
        f = functions.f15
    elif args[0] == "f16":
        f = functions.f16
    elif args[0] == "f17":
        f = functions.f17
    elif args[0] == "f18":
        f = functions.f18
    elif args[0] == "f19":
        f = functions.f19
    elif args[0] == "f20":
        f = functions.f20
    elif args[0] == "f21":
        f = functions.f21
    elif args[0] == "f22":
        f = functions.f22
    elif args[0] == "f23":
        f = functions.f23
    elif args[0] == "f24":
        f = functions.f24
    elif args[0] == "f25":
        f = functions.f25
    elif args[0] == "f26":
        f = functions.f26
    elif args[0] == "f27":
        f = functions.f27
    elif args[0] == "f28":
        f = functions.f28
    elif args[0] == "f29":
        f = functions.f29
    elif args[0] == "f30":
        f = functions.f30

    
    x = args[1]
    y = []
    for vec in x:
        y.append(f([vec])[0])
    return y

def well(vect):
    sum = 0
    for i in vect:
        sum += i*i
    return sum