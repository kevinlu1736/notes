

def g(x):
    def f():
        x = 2
        x = x + 5
    f()
    print(x)

g(5)

