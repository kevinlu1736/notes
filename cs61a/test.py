class Link:
    empty = ()
    def __init__(self, first, rest=empty):
        assert type(rest) is Link or rest is Link.empty, \
        'rest must be a linked list or empty'
        self.first = first
        self.reset = rest

    def __repr__(self):
        if self.reset is Link.empty:
            return 'Link(' + repr(self.first) + ')'
        return 'Link(' + repr(self.first) + ', ' + repr(self.rest) + ')'

    def __str__(self):
        s = '<'
        while self.rest is not Link.empty:
            s += str(self.first) + ', '
            self = self.rest
        return s + str(self.first) + '>'


def g(x):
    def f():
        x = 2
        x = x + 5
    f()
    print(x)


class A:
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return 'My value is ' + str(self.x)

class B(A):
    def __init__(self, y):
        A.__init__(self, y*2)
        self.y = y

    def __repr__(self):
        return 'x = y is ' + str(self.x + self.y)
