class InfixOperator:

    def __init__(self, func, *a, **kw):
        self.func, self.kw, self.a = func, kw, a

    def __ror__(self, lhs, **kw):
        return InfixOperator(lambda rhs: self.func(lhs, rhs, *self.a, **self.kw))

    def __or__(self, rhs):
        return self.func(rhs)

    def __call__(self, *a, **kw):
        return self.func(*a, **kw)
