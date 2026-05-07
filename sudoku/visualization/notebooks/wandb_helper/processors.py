from functools import reduce
def chain(*funcs):
    def chained(inp):
        return reduce(lambda x, f: f(x), reversed(funcs), inp)
    return chained


