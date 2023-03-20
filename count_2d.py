import numba as nb

''' numba implementation of a function similar to ak.count that works on 2d arrays
counts the number of times each element appears in the each subarray.
Output: the list of the same size as 'data', but each element is replaced by the number of times it is repeated in the subarray.
numba+awkward array example emplementation taken from
https://github.com/scikit-hep/awkward/discussions/902#discussioncomment-844323
'''

### The wrapper to make numba+awkward work
def njit_at_dim(dim=1):
    def wrapper(impl_dim):
        def token(data, builder):
            pass

        def impl_nd(data, builder):
            for inner in data:
                builder.begin_list()
                token(inner, builder)
                builder.end_list()
            return builder

        @nb.extending.overload(token)
        def dispatch(data, builder):
            if data.type.ndim == dim:
                return impl_dim
            else:
                return impl_nd

        @nb.njit
        def jitted(data, builder):
            return token(data, builder)

        return jitted
    return wrapper

### The implementation part
@njit_at_dim()
def count_2d(data, builder):
    for ii in range(len(data)):
        count = 0
        a = data[ii]
        for jj in range(len(data)):
            if a==data[jj]:
                count+=1
        builder.integer(count)
    return builder