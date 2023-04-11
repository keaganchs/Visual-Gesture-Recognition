import time

def time_this(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        finish = time.time()
        print(f"{func.__name__} took {finish - start} seconds to execute")
        return result
    return wrapper

