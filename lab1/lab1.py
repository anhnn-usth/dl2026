def f(x):
    return x**2

def df(x):
    return 2*x

def grad_desc_iter(lr, x_0, iter):
    # x = x_0
    # print(f"{'Step':<10} | {'x':<10} | {'f(x)':<10}")
    # print("-" * 35)
    
    # for i in range(iter):
    #     print(f"{i:<10} | {x:<10.4f} | {f(x):<10.4f}")
        
    #     x = x - lr * df(x)
    return

def grad_desc_thres(lr, x_0, thres):
    x = x_0
    i=0
    print(f"{'Step':<10} | {'x':<10} | {'f(x)':<10}")
    print("-" * 35)
    
    while f(x) > thres:
        print(f"{i:<10} | {x:<10.4f} | {f(x):<10.4f}")
        x = x - lr * df(x)
        i+=1
        
    print(f"{i:<10} | {x:<10.4f} | {f(x):<10.4f}")

grad_desc_thres(0.1, 10, 0.005)
# grad_desc_iter(lr=0.1, x_0=10, iter=20)