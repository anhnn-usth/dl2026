import csv

def dL_w0(w1, x, w0, y):
    return w1*x+w0-y

def dL_w1(w1, x, w0, y):
    return x*(w1*x+w0-y)

def f_w0_w1(w0, w1, x, y):
    return 1/2 * (w1*x+w0-y)**2

def grad_desc(thres, lr, x_list, y_list):
    w0 = 0
    w1 = 1
    log =[]
    i=0

    for x, y in zip(x_list, y_list):
        current_loss = f_w0_w1(w0, w1, x, y)

        while current_loss > thres:
            log.append((w0, w1))
            print(f"{i:<10} | {x:<10.4f} | {f_w0_w1(w0, w1, x, y):<10.4f}")

            w0 = w0 - lr * dL_w0(w1, x, w0, y)
            w1 = w1 - lr * dL_w1(w1, x, w0, y) 
            i+=1
            current_loss = f_w0_w1(w0, w1, x, y)
        log.append((w0,w1))
        print(f"{i:<10} | {x:<10.4f} | {f_w0_w1(w0, w1, x, y):<10.4f}")

    return log

