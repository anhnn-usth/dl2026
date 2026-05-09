import math

def sigma(z):
    return 1/(1+math.exp(-z))

def df_w2(w1, x1, x2, w0, y, w2):
    return -y*x2 + x2*(1-sigma(-(w1*x1+w2*x2+w0)))

def df_w1(w1, x1, x2, w0, y, w2):
    return -y*x1 + x1 * (1-sigma(-(w1*x1+w2*x2+w0)))

def df_w0(w1, x1, x2, w0, y, w2):
    return 1- y - sigma(-(w1*x1+w2*x2+w0))

def f_w0_w1_w2(w1, x1, x2, w0, y, w2):
    return -y * (w1*x1 + w2*x2 + w0) + math.log(1 + math.exp(w1*x1 + w2*x2 + w0))

def grad_desc(thres, lr, x1_list, x2_list, y_list):
    w0 = 0
    w1 = 1
    w2 = 2
    log =[]
    i=0

    for x1, x2, y in zip(x1_list, x2_list, y_list):
        current_loss = f_w0_w1_w2(w1, x1, x2, w0, y, w2)

        while current_loss > thres:
            log.append((w0, w1, w2))
            print(f"{i:<10} | {x1:<10.4f} | {x2:<10.4f} | {y:<10.4f}{f_w0_w1_w2(w1, x1, x2, w0, y, w2):<10.4f}")

            w0 = w0 - lr * df_w0(w1, x1, x2, w0, y, w2)
            w1 = w1 - lr * df_w1(w1, x1, x2, w0, y, w2)
            w2 = w2 =lr * df_w2(w1, x1, x2, w0, y, w2)
            
            current_loss = f_w0_w1_w2(w1, x1, x2, w0, y, w2)
            i+=1
        log.append((w0,w1,w2))
        print(f"{i:<10} | {x1:<10.4f} | {x2:<10.4f} | {y:<10.4f}{f_w0_w1_w2(w1, x1, x2, w0, y, w2):<10.4f}")

    return log

