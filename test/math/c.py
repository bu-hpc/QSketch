import math
from scipy.special import comb, perm

# print(perm(3, 2))
# print(comb(3, 2))



def cal(m):
    sum=0
    exp = 0

    for x in range(0,m + 1):
        val=comb(m, x) * pow(1.0/8, x) * pow(7.0/8, m - x)
        ios=math.ceil(x / 4)
        # print(str(x) + ": " + str(val) + ", " + str(ios))
        if (x != 0):
            exp += val * (ios * 4 / x) 

        sum+=val
        # print(str(x) + ": " + str(val) + ", " + str(sum))
        pass

    # for x in range(0,33):
    #     pass

    # print("sum:" + str(sum))
    print("exp: " + str(exp))

    pass

cal(32)
cal(64)
cal(96)
cal(128)
cal(256)
cal(1024)