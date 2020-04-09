def test_str():
    a = '28.00%'
    print(a[:-1])

def dist_eclud(vecA, vecB):
    len_a = len(vecA)
    len_b = len(vecB)
    if len_a != len_b:
        raise ValueError
    sum = 0
    for i in range(len_a):
        sum += (float(vecA[i]) - float(vecB[i]))**2
    return sum ** 0.5

def test_min():
    vec_a = ['1','2','3']
    vec_b = ['2','3','4']
    print(dist_eclud(vec_a, vec_b))

if __name__ == '__main__':
    test_min()