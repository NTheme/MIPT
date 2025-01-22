def pascal_triangle():
    arr_cur = [0, 1, 0]
    yield arr_cur[1]
    while True:
        arr_new = [0]
        for i in range(1, len(arr_cur)):
            arr_new.append(arr_cur[i - 1] + arr_cur[i])
            yield arr_new[len(arr_new) - 1]
        arr_new.append(0)
        arr_cur =  arr_new.copy()

