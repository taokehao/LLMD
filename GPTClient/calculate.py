def combination(n, k):
    # 处理边界情况
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # 取较小的k值减少计算量
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i  # 逐项相乘并整除，确保结果为整数
    return result

result = combination(10,6)
print(result)
