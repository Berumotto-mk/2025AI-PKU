import numpy as np

def polynomial_gcd(f, g):
    """
    计算两个多项式的最大公因式（GCD），使用辗转相除法。
    :param f: 多项式系数列表，例如 [1, -5, 7, -2, 4, -8] 表示 x^5 - 5x^4 + 7x^3 - 2x^2 + 4x - 8
    :param g: 多项式系数列表
    :return: 最大公因式的系数列表
    """
    # 将输入转换为 numpy 多项式对象
    f_poly = np.poly1d(f)
    g_poly = np.poly1d(g)

    # 辗转相除
    while True:
        # 计算余数
        remainder = f_poly % g_poly
        if remainder.c == 0:  # 如果余数为 0，返回当前的 g_poly
            return g_poly.c
        # 更新 f_poly 和 g_poly
        f_poly = g_poly
        g_poly = remainder

# 示例多项式
f = [1, -5, 7, -2, 4, -8]  # f(x) = x^5 - 5x^4 + 7x^3 - 2x^2 + 4x - 8
g = [5, -20, 21, -4, 4]    # f'(x) = 5x^4 - 20x^3 + 21x^2 - 4x + 4

# 计算 GCD
gcd_coefficients = polynomial_gcd(f, g)
print("最大公因式的系数:", gcd_coefficients)