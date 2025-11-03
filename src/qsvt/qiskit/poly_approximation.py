import numpy as np


def fit_invx_poly(delta: float, degree: int, upper_limit: float = 1.5):
    """
    L2([delta, upper_limit]) で 1/x を奇数次項 x, x^3, x^5, ..., x^degree で最小二乗近似

    Args:
        delta: 区間の下限 [delta, upper_limit]
        degree: 最大次数（奇数のみ使用）
        upper_limit: 区間の上限（デフォルト1.5）

    Returns:
        (coeffs, A, b, J): 係数配列、行列A、ベクトルb、目的関数値
    """
    # 奇数次項のみを生成
    if degree % 2 == 0:
        degree -= 1  # 偶数の場合は1つ下の奇数にする

    exps = np.arange(1, degree + 1, 2, dtype=float)

    # A_ij = ∫ x^{e_i+e_j} dx from delta to upper_limit
    E = exps[:, None] + exps[None, :]
    A = (upper_limit ** (E + 1) - delta ** (E + 1)) / (E + 1)

    # b_i = ∫ x^{e_i-1} dx from delta to upper_limit
    b = (upper_limit ** exps - delta ** exps) / exps

    # Solve A c = b
    coeffs = np.linalg.solve(A, b)

    # 目的関数の理論値
    int_invx2 = (1.0 / delta) - (1.0 / upper_limit)  # ∫_{delta}^{upper_limit} x^{-2} dx
    J = float(int_invx2 - 2 * coeffs @ b + coeffs @ (A @ coeffs))

    # 偶数項を含めた係数
    coeffs_with_even = []
    for i in range(len(coeffs) * 2):
        if i % 2 == 0:
            coeffs_with_even.append(0.0)
        else:
            coeffs_with_even.append(coeffs[i // 2])

    return coeffs, A, b, J, exps, coeffs_with_even


def evaluate_poly(x, coeffs, exps):
    """
    多項式を評価: sum(coeffs[i] * x^exps[i])
    """
    result = np.zeros_like(x, dtype=float)
    for i, (c, exp) in enumerate(zip(coeffs, exps)):
        result += c * (x ** exp)
    return result


def verify_plot(coeffs, exps, delta, upper_limit=1.5, n=2000):
    """
    近似結果をプロットして検証
    """
    for i, (c, exp) in enumerate(zip(coeffs, exps)):
        print(f"  c_{int(exp)}: {c:.10f}")

    xs = np.linspace(delta, upper_limit, n)
    y_true = 1.0 / xs
    y_fit = evaluate_poly(xs, coeffs, exps)
    err = y_true - y_fit

    # 数値積分で目的関数確認
    J_num = np.trapz(err**2, xs)

    # 理論値と比較
    _, _, _, J_theory, _, _ = fit_invx_poly(delta, int(max(exps)), upper_limit)

    print(f"delta = {delta}, upper_limit = {upper_limit}")
    coeffs_with_even = []
    for i in range(len(coeffs) * 2):
        if i % 2 == 0:
            coeffs_with_even.append(0.0)
        else:
            coeffs_with_even.append(coeffs[i // 2])
    print(f"Polynomial coefficients (including zeros for even powers): {coeffs_with_even}")

    print(f"max degree = {int(max(exps))}")
    print(f"Objective (theory):  J = {J_theory:.10e}")
    print(f"Objective (numeric): J = {J_num:.10e}")

    # 多項式の文字列表現
    poly_str = " + ".join([f"{c:.6f}*x^{int(exp)}" for c, exp in zip(coeffs, exps)])
    print(f"Polynomial: {poly_str}")


if __name__ == "__main__":
    # 修正された使い方
    delta = 0.2
    upper_limit = 1.5
    coeffs, A, b, J, exps, coeffs_with_even = fit_invx_poly(delta, degree=15, upper_limit=upper_limit)
    verify_plot(coeffs, exps, delta, upper_limit)
