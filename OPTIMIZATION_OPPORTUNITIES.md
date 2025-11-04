# DF-VQLS 最適化の機会

## 現在の実装の問題点

### 1. 毎回再計算しているもの

- **vec(K), vec(K^T), f_norm**: 毎回`prepare_matrix`を呼び出しているが、K と f は変わらない
- **|u(θ)⟩の計算**: パラメータが同じなら再利用できる（ただし、最適化中はパラメータが変わるので難しい）
- **transpile**: 同じ回路構造なら再利用可能

### 2. 並列化の機会

- **分子回路と分母回路**: 独立なので並列実行可能
- **複数のパラメータ候補**: バッチ処理で並列実行可能

## モンテカルロ法の最適化手法の適用可能性

### ✅ 適用可能な手法

#### 1. **状態ベクトルのキャッシュ**

```python
# 現在: 毎回計算
vec_K, norm_K = self.state_preparer.prepare_matrix(K)  # 毎回再計算

# 最適化後: 一度だけ計算してキャッシュ
if self._K_hash != hash(K):
    self._vec_K_cache, self._norm_K_cache = vectorize_matrix(K)
    self._K_hash = hash(K)
```

#### 2. **transpile 結果のキャッシュ**

```python
# 現在: 毎回transpile
transpiled_num = transpile(circ_num, self.simulator)

# 最適化後: 回路構造が同じならキャッシュ
circuit_key = hash(circ_num.qasm())
if circuit_key not in self._transpile_cache:
    self._transpile_cache[circuit_key] = transpile(circ_num, self.simulator)
transpiled_num = self._transpile_cache[circuit_key]
```

#### 3. **並列実行（分子と分母）**

```python
# 現在: 順次実行
result_num = self.simulator.run(transpiled_num).result()
result_den = self.simulator.run(transpiled_den).result()

# 最適化後: 並列実行
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    future_num = executor.submit(self.simulator.run, transpiled_num)
    future_den = executor.submit(self.simulator.run, transpiled_den)
    result_num = future_num.result().result()
    result_den = future_den.result().result()
```

#### 4. **状態ベクトル計算の効率化**

```python
# 現在: 毎回回路を実行
u_theta = self._compute_u_theta(params)

# 最適化案: パラメータが近い場合は近似計算
# または、パラメータの差分だけを計算（ただし、VQAでは難しい）
```

#### 5. **StatePreparer のキャッシュ活用**

```python
# 現在: キャッシュはあるが、毎回新しい計算をしている
def prepare_matrix(self, K: np.ndarray):
    vec_K, norm_K = vectorize_matrix(K)  # 毎回計算
    self._vec_K_cache = vec_K
    return vec_K, norm_K

# 最適化後: キャッシュをチェック
def prepare_matrix(self, K: np.ndarray):
    K_hash = hash(K.tobytes())
    if self._K_hash == K_hash and self._vec_K_cache is not None:
        return self._vec_K_cache, self._norm_K_cache
    vec_K, norm_K = vectorize_matrix(K)
    self._vec_K_cache = vec_K
    self._norm_K_cache = norm_K
    self._K_hash = K_hash
    return vec_K, norm_K
```

### ⚠️ 適用が難しい手法

#### 1. **固有値分解を使った expm 計算**

- DF-VQLS では行列指数関数を使っていない
- ただし、状態ベクトルの計算を効率化する手法は適用可能

#### 2. **パラメータの差分計算**

- VQA ではパラメータが毎回変わるため、差分計算は難しい
- ただし、近似的に前回の結果を再利用する手法は可能

## 実装優先順位

### Phase 1: 簡単で効果が大きい最適化

1. **StatePreparer のキャッシュ活用** - K と f は変わらないので、最初に 1 回だけ計算
2. **transpile 結果のキャッシュ** - 回路構造が同じなら再利用

### Phase 2: 並列化

3. **分子と分母回路の並列実行** - 約 2 倍の高速化が期待できる

### Phase 3: 高度な最適化

4. **状態ベクトル計算の近似** - パラメータが近い場合の近似計算
5. **バッチ処理** - 複数のパラメータ候補を並列評価

## 期待される効果

- **StatePreparer キャッシュ**: 各反復で約 0.1 秒削減（K と f の再計算を避ける）
- **transpile キャッシュ**: 各反復で約 0.2 秒削減（transpile の再計算を避ける）
- **並列実行**: 分子と分母の実行時間の最大値まで削減（約 2 倍の高速化）

**合計**: 1 回の反復あたり約 0.3-0.5 秒削減、全体で約 30-50 秒の高速化（100 反復の場合）
