"""
Hybrid Solver implementation

QSVTとDFVQLSを組み合わせたハイブリッドソルバーを提供します。
QSVTで初期解を求め、それを初期値としてDFVQLSで精密化します。
"""

from typing import Tuple, Optional
import numpy as np

from .base import LinearSystemSolver
from .dfvqls_solver import DFVQLSSolver

# QSVTソルバーのインポート試行
QSVT_AVAILABLE = False
QSVTSolver = None
try:
    from .qsvt_solver import QSVTSolver
    QSVT_AVAILABLE = True
except ImportError:
    # QSVTが利用できない場合はDFVQLSのみで動作
    pass


class HybridSolver(LinearSystemSolver):
    """
    ハイブリッドソルバー
    
    QSVTとDFVQLSを組み合わせて、より高精度な解を求めます。
    
    アプローチ:
    1. QSVTで初期解を求める（粗い近似）
    2. DFVQLSで精密化する（QSVTの解を初期値として使用）
    
    これにより、QSVTの高速性とDFVQLSの高精度を両立します。
    """
    
    def __init__(
        self,
        matrix_size: int,
        # QSVTパラメータ
        qsvt_poly_degree: int = 100,
        qsvt_kappa: float = 20.0,
        qsvt_use_statevector: bool = True,
        # DFVQLSパラメータ
        dfvqls_num_layers: int = 3,
        dfvqls_optimizer_method: str = 'COBYLA',
        dfvqls_max_iter: int = 200,
        dfvqls_random_seed: int = None,
        # ハイブリッドパラメータ
        use_qsvt_initialization: bool = True,
        use_parameter_init: bool = True,
        init_optimization_budget: int = 50,
        max_iter_refinement: Optional[int] = None,
        verbose: bool = True
    ):
        """
        ハイブリッドソルバーを初期化

        Args:
            matrix_size: 行列サイズ
            qsvt_poly_degree: QSVTの多項式近似の次数
            qsvt_kappa: QSVTの条件数推定値
            qsvt_use_statevector: QSVTでstatevectorモードを使用するか
            dfvqls_num_layers: DFVQLSのアンサッツの層数
            dfvqls_optimizer_method: DFVQLSの最適化手法
            dfvqls_max_iter: DFVQLSの最大反復回数
            dfvqls_random_seed: DFVQLSのランダムシード
            use_qsvt_initialization: QSVTの解を初期値として使用するか
            use_parameter_init: QSVTの解をDF-VQLSパラメータ初期化に使用するか
            init_optimization_budget: パラメータ初期化の最適化回数上限
            max_iter_refinement: 精密化の最大反復回数（Noneの場合はdfvqls_max_iterを使用）
            verbose: 詳細出力
        """
        self.matrix_size = matrix_size
        self.use_qsvt_initialization = use_qsvt_initialization
        self.use_parameter_init = use_parameter_init
        self.init_optimization_budget = init_optimization_budget
        self.max_iter_refinement = max_iter_refinement or dfvqls_max_iter
        self.verbose = verbose
        
        # QSVTソルバーの初期化（利用可能な場合）
        if QSVT_AVAILABLE and use_qsvt_initialization:
            try:
                self.qsvt_solver = QSVTSolver(
                    matrix_size=matrix_size,
                    poly_degree=qsvt_poly_degree,
                    kappa=qsvt_kappa,
                    random_seed=dfvqls_random_seed,
                    verbose=verbose,
                    use_statevector=qsvt_use_statevector
                )
                self.qsvt_available = True
            except Exception as e:
                if verbose:
                    print(f"⚠️  QSVTソルバーの初期化に失敗: {e}")
                self.qsvt_solver = None
                self.qsvt_available = False
        else:
            self.qsvt_solver = None
            self.qsvt_available = False
        
        # DFVQLSソルバーの初期化
        self.dfvqls_solver = DFVQLSSolver(
            matrix_size=matrix_size,
            num_layers=dfvqls_num_layers,
            optimizer_method=dfvqls_optimizer_method,
            max_iter=dfvqls_max_iter,
            random_seed=dfvqls_random_seed,
            verbose=verbose,
            use_parallel=False
        )
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        線形方程式系 Ax = b をハイブリッド手法で解く
        
        Args:
            A: 係数行列 (N×N)
            b: 右辺ベクトル (N,)
            
        Returns:
            Tuple of (解ベクトル, メタデータ)
        """
        metadata = {
            'method': 'hybrid',
            'qsvt_used': False,
            'dfvqls_used': True,
            'qsvt_result': None,
            'dfvqls_result': None
        }
        
        # Step 1: QSVTで初期解を求める（利用可能な場合）
        x_initial = None
        if self.qsvt_available and self.use_qsvt_initialization:
            try:
                if self.verbose:
                    print("\n" + "=" * 70)
                    print("ハイブリッドソルバー: Step 1 - QSVTで初期解を計算")
                    print("=" * 70)
                
                x_qsvt, qsvt_metadata = self.qsvt_solver.solve(A, b)
                x_initial = x_qsvt
                metadata['qsvt_used'] = True
                metadata['qsvt_result'] = {
                    'solution': x_qsvt,
                    'error': qsvt_metadata.get('error'),
                    'time': qsvt_metadata.get('time')
                }
                
                if self.verbose:
                    x_classical = np.linalg.solve(A, b)
                    qsvt_error = np.linalg.norm(x_qsvt - x_classical) / np.linalg.norm(x_classical)
                    print(f"✅ QSVT初期解: {x_qsvt}")
                    print(f"   相対誤差: {qsvt_error:.6e}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  QSVTでの初期解計算に失敗: {e}")
                    print("    DFVQLSのみで解を求めます")
                metadata['qsvt_used'] = False
        
        # Step 2: DFVQLSで精密化
        if self.verbose:
            print("\n" + "=" * 70)
            print("ハイブリッドソルバー: Step 2 - DFVQLSで精密化")
            print("=" * 70)

        # Convert QSVT solution to initial parameters if enabled
        initial_params = None
        if x_initial is not None and self.use_parameter_init:
            if self.verbose:
                print("QSVTの解を初期パラメータに変換中...")

            try:
                import time
                from ..vqls.generalized.utils import qsvt_to_theta_initialization

                t0 = time.time()

                initial_params = qsvt_to_theta_initialization(
                    x_qsvt=x_initial,
                    ansatz=self.dfvqls_solver.solver.ansatz,
                    num_qubits=self.dfvqls_solver.solver.n_qubits,
                    max_iter=self.init_optimization_budget,
                    verbose=self.verbose
                )

                t_convert = time.time() - t0

                if self.verbose:
                    print(f"パラメータ変換完了 ({t_convert:.2f}秒)")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  パラメータ変換失敗: {e}")
                    print("    ランダム初期化を使用します")
                initial_params = None
        elif x_initial is not None:
            if self.verbose:
                print("QSVTの解を取得済み（パラメータ初期化はスキップ）")
        else:
            if self.verbose:
                print("初期値はランダム（QSVT未使用）")

        # DFVQLSで解を求める (with or without initial params)
        x_final, dfvqls_metadata = self.dfvqls_solver.solve(
            A, b,
            initial_params=initial_params
        )
        
        metadata['dfvqls_result'] = {
            'solution': x_final,
            'error': dfvqls_metadata.get('error'),
            'time': dfvqls_metadata.get('time'),
            'iterations': dfvqls_metadata.get('iterations'),
            'cost': dfvqls_metadata.get('final_cost')
        }
        
        # 最終結果の評価
        x_classical = np.linalg.solve(A, b)
        final_error = np.linalg.norm(x_final - x_classical) / np.linalg.norm(x_classical)
        final_residual = np.linalg.norm(A @ x_final - b)
        
        metadata['final_error'] = final_error
        metadata['final_residual'] = final_residual
        
        if self.verbose:
            print(f"\n✅ ハイブリッド解: {x_final}")
            print(f"   相対誤差: {final_error:.6e}")
            print(f"   残差: {final_residual:.6e}")
            print(f"   反復回数: {dfvqls_metadata.get('iterations', 'N/A')}")
            print(f"   最終コスト: {dfvqls_metadata.get('final_cost', 'N/A'):.6f}")
        
        # QSVTとDFVQLSの両方の結果を返す解として使用
        # 将来的には、QSVTの解を初期値としてDFVQLSに渡す実装を追加可能
        
        return x_final, metadata

