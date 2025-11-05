"""
Base class for linear system solvers
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

__all__ = ['LinearSystemSolver']


class LinearSystemSolver(ABC):
    """
    線形方程式系を解くソルバーの基底クラス
    
    すべての量子線形方程式ソルバーはこのクラスを継承し、
    統一されたインターフェースを提供します。
    """
    
    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        線形方程式系 Ax = b を解く
        
        Args:
            A: 係数行列 (N×N)
            b: 右辺ベクトル (N,)
            
        Returns:
            Tuple of (解ベクトル, メタデータ)
            - 解ベクトル: np.ndarray, 量子アルゴリズムで得られた解
            - メタデータ: dict, アルゴリズム固有の情報（コスト、反復回数など）
        """
        pass
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """初期化メソッド（各実装で定義）"""
        pass

