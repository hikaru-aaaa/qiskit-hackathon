"""
State preparation for DF-VQLS

Handles vectorization and normalization of matrices and vectors.
"""

import numpy as np
from typing import Tuple

from .utils import vectorize_matrix, normalize_vector


class StatePreparer:
    """
    Handles state preparation for DF-VQLS.
    
    Prepares quantum states for:
    - |vec(K)⟩: Vectorized coefficient matrix
    - |vec(K^T)⟩: Vectorized transpose of coefficient matrix
    - |f⟩: Right-hand side vector
    """
    
    def __init__(self):
        """Initialize state preparer."""
        self._vec_K_cache = None
        self._vec_KT_cache = None
        self._f_norm_cache = None
        self._norm_K_cache = None
        self._K_hash = None
        self._f_hash = None
        
        # Statistics for cache hits/misses
        self._cache_hits = 0
        self._cache_misses = 0
    
    def prepare_matrix(self, K: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Prepare vectorized matrix |vec(K)⟩.
        
        Uses caching to avoid recomputation when K hasn't changed.
        
        Args:
            K: Input matrix (N×N)
        
        Returns:
            Tuple of (normalized_vector, frobenius_norm)
        """
        # Check if K has changed using hash
        K_hash = hash(K.tobytes())
        if self._K_hash == K_hash and self._vec_K_cache is not None:
            self._cache_hits += 1
            return self._vec_K_cache, self._norm_K_cache
        
        # Compute and cache
        self._cache_misses += 1
        vec_K, norm_K = vectorize_matrix(K)
        self._vec_K_cache = vec_K
        self._norm_K_cache = norm_K
        self._K_hash = K_hash
        return vec_K, norm_K
    
    def prepare_matrix_transpose(self, K: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Prepare vectorized transpose |vec(K^T)⟩.
        
        Uses caching to avoid recomputation when K hasn't changed.
        
        Args:
            K: Input matrix (N×N)
        
        Returns:
            Tuple of (normalized_vector, frobenius_norm)
        """
        # Check if K has changed using hash
        K_hash = hash(K.tobytes())
        if self._K_hash == K_hash and self._vec_KT_cache is not None:
            # norm_KT is same as norm_K
            self._cache_hits += 1
            return self._vec_KT_cache, self._norm_K_cache
        
        # Compute and cache
        self._cache_misses += 1
        vec_KT, norm_KT = vectorize_matrix(K.T)
        self._vec_KT_cache = vec_KT
        self._K_hash = K_hash
        return vec_KT, norm_KT
    
    def prepare_vector(self, f: np.ndarray) -> np.ndarray:
        """
        Prepare normalized right-hand side vector |f⟩.
        
        Uses caching to avoid recomputation when f hasn't changed.
        
        Args:
            f: Right-hand side vector
        
        Returns:
            Normalized vector
        """
        # Check if f has changed using hash
        f_hash = hash(f.tobytes())
        if self._f_hash == f_hash and self._f_norm_cache is not None:
            self._cache_hits += 1
            return self._f_norm_cache
        
        # Compute and cache
        self._cache_misses += 1
        f_norm = normalize_vector(f)
        self._f_norm_cache = f_norm
        self._f_hash = f_hash
        return f_norm
    
    def get_cached_vec_K(self) -> np.ndarray:
        """Get cached vec(K) if available."""
        if self._vec_K_cache is None:
            raise ValueError("vec(K) not prepared yet")
        return self._vec_K_cache
    
    def get_cached_vec_KT(self) -> np.ndarray:
        """Get cached vec(K^T) if available."""
        if self._vec_KT_cache is None:
            raise ValueError("vec(K^T) not prepared yet")
        return self._vec_KT_cache
    
    def get_cached_f_norm(self) -> np.ndarray:
        """Get cached normalized f if available."""
        if self._f_norm_cache is None:
            raise ValueError("f not prepared yet")
        return self._f_norm_cache
    
    def get_cached_norm_K(self) -> float:
        """Get cached Frobenius norm of K if available."""
        if self._norm_K_cache is None:
            raise ValueError("norm(K) not computed yet")
        return self._norm_K_cache
    
    def clear_cache(self):
        """Clear all cached values."""
        self._vec_K_cache = None
        self._vec_KT_cache = None
        self._f_norm_cache = None
        self._norm_K_cache = None
        self._K_hash = None
        self._f_hash = None
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return {'hits': 0, 'misses': 0, 'hit_rate': 0.0}
        hit_rate = self._cache_hits / total
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate
        }

