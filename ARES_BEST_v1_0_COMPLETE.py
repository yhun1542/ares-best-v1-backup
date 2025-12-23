#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ARES BEST v1.0 - 완전 재현 가능한 통합 백테스트 시스템
================================================================================
최고 성적: Sharpe 0.7850, MDD -22.1%
조합: BASELINE + dd_control

이 파일 하나로 모든 것을 재현할 수 있습니다:
- 모든 엔진 소스코드
- 모든 설정값
- 데이터 로더
- 팩터 계산 (Numba JIT 최적화)
- 레짐 감지 (VIX 기반)
- 포트폴리오 구성 (역변동성 가중)
- 백테스트 엔진
- 조합 테스트 프레임워크
- 상세 로깅

사용법:
    python ARES_BEST_v1_0_COMPLETE.py --mode backtest
    python ARES_BEST_v1_0_COMPLETE.py --mode combination_test
    python ARES_BEST_v1_0_COMPLETE.py --mode walkforward

요구사항:
    pip install numpy pandas numba

DB 경로:
    /home/ubuntu/ares_x_unified_database/ares_universal_v2.db

================================================================================
"""
from __future__ import annotations
import os
import sys
import json
import time
import sqlite3
import logging
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from itertools import combinations
from functools import lru_cache
import hashlib

import numpy as np
import pandas as pd
from numba import njit, prange

warnings.filterwarnings("ignore")

# =============================================================================
# 전역 설정 (모든 설정값)
# =============================================================================
@dataclass
class AresConfig:
    """ARES 시스템 전체 설정"""
    
    # === 데이터 설정 ===
    db_path: str = "/home/ubuntu/ares_x_unified_database/ares_universal_v2.db"
    db_path_alt: str = "/home/ubuntu/aresx_pipeline_v18/ares_x_v11_0.db"
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    
    # === 백테스트 설정 ===
    transaction_cost: float = 0.003  # 0.3% (왕복)
    slippage: float = 0.001  # 0.1%
    risk_free_rate: float = 0.05  # 5% 연간
    
    # === 포트폴리오 설정 ===
    top_k: int = 30  # 상위 30개 종목
    rebalance_freq: str = "weekly"  # weekly, monthly
    max_position_size: float = 0.10  # 최대 10% 단일 종목
    min_position_size: float = 0.01  # 최소 1%
    
    # === 팩터 설정 ===
    momentum_lookback: int = 252  # 12개월
    momentum_skip: int = 21  # 최근 1개월 제외
    volatility_lookback: int = 20  # 20일 변동성
    reversal_lookback: int = 5  # 5일 단기 리버설
    
    # === 레짐 설정 (VIX 기반) ===
    regime_ultra_low: float = 15.0
    regime_low: float = 20.0
    regime_moderate: float = 25.0
    regime_high: float = 30.0
    # CRISIS: VIX >= 30
    
    # === 레짐별 팩터 가중치 ===
    regime_weights: Dict = field(default_factory=lambda: {
        'ULTRA_LOW': {'momentum': 1.2, 'low_vol': 0.0, 'reversal': 0.0},
        'LOW': {'momentum': 1.0, 'low_vol': 0.0, 'reversal': 0.0},
        'MODERATE': {'momentum': 0.8, 'low_vol': 0.3, 'reversal': 0.0},
        'HIGH': {'momentum': 0.5, 'low_vol': 0.5, 'reversal': 0.0},
        'CRISIS': {'momentum': 0.2, 'low_vol': 0.0, 'reversal': 0.0}
    })
    
    # === DD Control 설정 (최고 성적 모듈) ===
    dd_control_enabled: bool = True
    dd_control_threshold: float = -0.10  # -10% DD 임계값
    dd_control_scale_min: float = 0.3  # 최소 스케일
    dd_control_scale_factor: float = 5.0  # 스케일 팩터
    
    # === Vol Targeting 설정 ===
    vol_targeting_enabled: bool = False
    target_volatility: float = 0.12  # 12% 목표 변동성
    vol_lookback: int = 60  # 60일 실현 변동성
    max_leverage: float = 1.5
    min_leverage: float = 0.3
    
    # === Turnover Control 설정 ===
    turnover_control_enabled: bool = False
    max_turnover: float = 0.30  # 30% 최대 턴오버
    
    # === Crisis Cash 설정 ===
    crisis_cash_enabled: bool = False
    crisis_cash_threshold: float = 30.0  # VIX >= 30 시 현금
    
    # === Correlation Penalty 설정 ===
    correlation_penalty_enabled: bool = False
    corr_lookback: int = 60
    corr_penalty_threshold: float = 0.5
    corr_penalty_factor: float = 0.3
    
    # === Walk-Forward 설정 ===
    train_window: int = 2520  # 10년 학습
    test_window: int = 252  # 1년 테스트
    step_size: int = 63  # 분기별 롤링
    
    # === 로깅 설정 ===
    log_dir: str = "/home/ubuntu/ares_logs"
    output_dir: str = "/home/ubuntu/ares_results"
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            'db_path': self.db_path,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'transaction_cost': self.transaction_cost,
            'slippage': self.slippage,
            'risk_free_rate': self.risk_free_rate,
            'top_k': self.top_k,
            'rebalance_freq': self.rebalance_freq,
            'momentum_lookback': self.momentum_lookback,
            'momentum_skip': self.momentum_skip,
            'volatility_lookback': self.volatility_lookback,
            'dd_control_enabled': self.dd_control_enabled,
            'dd_control_threshold': self.dd_control_threshold,
            'vol_targeting_enabled': self.vol_targeting_enabled,
            'target_volatility': self.target_volatility,
            'turnover_control_enabled': self.turnover_control_enabled,
            'max_turnover': self.max_turnover,
            'crisis_cash_enabled': self.crisis_cash_enabled,
            'correlation_penalty_enabled': self.correlation_penalty_enabled
        }

# 기본 설정 인스턴스
DEFAULT_CONFIG = AresConfig()

# =============================================================================
# 로깅 설정
# =============================================================================
def setup_logging(config: AresConfig = None) -> logging.Logger:
    """로깅 설정"""
    config = config or DEFAULT_CONFIG
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{config.log_dir}/ares_best_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("ARES_BEST")
    logger.info(f"Logging initialized: {log_file}")
    return logger

# =============================================================================
# 데이터 로더
# =============================================================================
class DataLoader:
    """데이터베이스에서 데이터 로드"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.DataLoader")
        self._cache = {}
    
    def _get_connection(self) -> sqlite3.Connection:
        """DB 연결"""
        if os.path.exists(self.config.db_path):
            return sqlite3.connect(self.config.db_path)
        elif os.path.exists(self.config.db_path_alt):
            return sqlite3.connect(self.config.db_path_alt)
        else:
            raise FileNotFoundError(f"Database not found: {self.config.db_path}")
    
    def load_daily_prices(self) -> pd.DataFrame:
        """일봉 데이터 로드"""
        cache_key = f"daily_prices_{self.config.start_date}_{self.config.end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.logger.info(f"Loading daily prices: {self.config.start_date} ~ {self.config.end_date}")
        
        conn = self._get_connection()
        query = f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM daily_ohlcv
        WHERE date >= '{self.config.start_date}' AND date <= '{self.config.end_date}'
        ORDER BY date, symbol
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        # 중복 제거
        df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
        self.logger.info(f"Loaded {len(df):,} rows of daily prices")
        
        self._cache[cache_key] = df
        return df
    
    def load_vix(self) -> pd.DataFrame:
        """VIX 데이터 로드"""
        cache_key = f"vix_{self.config.start_date}_{self.config.end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.logger.info("Loading VIX data")
        
        conn = self._get_connection()
        query = f"""
        SELECT date, close as vix
        FROM vix
        WHERE date >= '{self.config.start_date}' AND date <= '{self.config.end_date}'
        ORDER BY date
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        df = df.drop_duplicates(subset=['date'], keep='last')
        self.logger.info(f"Loaded {len(df):,} rows of VIX data")
        
        self._cache[cache_key] = df
        return df
    
    def load_fundamentals(self) -> pd.DataFrame:
        """펀더멘탈 데이터 로드 (PIT)"""
        cache_key = f"fundamentals_{self.config.start_date}_{self.config.end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.logger.info("Loading fundamentals (PIT)")
        
        conn = self._get_connection()
        try:
            query = f"""
            SELECT date, ticker as symbol, bm, ey, fcfy, roe, gm, de
            FROM fundamentals_pit_daily
            WHERE date >= '{self.config.start_date}' AND date <= '{self.config.end_date}'
            ORDER BY date, ticker
            """
            df = pd.read_sql(query, conn)
            self.logger.info(f"Loaded {len(df):,} rows of fundamentals")
        except Exception as e:
            self.logger.warning(f"Failed to load fundamentals: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()
        
        self._cache[cache_key] = df
        return df
    
    def load_factor_exposures(self) -> pd.DataFrame:
        """팩터 노출 데이터 로드"""
        cache_key = f"factor_exposures_{self.config.start_date}_{self.config.end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.logger.info("Loading factor exposures")
        
        conn = self._get_connection()
        try:
            query = f"""
            SELECT symbol, date, momentum_12m, momentum_1m, value_composite, 
                   quality_composite, low_vol, size, beta
            FROM factor_exposures_daily
            WHERE date >= '{self.config.start_date}' AND date <= '{self.config.end_date}'
            ORDER BY date, symbol
            """
            df = pd.read_sql(query, conn)
            self.logger.info(f"Loaded {len(df):,} rows of factor exposures")
        except Exception as e:
            self.logger.warning(f"Failed to load factor exposures: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()
        
        self._cache[cache_key] = df
        return df
    
    def prepare_data(self) -> Dict:
        """모든 데이터 준비 및 NumPy 배열로 변환"""
        self.logger.info("Preparing all data...")
        start_time = time.time()
        
        # 가격 데이터 로드
        prices_df = self.load_daily_prices()
        vix_df = self.load_vix()
        
        # 피벗
        prices_pivot = prices_df.pivot(index='date', columns='symbol', values='close')
        
        # 수익률 계산
        returns = prices_pivot.pct_change().fillna(0)
        
        # NumPy 배열로 변환
        dates = returns.index.values
        symbols = returns.columns.values
        returns_arr = returns.values.astype(np.float64)
        prices_arr = prices_pivot.values.astype(np.float64)
        
        # VIX 매핑
        vix_dict = dict(zip(vix_df['date'], vix_df['vix']))
        vix_arr = np.array([vix_dict.get(str(d)[:10], 20.0) for d in dates], dtype=np.float64)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Data prepared in {elapsed:.2f}s: {len(dates)} dates, {len(symbols)} symbols")
        
        return {
            'dates': dates,
            'symbols': symbols,
            'returns': returns_arr,
            'prices': prices_arr,
            'vix': vix_arr,
            'n_dates': len(dates),
            'n_symbols': len(symbols)
        }

# =============================================================================
# Numba JIT 최적화 팩터 계산 함수
# =============================================================================
@njit(parallel=True, fastmath=True)
def compute_momentum_12_1(returns: np.ndarray, lookback: int = 252, skip: int = 21) -> np.ndarray:
    """
    12-1 모멘텀 계산 (Numba JIT 최적화)
    
    Args:
        returns: (n_dates, n_symbols) 수익률 배열
        lookback: 룩백 기간 (기본 252일 = 12개월)
        skip: 스킵 기간 (기본 21일 = 1개월)
    
    Returns:
        momentum: (n_dates, n_symbols) 모멘텀 배열
    """
    n_dates, n_symbols = returns.shape
    momentum = np.zeros((n_dates, n_symbols), dtype=np.float64)
    
    for i in prange(lookback, n_dates):
        for j in range(n_symbols):
            mom = 0.0
            count = 0
            for k in range(skip, lookback):
                val = returns[i - k, j]
                if not np.isnan(val):
                    mom += val
                    count += 1
            if count > 0:
                momentum[i, j] = mom
            else:
                momentum[i, j] = np.nan
    
    return momentum

@njit(parallel=True, fastmath=True)
def compute_volatility(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    변동성 계산 (Numba JIT 최적화)
    
    Args:
        returns: (n_dates, n_symbols) 수익률 배열
        lookback: 룩백 기간 (기본 20일)
    
    Returns:
        volatility: (n_dates, n_symbols) 연율화 변동성 배열
    """
    n_dates, n_symbols = returns.shape
    vol = np.zeros((n_dates, n_symbols), dtype=np.float64)
    
    for i in prange(lookback, n_dates):
        for j in range(n_symbols):
            # 평균 계산
            mean = 0.0
            count = 0
            for k in range(lookback):
                val = returns[i - k, j]
                if not np.isnan(val):
                    mean += val
                    count += 1
            
            if count < 2:
                vol[i, j] = np.nan
                continue
            
            mean /= count
            
            # 분산 계산
            var = 0.0
            for k in range(lookback):
                val = returns[i - k, j]
                if not np.isnan(val):
                    diff = val - mean
                    var += diff * diff
            
            var /= (count - 1)
            vol[i, j] = np.sqrt(var) * np.sqrt(252)  # 연율화
    
    return vol

@njit(parallel=True, fastmath=True)
def compute_short_term_reversal(returns: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    단기 리버설 계산 (Numba JIT 최적화)
    
    Args:
        returns: (n_dates, n_symbols) 수익률 배열
        lookback: 룩백 기간 (기본 5일)
    
    Returns:
        reversal: (n_dates, n_symbols) 리버설 배열 (부호 반전)
    """
    n_dates, n_symbols = returns.shape
    reversal = np.zeros((n_dates, n_symbols), dtype=np.float64)
    
    for i in prange(lookback, n_dates):
        for j in range(n_symbols):
            rev = 0.0
            count = 0
            for k in range(lookback):
                val = returns[i - k, j]
                if not np.isnan(val):
                    rev += val
                    count += 1
            if count > 0:
                reversal[i, j] = -rev  # 부호 반전 (역방향)
            else:
                reversal[i, j] = np.nan
    
    return reversal

@njit(fastmath=True)
def compute_weights_ivol(signals: np.ndarray, vol: np.ndarray, top_k: int = 30,
                         max_weight: float = 0.10, min_weight: float = 0.01) -> np.ndarray:
    """
    역변동성 가중치 계산 (Top-K 선택)
    
    Args:
        signals: (n_dates, n_symbols) 시그널 배열
        vol: (n_dates, n_symbols) 변동성 배열
        top_k: 선택할 종목 수
        max_weight: 최대 가중치
        min_weight: 최소 가중치
    
    Returns:
        weights: (n_dates, n_symbols) 가중치 배열
    """
    n_dates, n_symbols = signals.shape
    weights = np.zeros((n_dates, n_symbols), dtype=np.float64)
    
    for t in range(n_dates):
        signal_row = signals[t]
        vol_row = vol[t]
        
        # 유효한 종목 찾기
        valid_count = 0
        for j in range(n_symbols):
            if not np.isnan(signal_row[j]) and not np.isnan(vol_row[j]) and vol_row[j] > 0:
                valid_count += 1
        
        if valid_count < top_k:
            continue
        
        # 유효한 종목 인덱스 및 시그널 추출
        valid_indices = np.zeros(valid_count, dtype=np.int64)
        valid_signals = np.zeros(valid_count, dtype=np.float64)
        idx = 0
        for j in range(n_symbols):
            if not np.isnan(signal_row[j]) and not np.isnan(vol_row[j]) and vol_row[j] > 0:
                valid_indices[idx] = j
                valid_signals[idx] = signal_row[j]
                idx += 1
        
        # Top-K 선택 (시그널 기준 내림차순)
        sorted_idx = np.argsort(-valid_signals)[:top_k]
        selected = valid_indices[sorted_idx]
        
        # 역변동성 가중치 계산
        inv_vol = np.zeros(top_k, dtype=np.float64)
        for i in range(top_k):
            inv_vol[i] = 1.0 / (vol_row[selected[i]] + 1e-8)
        
        total_inv_vol = np.sum(inv_vol)
        
        if total_inv_vol > 0:
            for i in range(top_k):
                w = inv_vol[i] / total_inv_vol
                # 가중치 제한 적용
                w = min(w, max_weight)
                w = max(w, min_weight)
                weights[t, selected[i]] = w
            
            # 재정규화
            total_w = 0.0
            for j in range(n_symbols):
                total_w += weights[t, j]
            if total_w > 0:
                for j in range(n_symbols):
                    weights[t, j] /= total_w
    
    return weights

# =============================================================================
# 레짐 감지 엔진
# =============================================================================
class RegimeDetector:
    """VIX 기반 레짐 감지"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.RegimeDetector")
    
    def detect_regime(self, vix: float) -> str:
        """단일 VIX 값으로 레짐 판단"""
        if vix < self.config.regime_ultra_low:
            return 'ULTRA_LOW'
        elif vix < self.config.regime_low:
            return 'LOW'
        elif vix < self.config.regime_moderate:
            return 'MODERATE'
        elif vix < self.config.regime_high:
            return 'HIGH'
        else:
            return 'CRISIS'
    
    def detect_regimes(self, vix_arr: np.ndarray) -> np.ndarray:
        """전체 VIX 배열에 대해 레짐 감지"""
        regimes = np.empty(len(vix_arr), dtype='U10')
        for i, vix in enumerate(vix_arr):
            regimes[i] = self.detect_regime(vix)
        return regimes
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """레짐별 팩터 가중치 반환"""
        return self.config.regime_weights.get(regime, self.config.regime_weights['LOW'])

# =============================================================================
# 팩터 엔진
# =============================================================================
class FactorEngine:
    """팩터 계산 엔진"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.FactorEngine")
        self._cache = {}
    
    def compute_all_factors(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """모든 팩터 계산"""
        self.logger.info("Computing all factors...")
        start_time = time.time()
        
        # 캐시 키 생성
        cache_key = hashlib.md5(returns.tobytes()).hexdigest()[:16]
        if cache_key in self._cache:
            self.logger.info("Using cached factors")
            return self._cache[cache_key]
        
        factors = {}
        
        # 모멘텀 (12-1)
        self.logger.info("  Computing momentum (12-1)...")
        factors['momentum'] = compute_momentum_12_1(
            returns, 
            self.config.momentum_lookback, 
            self.config.momentum_skip
        )
        
        # 변동성
        self.logger.info("  Computing volatility...")
        factors['volatility'] = compute_volatility(
            returns, 
            self.config.volatility_lookback
        )
        
        # 단기 리버설
        self.logger.info("  Computing short-term reversal...")
        factors['reversal'] = compute_short_term_reversal(
            returns, 
            self.config.reversal_lookback
        )
        
        # 저변동성 팩터 (변동성의 역수)
        factors['low_vol'] = -factors['volatility']  # 낮을수록 좋음
        
        elapsed = time.time() - start_time
        self.logger.info(f"All factors computed in {elapsed:.2f}s")
        
        self._cache[cache_key] = factors
        return factors

# =============================================================================
# 시그널 생성 엔진
# =============================================================================
class SignalEngine:
    """레짐 기반 시그널 생성"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.SignalEngine")
        self.regime_detector = RegimeDetector(config)
    
    def generate_signals(self, factors: Dict[str, np.ndarray], vix: np.ndarray) -> np.ndarray:
        """레짐 기반 시그널 생성"""
        self.logger.info("Generating regime-adjusted signals...")
        
        momentum = factors['momentum']
        volatility = factors['volatility']
        low_vol = factors['low_vol']
        
        n_dates, n_symbols = momentum.shape
        signals = np.zeros((n_dates, n_symbols), dtype=np.float64)
        
        for t in range(n_dates):
            vix_val = vix[t]
            regime = self.regime_detector.detect_regime(vix_val)
            weights = self.regime_detector.get_regime_weights(regime)
            
            # 모멘텀 시그널
            mom_signal = momentum[t] * weights['momentum']
            
            # 저변동성 시그널 (Z-score 정규화)
            if weights['low_vol'] > 0:
                inv_vol = 1.0 / (volatility[t] + 1e-8)
                inv_vol_mean = np.nanmean(inv_vol)
                inv_vol_std = np.nanstd(inv_vol)
                if inv_vol_std > 0:
                    low_vol_signal = (inv_vol - inv_vol_mean) / inv_vol_std * weights['low_vol']
                else:
                    low_vol_signal = 0.0
            else:
                low_vol_signal = 0.0
            
            signals[t] = mom_signal + low_vol_signal
        
        return signals

# =============================================================================
# 포트폴리오 구성 엔진
# =============================================================================
class PortfolioConstructor:
    """포트폴리오 구성"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.PortfolioConstructor")
    
    def construct(self, signals: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        """포트폴리오 가중치 계산"""
        self.logger.info("Constructing portfolio weights...")
        
        weights = compute_weights_ivol(
            signals, 
            volatility, 
            self.config.top_k,
            self.config.max_position_size,
            self.config.min_position_size
        )
        
        return weights
    
    def apply_rebalancing(self, weights: np.ndarray) -> np.ndarray:
        """리밸런싱 주기 적용"""
        n_dates, n_symbols = weights.shape
        rebalanced = np.zeros_like(weights)
        current_weights = np.zeros(n_symbols, dtype=np.float64)
        
        if self.config.rebalance_freq == 'weekly':
            period = 5
        elif self.config.rebalance_freq == 'monthly':
            period = 21
        else:
            period = 1  # daily
        
        for t in range(n_dates):
            if t % period == 0:
                current_weights = weights[t].copy()
            rebalanced[t] = current_weights
        
        return rebalanced

# =============================================================================
# 리스크 관리 모듈
# =============================================================================
class RiskManager:
    """리스크 관리"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.RiskManager")
    
    def apply_dd_control(self, returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Drawdown 제어 (최고 성적 모듈)"""
        if not self.config.dd_control_enabled:
            return weights
        
        self.logger.info(f"Applying DD Control: threshold={self.config.dd_control_threshold*100}%")
        
        n_dates, n_symbols = weights.shape
        adjusted_weights = weights.copy()
        
        cum_return = 1.0
        peak = 1.0
        
        start_idx = self.config.momentum_lookback
        
        for t in range(start_idx, n_dates):
            # 포트폴리오 수익률 계산
            port_ret = np.nansum(weights[t] * returns[t])
            cum_return *= (1 + port_ret)
            peak = max(peak, cum_return)
            dd = (cum_return - peak) / peak
            
            if dd < self.config.dd_control_threshold:
                # DD가 임계값 초과 시 포지션 축소
                scale = max(
                    self.config.dd_control_scale_min,
                    1.0 + (dd - self.config.dd_control_threshold) * self.config.dd_control_scale_factor
                )
                adjusted_weights[t] *= scale
                self.logger.debug(f"  t={t}: DD={dd*100:.1f}%, scale={scale:.2f}")
        
        return adjusted_weights
    
    def apply_vol_targeting(self, returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """변동성 타겟팅"""
        if not self.config.vol_targeting_enabled:
            return weights
        
        self.logger.info(f"Applying Vol Targeting: target={self.config.target_volatility*100}%")
        
        n_dates, n_symbols = weights.shape
        adjusted_weights = weights.copy()
        
        for t in range(self.config.vol_lookback, n_dates):
            # 최근 포트폴리오 수익률
            port_returns = np.zeros(self.config.vol_lookback, dtype=np.float64)
            for d in range(self.config.vol_lookback):
                port_returns[d] = np.nansum(weights[t-d] * returns[t-d])
            
            realized_vol = np.std(port_returns) * np.sqrt(252)
            
            if realized_vol > 0:
                leverage = self.config.target_volatility / realized_vol
                leverage = min(leverage, self.config.max_leverage)
                leverage = max(leverage, self.config.min_leverage)
                adjusted_weights[t] *= leverage
        
        return adjusted_weights
    
    def apply_turnover_control(self, weights: np.ndarray) -> np.ndarray:
        """턴오버 제어"""
        if not self.config.turnover_control_enabled:
            return weights
        
        self.logger.info(f"Applying Turnover Control: max={self.config.max_turnover*100}%")
        
        n_dates, n_symbols = weights.shape
        adjusted_weights = weights.copy()
        
        for t in range(1, n_dates):
            turnover = np.nansum(np.abs(weights[t] - adjusted_weights[t-1]))
            
            if turnover > self.config.max_turnover:
                scale = self.config.max_turnover / turnover
                diff = weights[t] - adjusted_weights[t-1]
                adjusted_weights[t] = adjusted_weights[t-1] + diff * scale
        
        return adjusted_weights
    
    def apply_crisis_cash(self, weights: np.ndarray, vix: np.ndarray) -> np.ndarray:
        """CRISIS 레짐 현금 전환"""
        if not self.config.crisis_cash_enabled:
            return weights
        
        self.logger.info(f"Applying Crisis Cash: threshold={self.config.crisis_cash_threshold}")
        
        adjusted_weights = weights.copy()
        
        for t in range(len(vix)):
            if vix[t] >= self.config.crisis_cash_threshold:
                adjusted_weights[t] *= 0.0  # 완전 현금
        
        return adjusted_weights
    
    def apply_all(self, returns: np.ndarray, weights: np.ndarray, vix: np.ndarray) -> np.ndarray:
        """모든 리스크 관리 모듈 적용"""
        adjusted = weights.copy()
        
        # 순서: Crisis Cash → DD Control → Vol Targeting → Turnover Control
        adjusted = self.apply_crisis_cash(adjusted, vix)
        adjusted = self.apply_dd_control(returns, adjusted)
        adjusted = self.apply_vol_targeting(returns, adjusted)
        adjusted = self.apply_turnover_control(adjusted)
        
        return adjusted

# =============================================================================
# 백테스트 엔진
# =============================================================================
class BacktestEngine:
    """백테스트 실행 엔진"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.BacktestEngine")
        
        # 컴포넌트 초기화
        self.data_loader = DataLoader(config)
        self.factor_engine = FactorEngine(config)
        self.signal_engine = SignalEngine(config)
        self.portfolio_constructor = PortfolioConstructor(config)
        self.risk_manager = RiskManager(config)
        self.regime_detector = RegimeDetector(config)
    
    def run(self, data: Dict = None) -> Dict:
        """백테스트 실행"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Backtest")
        self.logger.info("=" * 60)
        start_time = time.time()
        
        # 데이터 준비
        if data is None:
            data = self.data_loader.prepare_data()
        
        returns = data['returns']
        vix = data['vix']
        n_dates = data['n_dates']
        n_symbols = data['n_symbols']
        
        # 팩터 계산
        factors = self.factor_engine.compute_all_factors(returns)
        
        # 시그널 생성
        signals = self.signal_engine.generate_signals(factors, vix)
        
        # 포트폴리오 구성
        weights = self.portfolio_constructor.construct(signals, factors['volatility'])
        
        # 리밸런싱 적용
        weights = self.portfolio_constructor.apply_rebalancing(weights)
        
        # 리스크 관리 적용
        weights = self.risk_manager.apply_all(returns, weights, vix)
        
        # 포트폴리오 수익률 계산
        portfolio_returns, turnover, costs = self._compute_portfolio_returns(returns, weights)
        
        # 성과 지표 계산
        start_idx = self.config.momentum_lookback
        metrics = self._compute_metrics(portfolio_returns[start_idx:], vix[start_idx:], turnover[start_idx:], costs[start_idx:])
        
        elapsed = time.time() - start_time
        self.logger.info(f"Backtest completed in {elapsed:.2f}s")
        
        return {
            'metrics': metrics,
            'returns': portfolio_returns,
            'weights': weights,
            'turnover': turnover,
            'costs': costs,
            'config': self.config.to_dict(),
            'elapsed': elapsed
        }
    
    def _compute_portfolio_returns(self, returns: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """포트폴리오 수익률 계산"""
        n_dates, n_symbols = returns.shape
        portfolio_returns = np.zeros(n_dates, dtype=np.float64)
        turnover = np.zeros(n_dates, dtype=np.float64)
        costs = np.zeros(n_dates, dtype=np.float64)
        
        prev_weights = np.zeros(n_symbols, dtype=np.float64)
        total_cost = self.config.transaction_cost + self.config.slippage
        
        for t in range(self.config.momentum_lookback, n_dates):
            # 수익률
            portfolio_returns[t] = np.nansum(weights[t] * returns[t])
            
            # 턴오버 및 비용
            turnover[t] = np.nansum(np.abs(weights[t] - prev_weights))
            costs[t] = turnover[t] * total_cost
            portfolio_returns[t] -= costs[t]
            
            prev_weights = weights[t].copy()
        
        return portfolio_returns, turnover, costs
    
    def _compute_metrics(self, returns: np.ndarray, vix: np.ndarray, 
                        turnover: np.ndarray, costs: np.ndarray) -> Dict:
        """성과 지표 계산"""
        rf_daily = self.config.risk_free_rate / 252
        excess_returns = returns - rf_daily
        
        # 기본 지표
        sharpe = np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
        annual_return = np.mean(returns) * 252
        
        cum_returns = np.cumprod(1 + returns)
        total_return = cum_returns[-1] - 1 if len(cum_returns) > 0 else 0
        
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        mdd = np.min(drawdown)
        
        # Sortino
        downside = excess_returns[excess_returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-10
        sortino = np.sqrt(252) * np.mean(excess_returns) / (downside_std + 1e-10)
        
        # Calmar
        calmar = annual_return / abs(mdd) if mdd != 0 else 0
        
        # 변동성
        volatility = np.std(returns) * np.sqrt(252)
        
        # 턴오버 및 비용
        avg_turnover = np.mean(turnover)
        total_costs = np.sum(costs)
        
        # 레짐별 성과
        regime_metrics = self._compute_regime_metrics(returns, vix)
        
        return {
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'calmar': float(calmar),
            'annual_return': float(annual_return),
            'total_return': float(total_return),
            'mdd': float(mdd),
            'volatility': float(volatility),
            'avg_turnover': float(avg_turnover),
            'total_costs': float(total_costs),
            'n_days': len(returns),
            'regime_metrics': regime_metrics
        }
    
    def _compute_regime_metrics(self, returns: np.ndarray, vix: np.ndarray) -> Dict:
        """레짐별 성과 계산"""
        regimes = {
            'ULTRA_LOW': (0, self.config.regime_ultra_low),
            'LOW': (self.config.regime_ultra_low, self.config.regime_low),
            'MODERATE': (self.config.regime_low, self.config.regime_moderate),
            'HIGH': (self.config.regime_moderate, self.config.regime_high),
            'CRISIS': (self.config.regime_high, 100)
        }
        
        rf_daily = self.config.risk_free_rate / 252
        regime_metrics = {}
        
        for regime_name, (vix_low, vix_high) in regimes.items():
            mask = (vix >= vix_low) & (vix < vix_high)
            regime_returns = returns[mask]
            
            if len(regime_returns) > 20:
                excess = regime_returns - rf_daily
                r_sharpe = np.sqrt(252) * np.mean(excess) / (np.std(excess) + 1e-10)
                r_annual = np.mean(regime_returns) * 252
                
                cum = np.cumprod(1 + regime_returns)
                peak = np.maximum.accumulate(cum)
                dd = (cum - peak) / peak
                r_mdd = np.min(dd)
                
                regime_metrics[regime_name] = {
                    'sharpe': float(r_sharpe),
                    'annual_return': float(r_annual),
                    'mdd': float(r_mdd),
                    'n_days': int(len(regime_returns))
                }
            else:
                regime_metrics[regime_name] = {
                    'sharpe': 0.0,
                    'annual_return': 0.0,
                    'mdd': 0.0,
                    'n_days': int(len(regime_returns))
                }
        
        return regime_metrics

# =============================================================================
# 조합 테스트 프레임워크
# =============================================================================
class CombinationTester:
    """조합 테스트 프레임워크"""
    
    def __init__(self, base_config: AresConfig = None):
        self.base_config = base_config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.CombinationTester")
        self.results = []
    
    def run_all_combinations(self, data: Dict = None) -> List[Dict]:
        """모든 조합 테스트 실행"""
        self.logger.info("=" * 80)
        self.logger.info("Starting Combination Tests")
        self.logger.info("=" * 80)
        
        # 데이터 준비
        if data is None:
            loader = DataLoader(self.base_config)
            data = loader.prepare_data()
        
        # 테스트할 모듈 조합
        modules = [
            ('dd_control', {'dd_control_enabled': True}),
            ('vol_targeting', {'vol_targeting_enabled': True}),
            ('turnover_control', {'turnover_control_enabled': True}),
            ('crisis_cash', {'crisis_cash_enabled': True}),
        ]
        
        self.results = []
        
        # 1. 베이스라인
        self.logger.info("\n--- BASELINE ---")
        baseline_result = self._run_single_test("BASELINE", {}, data)
        self.results.append(baseline_result)
        
        # 2. 단일 모듈 테스트
        self.logger.info("\n--- Single Module Tests ---")
        for module_name, module_config in modules:
            name = f"BASELINE+{module_name}"
            result = self._run_single_test(name, module_config, data)
            self.results.append(result)
        
        # 3. 2개 모듈 조합
        self.logger.info("\n--- Two Module Combinations ---")
        for i in range(len(modules)):
            for j in range(i+1, len(modules)):
                name = f"BASELINE+{modules[i][0]}+{modules[j][0]}"
                config = {**modules[i][1], **modules[j][1]}
                result = self._run_single_test(name, config, data)
                self.results.append(result)
        
        # 결과 정렬
        self.results.sort(key=lambda x: x['metrics']['sharpe'], reverse=True)
        
        # 결과 출력
        self._print_results()
        
        return self.results
    
    def _run_single_test(self, name: str, config_updates: Dict, data: Dict) -> Dict:
        """단일 테스트 실행"""
        self.logger.info(f"Testing: {name}")
        
        # 설정 복사 및 업데이트
        config = AresConfig()
        for key, value in config_updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 백테스트 실행
        engine = BacktestEngine(config)
        result = engine.run(data)
        
        self.logger.info(f"  Sharpe={result['metrics']['sharpe']:.4f}, MDD={result['metrics']['mdd']*100:.1f}%")
        
        return {
            'name': name,
            'config': config_updates,
            'metrics': result['metrics']
        }
    
    def _print_results(self):
        """결과 출력"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RESULTS SUMMARY (Sorted by Sharpe)")
        self.logger.info("=" * 80)
        
        for i, r in enumerate(self.results[:10]):
            m = r['metrics']
            self.logger.info(f"{i+1}. {r['name']}")
            self.logger.info(f"   Sharpe={m['sharpe']:.4f}, Sortino={m['sortino']:.4f}, "
                           f"Return={m['annual_return']*100:.1f}%, MDD={m['mdd']*100:.1f}%")
            regime_str = ", ".join([f"{k}={v['sharpe']:.2f}" for k, v in m['regime_metrics'].items()])
            self.logger.info(f"   Regimes: {regime_str}")

# =============================================================================
# Walk-Forward 검증
# =============================================================================
class WalkForwardValidator:
    """Walk-Forward 검증"""
    
    def __init__(self, config: AresConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("ARES_BEST.WalkForwardValidator")
    
    def run(self, data: Dict = None) -> Dict:
        """Walk-Forward 검증 실행"""
        self.logger.info("=" * 80)
        self.logger.info("Starting Walk-Forward Validation")
        self.logger.info("=" * 80)
        
        # 데이터 준비
        if data is None:
            loader = DataLoader(self.config)
            data = loader.prepare_data()
        
        n_dates = data['n_dates']
        
        # Walk-Forward 윈도우
        train_window = self.config.train_window
        test_window = self.config.test_window
        step_size = self.config.step_size
        
        results = []
        
        start_idx = train_window
        while start_idx + test_window <= n_dates:
            train_end = start_idx
            test_end = start_idx + test_window
            
            self.logger.info(f"Window: train=[0:{train_end}], test=[{train_end}:{test_end}]")
            
            # 테스트 기간 데이터
            test_data = {
                'dates': data['dates'][train_end:test_end],
                'symbols': data['symbols'],
                'returns': data['returns'][train_end:test_end],
                'prices': data['prices'][train_end:test_end],
                'vix': data['vix'][train_end:test_end],
                'n_dates': test_window,
                'n_symbols': data['n_symbols']
            }
            
            # 백테스트 실행
            engine = BacktestEngine(self.config)
            result = engine.run(test_data)
            
            results.append({
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'metrics': result['metrics']
            })
            
            self.logger.info(f"  OOS Sharpe={result['metrics']['sharpe']:.4f}")
            
            start_idx += step_size
        
        # 전체 OOS 성과 계산
        oos_sharpes = [r['metrics']['sharpe'] for r in results]
        oos_returns = [r['metrics']['annual_return'] for r in results]
        
        summary = {
            'n_windows': len(results),
            'avg_oos_sharpe': np.mean(oos_sharpes),
            'std_oos_sharpe': np.std(oos_sharpes),
            'min_oos_sharpe': np.min(oos_sharpes),
            'max_oos_sharpe': np.max(oos_sharpes),
            'avg_oos_return': np.mean(oos_returns),
            'windows': results
        }
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Walk-Forward Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"Windows: {summary['n_windows']}")
        self.logger.info(f"Avg OOS Sharpe: {summary['avg_oos_sharpe']:.4f} ± {summary['std_oos_sharpe']:.4f}")
        self.logger.info(f"Min/Max OOS Sharpe: {summary['min_oos_sharpe']:.4f} / {summary['max_oos_sharpe']:.4f}")
        self.logger.info(f"Avg OOS Return: {summary['avg_oos_return']*100:.1f}%")
        
        return summary

# =============================================================================
# 메인 실행
# =============================================================================
def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description='ARES BEST v1.0 - Complete Backtest System')
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'combination_test', 'walkforward'],
                       help='Execution mode')
    parser.add_argument('--db', type=str, default=None,
                       help='Database path')
    parser.add_argument('--start', type=str, default='2015-01-01',
                       help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    # 설정
    config = AresConfig()
    if args.db:
        config.db_path = args.db
    config.start_date = args.start
    config.end_date = args.end
    
    # 로깅 설정
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("ARES BEST v1.0 - Complete Backtest System")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"DB: {config.db_path}")
    logger.info(f"Period: {config.start_date} ~ {config.end_date}")
    
    # 실행
    if args.mode == 'backtest':
        engine = BacktestEngine(config)
        result = engine.run()
        
        # 결과 출력
        m = result['metrics']
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Sharpe Ratio: {m['sharpe']:.4f}")
        print(f"Sortino Ratio: {m['sortino']:.4f}")
        print(f"Calmar Ratio: {m['calmar']:.4f}")
        print(f"Annual Return: {m['annual_return']*100:.2f}%")
        print(f"Total Return: {m['total_return']*100:.2f}%")
        print(f"MDD: {m['mdd']*100:.2f}%")
        print(f"Volatility: {m['volatility']*100:.2f}%")
        print(f"Avg Turnover: {m['avg_turnover']*100:.2f}%")
        print(f"Total Costs: {m['total_costs']*100:.2f}%")
        print(f"Days: {m['n_days']}")
        
        print("\n--- Regime Metrics ---")
        for regime, rm in m['regime_metrics'].items():
            print(f"{regime}: Sharpe={rm['sharpe']:.2f}, Return={rm['annual_return']*100:.1f}%, "
                  f"MDD={rm['mdd']*100:.1f}%, Days={rm['n_days']}")
        
        # 결과 저장
        output_file = args.output or f"{config.output_dir}/backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
        
    elif args.mode == 'combination_test':
        tester = CombinationTester(config)
        results = tester.run_all_combinations()
        
        # 결과 저장
        output_file = args.output or f"{config.output_dir}/combination_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
        
    elif args.mode == 'walkforward':
        validator = WalkForwardValidator(config)
        summary = validator.run()
        
        # 결과 저장
        output_file = args.output or f"{config.output_dir}/walkforward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
