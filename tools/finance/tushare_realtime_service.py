#!/usr/bin/env python3
"""
Tushare Realtime Service - A股实时行情服务
基于Tushare Pro API realtime_quote 接口（实时报价）和 rt_min 接口（分钟K线）
"""
import logging
import asyncio
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("tushare not installed. Service will not be available.")


# Valid frequency options
FreqType = Literal["1MIN", "5MIN", "15MIN", "30MIN", "60MIN"]


class TushareRealtimeService:
    """
    A股实时分钟行情服务
    
    基于 Tushare Pro API rt_min 接口，提供：
    - 1/5/15/30/60 分钟级别实时行情
    - 支持多股票批量查询
    - 支持股票名称/代码双向查询
    """
    
    FREQ_OPTIONS = ["1MIN", "5MIN", "15MIN", "30MIN", "60MIN"]
    
    def __init__(self, token: Optional[str] = None):
        self.name = "Tushare Realtime Service"
        self.token = token or os.getenv('TUSHARE_TOKEN')
        self.pro = None
        
        # Cache settings
        self.cache_dir = Path("data/cache/tushare")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expire_hours = 24  # Cache expiration for stock basic info
        
        if not self.token:
            logging.warning("TUSHARE_TOKEN not found in environment variables")

        if TUSHARE_AVAILABLE:
            try:
                if self.token:
                    ts.set_token(self.token)
                    self.pro = ts.pro_api(self.token)
                    logging.info("TushareRealtimeService initialized")
                else:
                    logging.warning("TushareRealtimeService initialized without token")
            except Exception as e:
                logging.error(f"Failed to initialize Tushare API: {str(e)}")
    
    def _load_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if valid"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    cache_time = cache_data.get('timestamp', 0)
                    if time.time() - cache_time < self.cache_expire_hours * 3600:
                        logging.debug(f"✅ Loaded {cache_key} from cache")
                        return cache_data.get('data')
            except Exception as e:
                logging.warning(f"Cache read error: {e}")
        return None
    
    def _save_cache(self, cache_key: str, data: Any):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            cache_data = {
                'timestamp': time.time(),
                'data': data
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logging.debug(f"💾 Saved {cache_key} to cache")
        except Exception as e:
            logging.warning(f"Cache write error: {e}")
    
    async def _get_stock_basic(self) -> Optional[pd.DataFrame]:
        """Get stock basic info with cache support"""
        if not self.pro:
            return None
        
        # Try load from cache first
        cached_data = self._load_cache('stock_basic')
        if cached_data:
            try:
                return pd.DataFrame(cached_data)
            except Exception:
                pass
        
        # Fetch from API
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None, 
                lambda: self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
            )
            
            if df is not None and not df.empty:
                self._save_cache('stock_basic', df.to_dict('records'))
                return df
        except Exception as e:
            logging.error(f"Error fetching stock_basic: {e}")
        
        return None

    async def _get_code_by_name(self, names: List[str]) -> Dict[str, str]:
        """Get stock codes by names"""
        df = await self._get_stock_basic()
        if df is None or df.empty:
            return {}
        
        name_map = {}
        for name in names:
            row = df[df['name'] == name]
            if not row.empty:
                name_map[name] = row.iloc[0]['ts_code']
        
        return name_map

    async def _get_name_by_code(self, codes: List[str]) -> Dict[str, str]:
        """Get stock names by codes"""
        df = await self._get_stock_basic()
        if df is None or df.empty:
            return {}
        
        code_map = {}
        for code in codes:
            row = df[df['ts_code'] == code]
            if not row.empty:
                code_map[code] = row.iloc[0]['name']
        
        return code_map

    def _validate_freq(self, freq: str) -> str:
        """Validate and normalize frequency parameter"""
        freq_upper = freq.upper()
        if freq_upper not in self.FREQ_OPTIONS:
            raise ValueError(f"Invalid freq: {freq}. Must be one of {self.FREQ_OPTIONS}")
        return freq_upper

    async def _get_realtime_minute(
        self, 
        ts_codes: str, 
        freq: str = "1MIN"
    ) -> Dict[str, Any]:
        """
        获取A股实时分钟行情数据 (内部方法)
        
        Args:
            ts_codes: 股票代码，支持多个逗号分隔 (e.g., '600000.SH' 或 '600000.SH,000001.SZ')
            freq: 分钟周期，支持 5MIN/15MIN/30MIN/60MIN
            
        Returns:
            Dict: 包含实时行情数据
                - success (bool): 是否成功
                - data (list): 行情数据列表
                - metadata (dict): 元数据
        """
        if not self.pro:
            return {"success": False, "error": "Tushare not initialized"}
        
        try:
            freq = self._validate_freq(freq)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.pro.rt_min(ts_code=ts_codes, freq=freq)
            )
            
            if df is None or df.empty:
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "ts_codes": ts_codes,
                        "freq": freq,
                        "count": 0,
                        "message": "No data returned (market may be closed)"
                    }
                }
            
            # Get stock names for enrichment
            codes_list = ts_codes.split(',')
            code_name_map = await self._get_name_by_code(codes_list)
            
            # Process data
            records = []
            for _, row in df.iterrows():
                code = row['ts_code']
                record = {
                    "ts_code": code,
                    "name": code_name_map.get(code, ""),
                    "time": str(row['time']) if pd.notna(row['time']) else None,
                    "open": float(row['open']) if pd.notna(row['open']) else None,
                    "close": float(row['close']) if pd.notna(row['close']) else None,
                    "high": float(row['high']) if pd.notna(row['high']) else None,
                    "low": float(row['low']) if pd.notna(row['low']) else None,
                    "vol": float(row['vol']) if pd.notna(row['vol']) else None,
                    "amount": float(row['amount']) if pd.notna(row['amount']) else None,
                }
                records.append(record)
            
            return {
                "success": True,
                "data": records,
                "metadata": {
                    "ts_codes": ts_codes,
                    "freq": freq,
                    "count": len(records),
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            logging.error(f"Error fetching realtime minute data: {e}")
            return {"success": False, "error": str(e)}

    async def _get_minute_by_name(
        self, 
        stock_names: str, 
        freq: str = "1MIN"
    ) -> Dict[str, Any]:
        """
        按股票名称获取分钟K线行情 (内部方法)
        
        Args:
            stock_names: 股票名称，支持多个逗号分隔
            freq: 分钟周期，支持 1MIN/5MIN/15MIN/30MIN/60MIN
            
        Returns:
            Dict: 包含分钟K线数据
        """
        if not self.pro:
            return {"success": False, "error": "Tushare not initialized"}
        
        names_list = [n.strip() for n in stock_names.split(',') if n.strip()]
        if not names_list:
            return {"success": False, "error": "No stock names provided"}
        
        name_code_map = await self._get_code_by_name(names_list)
        
        if not name_code_map:
            return {"success": False, "error": f"Could not find codes for: {stock_names}"}
        
        not_found = [n for n in names_list if n not in name_code_map]
        
        ts_codes = ",".join(name_code_map.values())
        result = await self._get_realtime_minute(ts_codes, freq)
        
        if not_found and result.get("success"):
            result["metadata"]["not_found_names"] = not_found
        
        return result

    async def _get_realtime_quote(
        self, 
        ts_codes: str,
        src: str = "sina"
    ) -> Dict[str, Any]:
        """
        获取实时报价数据 (内部方法)
        基于 realtime_quote 接口，返回真正的实时价格
        
        Args:
            ts_codes: 股票代码，支持多个逗号分隔 (sina源最多50个，dc源只支持单个)
            src: 数据源 sina-新浪(默认) dc-东方财富
            
        Returns:
            Dict: 包含实时报价数据
        """
        if not TUSHARE_AVAILABLE:
            return {"success": False, "error": "Tushare not available"}
        
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                lambda: ts.realtime_quote(ts_code=ts_codes, src=src)
            )
            
            if df is None or df.empty:
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "ts_codes": ts_codes,
                        "src": src,
                        "count": 0,
                        "message": "No data returned (market may be closed)"
                    }
                }
            
            # Process data
            records = []
            for _, row in df.iterrows():
                record = {
                    "ts_code": str(row.get('TS_CODE', '')),
                    "name": str(row.get('NAME', '')),
                    "price": float(row['PRICE']) if pd.notna(row.get('PRICE')) else None,
                    "change": round(float(row['PRICE']) - float(row['PRE_CLOSE']), 2) if pd.notna(row.get('PRICE')) and pd.notna(row.get('PRE_CLOSE')) else None,
                    "pct_change": round((float(row['PRICE']) - float(row['PRE_CLOSE'])) / float(row['PRE_CLOSE']) * 100, 2) if pd.notna(row.get('PRICE')) and pd.notna(row.get('PRE_CLOSE')) and float(row.get('PRE_CLOSE', 0)) != 0 else None,
                    "open": float(row['OPEN']) if pd.notna(row.get('OPEN')) else None,
                    "high": float(row['HIGH']) if pd.notna(row.get('HIGH')) else None,
                    "low": float(row['LOW']) if pd.notna(row.get('LOW')) else None,
                    "pre_close": float(row['PRE_CLOSE']) if pd.notna(row.get('PRE_CLOSE')) else None,
                    "volume": int(row['VOLUME']) if pd.notna(row.get('VOLUME')) else None,
                    "amount": float(row['AMOUNT']) if pd.notna(row.get('AMOUNT')) else None,
                    "bid": float(row['BID']) if pd.notna(row.get('BID')) else None,
                    "ask": float(row['ASK']) if pd.notna(row.get('ASK')) else None,
                    "date": str(row.get('DATE', '')),
                    "time": str(row.get('TIME', '')),
                }
                records.append(record)
            
            return {
                "success": True,
                "data": records,
                "metadata": {
                    "ts_codes": ts_codes,
                    "src": src,
                    "count": len(records),
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            logging.error(f"Error fetching realtime quote: {e}")
            return {"success": False, "error": str(e)}

    async def get_realtime_by_name(
        self, 
        stock_names: str
    ) -> Dict[str, Any]:
        """
        按股票名称获取实时报价（MCP工具）
        基于 realtime_quote 接口，返回真正的实时价格
        
        Args:
            stock_names: 股票名称，支持多个逗号分隔 (e.g., '浦发银行' 或 '浦发银行,平安银行')，最多50个
            
        Returns:
            Dict: 包含实时报价数据，包括：
                - price: 当前价格
                - change: 涨跌额
                - pct_change: 涨跌幅(%)
                - open/high/low: 今日开盘/最高/最低
                - pre_close: 昨收价
                - volume: 成交量
                - amount: 成交额
                - bid/ask: 买一/卖一价
        """
        names_list = [n.strip() for n in stock_names.split(',') if n.strip()]
        if not names_list:
            return {"success": False, "error": "No stock names provided"}
        
        if len(names_list) > 50:
            return {"success": False, "error": "Maximum 50 stocks per request"}
        
        # Convert names to codes
        name_code_map = await self._get_code_by_name(names_list)
        
        if not name_code_map:
            return {"success": False, "error": f"Could not find codes for: {stock_names}"}
        
        # Find names that couldn't be resolved
        not_found = [n for n in names_list if n not in name_code_map]
        
        ts_codes = ",".join(name_code_map.values())
        result = await self._get_realtime_quote(ts_codes, src="sina")
        
        if not_found and result.get("success"):
            result["metadata"]["not_found_names"] = not_found
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def test():
        service = TushareRealtimeService()
        
        # Test: Get realtime quote by name (main tool)
        print("\n=== Test: Realtime quote by name ===")
        result = await service.get_realtime_by_name('浦发银行,平安银行')
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(test())

