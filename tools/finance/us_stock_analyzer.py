#!/usr/bin/env python3
"""
US Stock Analyzer Service
Comprehensive US stock market analysis integrating multiple data sources.
Provides unified access to fundamental, technical, and sentiment indicators.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from .finnhub_service import FinnhubService
from .massive_service import MassiveService
from .fiscal_service import FiscalService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class USStockAnalyzerService:
    """
    US Stock Comprehensive Analyzer Service.
    
    Integrates multiple data sources (Finnhub, Polygon.io) to provide:
    - Fundamental Analysis: Valuation, financials, ownership, executive info, earnings calls
    - Technical Analysis: Technical indicators (SMA, EMA, MACD, RSI), patterns, support/resistance
    - Sentiment Analysis: News sentiment, social media, analyst ratings, insider activity
    - Comprehensive Analysis: All-in-one report combining all dimensions
    
    Usage Example:
        analyzer = USStockAnalyzerService()
        
        # Get fundamental analysis
        fundamental = await analyzer.get_fundamental_analysis('AAPL', '2024-11-01', '2024-12-10')
        
        # Get technical analysis
        technical = await analyzer.get_technical_analysis('AAPL', '2024-11-01', '2024-12-10')
        
        # Get sentiment analysis
        sentiment = await analyzer.get_sentiment_analysis('AAPL', '2024-11-01', '2024-12-10')
        
        # Get comprehensive report (all dimensions)
        comprehensive = await analyzer.get_comprehensive_analysis('AAPL', '2024-11-01', '2024-12-10')
    """
    
    def __init__(self):
        self.name = "US Stock Analyzer"
        self.version = "1.0.0"
        
        # Initialize underlying services
        self.finnhub = FinnhubService()
        self.massive = MassiveService()
        self.fiscal = FiscalService()
        
        if not self.finnhub.api_key:
            logger.warning("Finnhub API key not found. Some features will be unavailable.")
        
        if not self.massive.api_key:
            logger.warning("Polygon API key not found. Technical indicators will be unavailable.")

        if not self.fiscal.api_key:
            logger.warning("Fiscal.ai API key not found. KPI features will be unavailable.")
        
        logger.info(f"USStockAnalyzerService v{self.version} initialized")
    
    async def get_company_kpis(self, symbol: str) -> Dict[str, Any]:
        """
        Get company segments and KPIs from Fiscal.ai.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'MSFT').
            
        Returns:
            Dict: Segments and KPIs data.
        """
        logger.info(f"Fetching KPIs for {symbol}")
        return await self.fiscal._get_segments_and_kpis(symbol)

    async def get_fundamental_analysis(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get comprehensive fundamental analysis for a US stock.
        
        Includes:
        - Company Profile: Basic info, industry, market cap, description, peers
        - Financial Metrics: Valuation ratios (P/E, P/B, P/S), profitability, financial health
        - Earnings Data: Earnings surprises, EPS history
        - Dividend Information: Dividend history and yield
        - Ownership Structure: Institutional ownership, fund ownership, insider transactions
        - Executive Information: Company executives and their compensation
        - SEC Filings: Recent regulatory filings (10-K, 10-Q, 8-K)
        - Earnings Call Transcripts: Conference call summaries and full transcripts
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA').
            start_date: Analysis start date in 'YYYY-MM-DD' format.
            end_date: Analysis end date in 'YYYY-MM-DD' format.
        
        Returns:
            Dict: Structured fundamental data with the following keys:
                - success (bool): Whether the analysis succeeded
                - data (dict): Contains all fundamental indicators
                    - company_profile: Company overview and peer comparisons
                    - financial_metrics: Key financial ratios and metrics
                    - ownership: Institutional, fund, and insider ownership data
                    - executives: Executive team information
                    - sec_filings: Recent SEC filing records
                    - earnings_calls: Transcripts and summaries of earnings calls
                - metadata (dict): Query metadata (symbol, date range, timestamp)
        
        Example Response Structure:
            {
                "success": True,
                "data": {
                    "company_profile": {...},
                    "financial_metrics": {...},
                    "ownership": {...},
                    "executives": {...},
                    "sec_filings": [...],
                    "earnings_calls": {...}
                },
                "metadata": {...}
            }
        """
        logger.info(f"Starting fundamental analysis for {symbol}")
        
        data = {}
        
        try:
            # 1. Company Profile & Peers
            profile_res = await self.finnhub._get_company_profile(symbol)
            if profile_res.get("success"):
                data['company_profile'] = profile_res.get("data")
            
            # 2. Financial Data (metrics, earnings, dividends)
            financial_res = await self.finnhub._get_financials(symbol, start_date, end_date)
            if financial_res.get("success"):
                data['financial_metrics'] = financial_res.get("data")
            
            # 3. Ownership Data
            ownership_data = {}
            
            # Institutional Ownership
            inst_res = await self._fetch_institutional_ownership(symbol, start_date, end_date)
            if inst_res.get("success"):
                ownership_data['institutional'] = inst_res.get("data")
            
            # Fund Ownership
            fund_res = await self._fetch_fund_ownership(symbol)
            if fund_res.get("success"):
                ownership_data['fund'] = fund_res.get("data")
            
            # Insider Transactions
            insider_res = await self._fetch_insider_transactions(symbol, start_date, end_date)
            if insider_res.get("success"):
                ownership_data['insider_transactions'] = insider_res.get("data")
            
            data['ownership'] = ownership_data
            
            # 4. Executive Information
            exec_res = await self._fetch_company_executives(symbol)
            if exec_res.get("success"):
                data['executives'] = exec_res.get("data")
            
            # 5. SEC Filings
            filings_res = await self._fetch_sec_filings(symbol, start_date, end_date)
            if filings_res.get("success"):
                data['sec_filings'] = filings_res.get("data")
            
            # 6. Earnings Call Transcripts
            transcripts_data = {}
            transcripts_list_res = await self._fetch_transcripts_list(symbol)
            if transcripts_list_res.get("success"):
                transcripts_data['list'] = transcripts_list_res.get("data")
                
                # Fetch the most recent transcript content
                transcript_list = transcripts_list_res.get("data")
                if transcript_list and isinstance(transcript_list, list) and len(transcript_list) > 0:
                    latest_id = transcript_list[0].get('id') if isinstance(transcript_list[0], dict) else None
                    if latest_id:
                        transcript_res = await self._fetch_transcripts(latest_id)
                        if transcript_res.get("success"):
                            transcripts_data['latest_content'] = transcript_res.get("data")
            
            data['earnings_calls'] = transcripts_data
            
            # 7. Segments and KPIs
            kpi_res = await self.get_company_kpis(symbol)
            if kpi_res.get("success"):
                data['segments_kpis'] = kpi_res.get("data")

            return {
                "success": True,
                "data": data,
                "metadata": {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analysis_type": "fundamental",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    async def get_technical_analysis(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get comprehensive technical analysis for a US stock.
        
        Includes:
        - Moving Averages: SMA (20, 50, 200-day), EMA (12, 26-day)
        - Momentum Indicators: RSI (14-day), MACD (12,26,9)
        - Pattern Recognition: Chart patterns (head & shoulders, triangles, flags, etc.)
        - Support & Resistance: Key price levels
        - Aggregate Signals: Overall buy/sell/hold signals from multiple indicators
        - Current Quote: Real-time price, volume, and market data
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA').
            start_date: Analysis start date in 'YYYY-MM-DD' format.
            end_date: Analysis end date in 'YYYY-MM-DD' format.
        
        Returns:
            Dict: Structured technical data with the following keys:
                - success (bool): Whether the analysis succeeded
                - data (dict): Contains all technical indicators
                    - quote: Current price and market data
                    - moving_averages: SMA and EMA values
                    - momentum: RSI and MACD indicators
                    - patterns: Detected chart patterns
                    - support_resistance: Key price levels
                    - aggregate_signals: Overall technical signals
                - metadata (dict): Query metadata
        
        Example Response Structure:
            {
                "success": True,
                "data": {
                    "quote": {"c": 180.5, "h": 182.0, ...},
                    "moving_averages": {
                        "sma_20": [...],
                        "sma_50": [...],
                        "ema_12": [...]
                    },
                    "momentum": {
                        "rsi": [...],
                        "macd": [...]
                    },
                    "patterns": {...},
                    "support_resistance": {...},
                    "aggregate_signals": {...}
                },
                "metadata": {...}
            }
        """
        logger.info(f"Starting technical analysis for {symbol}")
        
        data = {}
        
        try:
            # 1. Current Quote
            quote_res = await self.finnhub._get_market_data(symbol, start_date, end_date)
            if quote_res.get("success"):
                data['quote'] = quote_res.get("data", {}).get('quote')
            
            # 2. Moving Averages
            ma_data = await self._calculate_moving_averages(symbol)
            data['moving_averages'] = ma_data
            
            # 3. Momentum Indicators
            momentum_data = await self._calculate_momentum_indicators(symbol)
            data['momentum'] = momentum_data
            
            # 4. Pattern Recognition
            pattern_res = await self._fetch_pattern_recognition(symbol)
            if pattern_res.get("success"):
                data['patterns'] = pattern_res.get("data")
            
            # 5. Support & Resistance
            sr_res = await self._fetch_support_resistance(symbol)
            if sr_res.get("success"):
                data['support_resistance'] = sr_res.get("data")
            
            # 6. Aggregate Technical Signals
            agg_res = await self._fetch_aggregate_indicator(symbol)
            if agg_res.get("success"):
                data['aggregate_signals'] = agg_res.get("data")
            
            return {
                "success": True,
                "data": data,
                "metadata": {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analysis_type": "technical",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    async def get_sentiment_analysis(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis for a US stock.
        
        Includes:
        - News Sentiment: Sentiment scores from financial news articles
        - Social Media Sentiment: Reddit and Twitter sentiment metrics
        - Insider Sentiment: Aggregated insider trading sentiment (MSPR)
        - Analyst Ratings: Recommendation trends (buy/hold/sell distribution)
        - Rating Changes: Recent analyst upgrades and downgrades
        - Company News: Recent news articles and headlines
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA').
            start_date: Analysis start date in 'YYYY-MM-DD' format.
            end_date: Analysis end date in 'YYYY-MM-DD' format.
        
        Returns:
            Dict: Structured sentiment data with the following keys:
                - success (bool): Whether the analysis succeeded
                - data (dict): Contains all sentiment indicators
                    - news_sentiment: News sentiment scores and article count
                    - social_sentiment: Social media sentiment from Reddit/Twitter
                    - insider_sentiment: Insider trading sentiment (MSPR)
                    - analyst_ratings: Current analyst recommendation distribution
                    - rating_changes: Recent upgrades and downgrades
                    - recent_news: Latest news articles
                - metadata (dict): Query metadata
        
        Example Response Structure:
            {
                "success": True,
                "data": {
                    "news_sentiment": {
                        "buzz": {...},
                        "sentiment": {...}
                    },
                    "social_sentiment": {...},
                    "insider_sentiment": {...},
                    "analyst_ratings": {...},
                    "rating_changes": [...],
                    "recent_news": [...]
                },
                "metadata": {...}
            }
        """
        logger.info(f"Starting sentiment analysis for {symbol}")
        
        data = {}
        
        try:
            # 1. News Sentiment
            news_sent_res = await self._fetch_news_sentiment(symbol)
            if news_sent_res.get("success"):
                data['news_sentiment'] = news_sent_res.get("data")
            
            # 2. Social Media Sentiment
            social_res = await self._fetch_social_sentiment(symbol)
            if social_res.get("success"):
                data['social_sentiment'] = social_res.get("data")
            
            # 3. Insider Sentiment
            insider_sent_res = await self._fetch_insider_sentiment(symbol, start_date, end_date)
            if insider_sent_res.get("success"):
                data['insider_sentiment'] = insider_sent_res.get("data")
            
            # 4. Analyst Ratings & Trends
            ratings_res = await self.finnhub._get_estimates_and_analysis(symbol, start_date, end_date)
            if ratings_res.get("success"):
                data['analyst_ratings'] = ratings_res.get("data")
            
            # 5. Upgrade/Downgrade Events
            upgrade_res = await self._fetch_upgrade_downgrade(symbol, start_date, end_date)
            if upgrade_res.get("success"):
                data['rating_changes'] = upgrade_res.get("data")
            
            # 6. Recent News
            news_res = await self.finnhub._get_sentiment_and_news(symbol, start_date, end_date)
            if news_res.get("success"):
                data['recent_news'] = news_res.get("data", {}).get('news')
            
            return {
                "success": True,
                "data": data,
                "metadata": {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analysis_type": "sentiment",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    async def get_comprehensive_analysis(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get comprehensive multi-dimensional analysis combining fundamental, technical, and sentiment data.
        
        This is the all-in-one analysis function that provides a complete picture of a stock
        by aggregating data from all other analysis dimensions. Similar to the `get_daban_indicators`
        function in the A-share analysis service.
        
        Includes all indicators from:
        - get_fundamental_analysis(): Company profile, financials, ownership, executives, filings, earnings calls
        - get_technical_analysis(): Technical indicators, patterns, support/resistance, signals
        - get_sentiment_analysis(): News/social/insider sentiment, analyst ratings, rating changes
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA').
            start_date: Analysis start date in 'YYYY-MM-DD' format.
            end_date: Analysis end date in 'YYYY-MM-DD' format.
        
        Returns:
            Dict: Complete analysis report with the following structure:
                - success (bool): Whether the analysis succeeded
                - data (dict): Contains all dimensions
                    - fundamental: All fundamental indicators
                    - technical: All technical indicators
                    - sentiment: All sentiment indicators
                    - summary: Key highlights and overall assessment
                - metadata (dict): Comprehensive metadata
        
        Example Response Structure:
            {
                "success": True,
                "data": {
                    "fundamental": {...},
                    "technical": {...},
                    "sentiment": {...},
                    "summary": {
                        "overall_score": 7.5,
                        "recommendation": "BUY",
                        "key_highlights": [...]
                    }
                },
                "metadata": {
                    "symbol": "AAPL",
                    "analysis_date": "2024-12-10",
                    "timestamp": "..."
                }
            }
        """
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        try:
            # Fetch all dimensions in parallel for better performance
            fundamental_task = self.get_fundamental_analysis(symbol, start_date, end_date)
            technical_task = self.get_technical_analysis(symbol, start_date, end_date)
            sentiment_task = self.get_sentiment_analysis(symbol, start_date, end_date)
            
            fundamental_res, technical_res, sentiment_res = await asyncio.gather(
                fundamental_task,
                technical_task,
                sentiment_task,
                return_exceptions=True
            )
            
            # Build comprehensive report
            data = {}
            
            if isinstance(fundamental_res, dict) and fundamental_res.get("success"):
                data['fundamental'] = fundamental_res.get("data")
            
            if isinstance(technical_res, dict) and technical_res.get("success"):
                data['technical'] = technical_res.get("data")
            
            if isinstance(sentiment_res, dict) and sentiment_res.get("success"):
                data['sentiment'] = sentiment_res.get("data")
            
            # Generate summary
            summary = self._generate_analysis_summary(data)
            data['summary'] = summary
            
            return {
                "success": True,
                "data": data,
                "metadata": {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analysis_type": "comprehensive",
                    "timestamp": datetime.now().isoformat(),
                    "dimensions_analyzed": list(data.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    # ============= Private Helper Methods =============
    
    async def _fetch_transcripts(self, transcript_id: str) -> Dict[str, Any]:
        """Fetch earnings call transcript content."""
        return await self.finnhub._fetch_transcripts(transcript_id)
    
    async def _fetch_transcripts_list(self, symbol: str) -> Dict[str, Any]:
        """Fetch list of available earnings call transcripts."""
        return await self.finnhub._fetch_transcripts_list(symbol)
    
    async def _fetch_company_executives(self, symbol: str) -> Dict[str, Any]:
        """Fetch company executive information."""
        return await self.finnhub._fetch_company_executives(symbol)
    
    async def _fetch_sec_filings(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch SEC filings."""
        return await self.finnhub._fetch_sec_filings(symbol, start_date, end_date)
    
    async def _fetch_insider_transactions(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch insider trading transactions."""
        return await self.finnhub._fetch_insider_transactions(symbol, start_date, end_date)
    
    async def _fetch_institutional_ownership(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch institutional ownership data."""
        return await self.finnhub._fetch_institutional_ownership(symbol, start_date, end_date)
    
    async def _fetch_fund_ownership(self, symbol: str) -> Dict[str, Any]:
        """Fetch mutual fund ownership data."""
        return await self.finnhub._fetch_fund_ownership(symbol)
    
    async def _fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch news sentiment scores."""
        return await self.finnhub._fetch_news_sentiment(symbol)
    
    async def _fetch_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch social media sentiment."""
        return await self.finnhub._fetch_social_sentiment(symbol)
    
    async def _fetch_insider_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch insider trading sentiment."""
        return await self.finnhub._fetch_insider_sentiment(symbol, start_date, end_date)
    
    async def _fetch_upgrade_downgrade(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch analyst rating changes."""
        return await self.finnhub._fetch_upgrade_downgrade(symbol, start_date, end_date)
    
    async def _fetch_pattern_recognition(self, symbol: str) -> Dict[str, Any]:
        """Fetch technical pattern recognition."""
        return await self.finnhub._fetch_pattern_recognition(symbol, 'D')
    
    async def _fetch_support_resistance(self, symbol: str) -> Dict[str, Any]:
        """Fetch support and resistance levels."""
        return await self.finnhub._fetch_support_resistance(symbol, 'D')
    
    async def _fetch_aggregate_indicator(self, symbol: str) -> Dict[str, Any]:
        """Fetch aggregate technical signals."""
        return await self.finnhub._fetch_aggregate_indicator(symbol, 'D')
    
    async def _calculate_moving_averages(self, symbol: str) -> Dict[str, Any]:
        """Calculate multiple moving averages (SMA 20/50/200, EMA 12/26)."""
        ma_data = {}
        
        try:
            # SMA 20-day
            sma20_res = await self.massive._get_sma(symbol, timespan='day', window=20, limit=10)
            if sma20_res.get("success"):
                ma_data['sma_20'] = sma20_res.get("data")
            
            # SMA 50-day
            sma50_res = await self.massive._get_sma(symbol, timespan='day', window=50, limit=10)
            if sma50_res.get("success"):
                ma_data['sma_50'] = sma50_res.get("data")
            
            # SMA 200-day
            sma200_res = await self.massive._get_sma(symbol, timespan='day', window=200, limit=10)
            if sma200_res.get("success"):
                ma_data['sma_200'] = sma200_res.get("data")
            
            # EMA 12-day
            ema12_res = await self.massive._get_ema(symbol, timespan='day', window=12, limit=10)
            if ema12_res.get("success"):
                ma_data['ema_12'] = ema12_res.get("data")
            
            # EMA 26-day
            ema26_res = await self.massive._get_ema(symbol, timespan='day', window=26, limit=10)
            if ema26_res.get("success"):
                ma_data['ema_26'] = ema26_res.get("data")
                
        except Exception as e:
            logger.warning(f"Failed to calculate moving averages: {e}")
        
        return ma_data
    
    async def _calculate_momentum_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calculate momentum indicators (RSI, MACD)."""
        momentum_data = {}
        
        try:
            # RSI 14-day
            rsi_res = await self.massive._get_rsi(symbol, timespan='day', window=14, limit=10)
            if rsi_res.get("success"):
                momentum_data['rsi'] = rsi_res.get("data")
            
            # MACD (12,26,9)
            macd_res = await self.massive._get_macd(symbol, timespan='day', limit=10)
            if macd_res.get("success"):
                momentum_data['macd'] = macd_res.get("data")
                
        except Exception as e:
            logger.warning(f"Failed to calculate momentum indicators: {e}")
        
        return momentum_data
    
    def _generate_analysis_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary from comprehensive analysis data.
        This is a simplified version - can be enhanced with more sophisticated scoring.
        """
        summary = {
            "overall_score": None,
            "recommendation": "HOLD",
            "key_highlights": [],
            "risk_factors": []
        }
        
        try:
            # Extract key metrics for scoring
            highlights = []
            risks = []
            
            # Technical signals
            if 'technical' in data:
                tech_data = data['technical']
                if 'aggregate_signals' in tech_data:
                    agg_signals = tech_data.get('aggregate_signals', {})
                    if agg_signals:
                        highlights.append(f"Technical Signal: {agg_signals.get('signal', 'N/A')}")
                
                # RSI check
                if 'momentum' in tech_data and 'rsi' in tech_data['momentum']:
                    rsi_data = tech_data['momentum']['rsi']
                    if isinstance(rsi_data, dict) and 'values' in rsi_data:
                        rsi_values = rsi_data['values']
                        if rsi_values and len(rsi_values) > 0:
                            latest_rsi = rsi_values[0].get('value') if isinstance(rsi_values[0], dict) else None
                            if latest_rsi:
                                if latest_rsi < 30:
                                    highlights.append(f"RSI Oversold: {latest_rsi:.2f} (Potential Buy)")
                                elif latest_rsi > 70:
                                    risks.append(f"RSI Overbought: {latest_rsi:.2f} (Potential Correction)")
            
            # Sentiment signals
            if 'sentiment' in data:
                sent_data = data['sentiment']
                if 'analyst_ratings' in sent_data:
                    ratings = sent_data['analyst_ratings'].get('recommendation_trends', [])
                    if ratings and len(ratings) > 0:
                        latest = ratings[0]
                        buy_count = latest.get('buy', 0) + latest.get('strongBuy', 0)
                        sell_count = latest.get('sell', 0) + latest.get('strongSell', 0)
                        if buy_count > sell_count:
                            highlights.append(f"Analyst Consensus: Positive ({buy_count} Buy vs {sell_count} Sell)")
                        elif sell_count > buy_count:
                            risks.append(f"Analyst Consensus: Negative ({sell_count} Sell vs {buy_count} Buy)")
            
            summary['key_highlights'] = highlights if highlights else ["Insufficient data for highlights"]
            summary['risk_factors'] = risks if risks else ["No major risk factors identified"]
            
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
        
        return summary

if __name__ == "__main__":
    # Test Block
    import json
    
    async def test():
        analyzer = USStockAnalyzerService()
        symbol = "MSFT"
        
        # Dynamic date range: Last 30 days
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=30)
        
        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%d")
        
        print(f"\n{'='*60}")
        print(f"Testing US Stock Analyzer for {symbol}")
        print(f"Period: {start} to {end}")
        print(f"{'='*60}\n")
        
        # Test 1: Fundamental Analysis
        print(f"\n--- Test 1: Fundamental Analysis ---")
        fundamental = await analyzer.get_fundamental_analysis(symbol, start, end)
        print(f"Success: {fundamental.get('success')}")
        if fundamental.get('success'):
            data = fundamental.get('data', {})
            print(f"Available Keys: {list(data.keys())}")
            if 'company_profile' in data:
                profile = data['company_profile'].get('profile', {})
                print(f"Company Name: {profile.get('name')}")
                print(f"Market Cap: {profile.get('marketCapitalization')}")
            if 'segments_kpis' in data:
                kpis = data['segments_kpis']
                print(f"KPI Data Found: {len(kpis.get('metrics', []))} metrics")
                # Print sample KPI values
                if 'data' in kpis and kpis['data']:
                    latest = kpis['data'][0]
                    vals = latest.get('metricsValues', {})
                    # Map IDs to names
                    id_map = {m['metricId']: m['metricName'] for m in kpis.get('metrics', [])}
                    print("Sample KPI Values (Latest):")
                    count = 0
                    for kid, kval in vals.items():
                        if count >= 3: break
                        kname = id_map.get(kid, kid)
                        val_num = kval.get('value') if isinstance(kval, dict) else kval
                        print(f"  - {kname}: {val_num}")
                        count += 1
        
        # Test 1.5: Direct KPI Test
        print(f"\n--- Test 1.5: Direct KPI Analysis ---")
        kpi_res = await analyzer.get_company_kpis(symbol)
        print(f"Success: {kpi_res.get('success')}")
        if kpi_res.get('success'):
            data = kpi_res.get('data', {})
            print(f"Keys: {list(data.keys())}")
            
            # Detailed KPI print
            if 'data' in data and data['data']:
                latest = data['data'][0]
                vals = latest.get('metricsValues', {})
                id_map = {m['metricId']: m['metricName'] for m in data.get('metrics', [])}
                print(f"Latest Period: {latest.get('periodId')}")
                print("Top KPI Values:")
                count = 0
                for kid, kval in vals.items():
                    if count >= 3: break
                    kname = id_map.get(kid, kid)
                    val_num = kval.get('value') if isinstance(kval, dict) else kval
                    print(f"  - {kname}: {val_num}")
                    count += 1
        
        # Test 2: Technical Analysis
        print(f"\n--- Test 2: Technical Analysis ---")
        technical = await analyzer.get_technical_analysis(symbol, start, end)
        print(f"Success: {technical.get('success')}")
        if technical.get('success'):
            data = technical.get('data', {})
            print(f"Available Keys: {list(data.keys())}")
            if 'quote' in data:
                quote = data['quote']
                print(f"Current Price: ${quote.get('c')}")
        
        # Test 3: Sentiment Analysis
        print(f"\n--- Test 3: Sentiment Analysis ---")
        sentiment = await analyzer.get_sentiment_analysis(symbol, start, end)
        print(f"Success: {sentiment.get('success')}")
        if sentiment.get('success'):
            data = sentiment.get('data', {})
            print(f"Available Keys: {list(data.keys())}")
        
        # Test 4: Comprehensive Analysis
        print(f"\n--- Test 4: Comprehensive Analysis ---")
        comprehensive = await analyzer.get_comprehensive_analysis(symbol, start, end)
        print(f"Success: {comprehensive.get('success')}")
        if comprehensive.get('success'):
            data = comprehensive.get('data', {})
            print(f"Analysis Dimensions: {list(data.keys())}")
            if 'summary' in data:
                summary = data['summary']
                print(f"\nSummary:")
                print(f"  Recommendation: {summary.get('recommendation')}")
                print(f"  Key Highlights: {summary.get('key_highlights')}")
                print(f"  Risk Factors: {summary.get('risk_factors')}")
        
        print(f"\n{'='*60}")
        print(f"Testing Complete")
        print(f"{'='*60}\n")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test())

