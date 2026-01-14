#!/usr/bin/env python3
"""
X (Twitter) Search Service
Retrieves and ranks X posts for market sentiment and news analysis.
Optimized for short-term trading signals with popularity scoring and time decay.
"""
import os
import logging
import requests
import math
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XSentimentService:
    """
    X Search and Sentiment Analysis Service.
    
    Features:
    - Flexible query-based search (LLM provides the search query)
    - Low-noise filtering (excludes retweets and replies)
    - Custom popularity scoring with engagement metrics
    - Time-decay weighting for recency (6-hour half-life)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize X Sentiment Service.
        
        Args:
            api_key: X API Bearer token. If not provided, reads from X_API_KEY env variable.
        """
        self.name = "X Search and Sentiment Service"
        self.api_key = api_key or os.getenv('X_API_KEY')
        
        if not self.api_key:
            logger.warning("X_API_KEY not found in environment variables")
        else:
            logger.info("XSentimentService initialized with API key")
        
        self.base_url = "https://api.x.com/2/tweets/search/recent"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _calculate_popularity_score(self, metrics: Dict[str, int], created_at: str) -> float:
        """
        Calculate popularity score with time decay.
        
        Scoring formula:
        - Impressions: 1.5x weight (reach)
        - Likes: 1.0x weight (engagement)
        - Retweets: 2.0x weight (virality)
        - Replies: 1.0x weight (discussion)
        - Time decay: exponential decay over 6 hours
        
        Args:
            metrics: Public metrics dict with impression_count, like_count, etc.
            created_at: ISO timestamp of tweet creation
            
        Returns:
            float: Calculated popularity score
        """
        # Extract metrics (handle missing values)
        impressions = metrics.get('impression_count', 0)
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        replies = metrics.get('reply_count', 0)
        
        # Base score calculation (log scale to handle large variations)
        base_score = (
            1.5 * math.log(impressions + 1) +
            1.0 * math.log(likes + 1) +
            2.0 * math.log(retweets + 1) +
            1.0 * math.log(replies + 1)
        )
        
        # Time decay calculation (6-hour half-life for short-term trading)
        try:
            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            now = datetime.now(created_time.tzinfo)
            hours_old = (now - created_time).total_seconds() / 3600
            
            # Exponential decay: e^(-t/6h)
            time_factor = math.exp(-hours_old / 6.0)
        except Exception as e:
            logger.warning(f"Time decay calculation failed: {e}")
            time_factor = 1.0
        
        final_score = base_score * time_factor
        
        return round(final_score, 2)

    def _fetch_tweets(self, query: str, max_results: int = 50) -> Optional[Dict[str, Any]]:
        """
        Fetch tweets from X API.
        
        Args:
            query: X API search query string
            max_results: Maximum number of results to fetch (max 100)
            
        Returns:
            Dict: Raw API response or None if error
        """
        if not self.api_key:
            logger.error("Cannot fetch tweets: API key is missing")
            return None
        
        params = {
            "query": query,
            "max_results": min(max_results, 100),  # API limit is 100
            "tweet.fields": "created_at,public_metrics,author_id,lang",
            "expansions": "author_id",
            "user.fields": "username,name,profile_image_url,verified"
        }
        
        try:
            logger.info(f"Fetching tweets with query: {query}")
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                tweet_count = len(data.get('data', []))
                logger.info(f"Successfully fetched {tweet_count} tweets")
                return data
            else:
                logger.error(f"X API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch tweets: {str(e)}")
            return None

    def _process_and_rank_tweets(self, raw_data: Dict[str, Any], query: str, min_engagement: int = 0) -> List[Dict[str, Any]]:
        """
        Process raw tweet data and rank by popularity score.
        
        Args:
            raw_data: Raw response from X API
            query: Search query used (for logging)
            min_engagement: Minimum total engagement (likes + retweets + replies) to include
            
        Returns:
            List[Dict]: Sorted list of processed tweets with scores
        """
        if not raw_data or 'data' not in raw_data:
            return []
        
        tweets = raw_data.get('data', [])
        users = {user['id']: user for user in raw_data.get('includes', {}).get('users', [])}
        
        processed_tweets = []
        
        for tweet in tweets:
            # Get author info
            author_id = tweet.get('author_id')
            author = users.get(author_id, {})
            
            # Calculate popularity score
            metrics = tweet.get('public_metrics', {})
            created_at = tweet.get('created_at', '')
            
            # Calculate total engagement
            total_engagement = (
                metrics.get('like_count', 0) + 
                metrics.get('retweet_count', 0) + 
                metrics.get('reply_count', 0)
            )
            
            # Skip tweets below minimum engagement threshold
            if total_engagement < min_engagement:
                continue
            
            score = self._calculate_popularity_score(metrics, created_at)
            
            # Build processed tweet object
            processed = {
                "id": tweet.get('id'),
                "text": tweet.get('text'),
                "created_at": created_at,
                "author": {
                    "username": author.get('username', 'unknown'),
                    "name": author.get('name', 'Unknown User'),
                    "verified": author.get('verified', False),
                    "profile_image": author.get('profile_image_url', '')
                },
                "metrics": {
                    "impressions": metrics.get('impression_count', 0),
                    "likes": metrics.get('like_count', 0),
                    "retweets": metrics.get('retweet_count', 0),
                    "replies": metrics.get('reply_count', 0),
                    "total_engagement": total_engagement
                },
                "score": score,
                "url": f"https://x.com/{author.get('username', 'i')}/status/{tweet.get('id')}"
            }
            
            processed_tweets.append(processed)
        
        # Sort by score (highest first)
        processed_tweets.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Processed and ranked {len(processed_tweets)} tweets (filtered from {len(tweets)})")
        return processed_tweets

    def search_x_posts(self, 
                      query: str,
                      max_results: int = 20,
                      exclude_retweets: bool = True,
                      exclude_replies: bool = True,
                      require_links: bool = True,
                      language: str = "en",
                      min_engagement: int = 5) -> Dict[str, Any]:
        """
        Search X (Twitter) for posts and return ranked results by popularity and recency.
        Designed for market sentiment analysis and news monitoring.
        
        Args:
            query (str): The search query. For stock analysis, use specific queries like:
                        - "Meta stock" OR "META earnings" (stock-specific)
                        - "Tesla deliveries" OR "TSLA production"
                        - "NVDA OR NVIDIA" AND "revenue OR earnings OR guidance"
                        Avoid generic platform names without context.
            max_results (int): Maximum number of results to return. Defaults to 20.
            exclude_retweets (bool): If True, excludes retweets from results. Defaults to True.
            exclude_replies (bool): If True, excludes reply tweets. Defaults to True.
            require_links (bool): If True, only returns tweets with links (higher quality news). Defaults to True.
            language (str): Language code filter. Defaults to "en".
            min_engagement (int): Minimum total engagement (likes + retweets + replies) to include. 
                                 Higher values filter noise. Defaults to 5.
            
        Returns:
            Dict[str, Any]: Search results containing:
                - success: bool indicating if request succeeded
                - query_used: the actual X API query with filters
                - results_count: number of results returned
                - results: list of ranked tweets sorted by popularity score, each containing:
                    - id: tweet ID
                    - text: tweet content
                    - created_at: timestamp
                    - author: username, name, verified status
                    - metrics: impressions, likes, retweets, replies, total_engagement
                    - score: calculated popularity score with time decay
                    - url: direct link to tweet
                - metadata: query timestamp, fetch statistics, and filter info
        """
        # Build complete query with filters
        query_parts = [f"({query})"]
        
        if exclude_retweets:
            query_parts.append("-is:retweet")
        if exclude_replies:
            query_parts.append("-is:reply")
        if require_links:
            query_parts.append("has:links")
        if language:
            query_parts.append(f"lang:{language}")
        
        full_query = " ".join(query_parts)
        
        # Fetch tweets (get more to account for filtering)
        raw_data = self._fetch_tweets(full_query, max_results=100)
        
        if raw_data is None:
            return {
                "success": False,
                "query_used": full_query,
                "error": "Failed to fetch data from X API",
                "metadata": {
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        
        # Process and rank tweets with engagement filter
        ranked_tweets = self._process_and_rank_tweets(raw_data, query, min_engagement)
        
        # Limit to requested number of results
        final_results = ranked_tweets[:max_results]
        
        return {
            "success": True,
            "query_used": full_query,
            "results_count": len(final_results),
            "results": final_results,
            "metadata": {
                "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_fetched": len(raw_data.get('data', [])),
                "after_filtering": len(ranked_tweets),
                "returned": len(final_results),
                "min_engagement_filter": min_engagement
            }
        }


if __name__ == "__main__":
    # Test the service
    service = XSentimentService()
    
    if service.api_key:
        print(f"=== Testing {service.name} ===\n")
        
        # Test with a sample query
        query = "Meta layoffs OR Facebook layoffs"
        print(f"Searching X for: {query}\n")
        result = service.search_x_posts(query, max_results=10)
        
        if result.get('success'):
            print(f"✅ Success!")
            print(f"Query used: {result['query_used']}")
            print(f"Results: {result['results_count']}\n")
            
            # Print top 3 tweets
            for i, tweet in enumerate(result['results'][:3], 1):
                print(f"\n--- Tweet #{i} (Score: {tweet['score']}) ---")
                print(f"Author: @{tweet['author']['username']} ({tweet['author']['name']})")
                verified = "✓" if tweet['author']['verified'] else ""
                if verified:
                    print(f"Verified: {verified}")
                print(f"Posted: {tweet['created_at']}")
                print(f"Text: {tweet['text'][:200]}...")
                print(f"Metrics: 👁 {tweet['metrics']['impressions']:,} | "
                      f"❤️ {tweet['metrics']['likes']:,} | "
                      f"🔁 {tweet['metrics']['retweets']:,} | "
                      f"💬 {tweet['metrics']['replies']:,}")
                print(f"URL: {tweet['url']}")
        else:
            print(f"❌ Failed: {result.get('error')}")
    else:
        print("❌ Service initialized but no API key found")
        print("Please set X_API_KEY in your environment variables")
