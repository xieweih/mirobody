import tushare as ts
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class TwoBanService:
    def __init__(self):
        self.token = os.getenv('TUSHARE_TOKEN')
        if not self.token:
            raise ValueError("TUSHARE_TOKEN not found in environment variables")
        ts.set_token(self.token)
        self.pro = ts.pro_api()

    def is_limit_up(self, row):
        """
        判断单只股票是否涨停
        根据板块不同，阈值不同
        """
        code = row['ts_code']
        pct_chg = row['pct_chg']
        
        # 北交所 (8xx, 43x) - 30%
        if code.startswith('8') or code.startswith('43'):
             return pct_chg > 29.5
             
        # 科创板 (688), 创业板 (300, 301) - 20%
        if code.startswith('688') or code.startswith('300') or code.startswith('301'):
            return pct_chg > 19.5
            
        # 主板 - 10% (ST 5% 暂不单独处理，通常 >4.9 但这里简化为通用主板逻辑 >9.5 可能会漏掉ST涨停，但过滤了非涨停)
        # 为了严谨，主板非ST是10%，ST是5%。
        # 如果要精确，最好有 name 判断 ST。
        # 简单起见，目前只区分 10% 和 20% 板。
        return pct_chg > 9.5

    def get_limit_up_stocks(self, date):
        """
        获取指定日期的涨停板股票
        :param date: 日期格式 'YYYYMMDD'
        :return: DataFrame
        """
        try:
            # 尝试使用 limit_list 接口
            df = self.pro.limit_list(trade_date=date)
            if not df.empty:
                return df
            
            # Fallback: 如果 limit_list 返回空（可能是权限或数据问题），尝试自行计算
            print(f"limit_list empty for {date}, trying calculation via daily data...")
            
            # 获取当日行情
            print(f"Fetching daily data for {date}...")
            df_daily = self.pro.daily(trade_date=date)
            if df_daily.empty:
                print(f"Daily data empty for {date}")
                return pd.DataFrame()
            
            # 智能筛选涨停
            # df_daily['is_limit'] = df_daily.apply(self.is_limit_up, axis=1) # 这种写法慢
            
            # 向量化筛选更快
            # 1. 拆分板块
            is_bj = df_daily['ts_code'].str.match(r'^(8|43)')
            is_kc_cy = df_daily['ts_code'].str.match(r'^(688|300|301)')
            is_main = ~(is_bj | is_kc_cy)
            
            # 过滤ST股 (根据名称判断，包含 ST 或 *ST)
            # 先获取基础信息以判断名称
            base_info = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            df_daily = pd.merge(df_daily, base_info[['ts_code', 'name']], on='ts_code', how='left')
            is_st = df_daily['name'].str.contains('ST', case=False, na=False)

            # 用户需求：非ST，非科创，非创业，非北交所 -> 只保留主板且非ST
            # 主板非ST涨停逻辑：> 9.5%
            cond_main_no_st = is_main & (~is_st) & (df_daily['pct_chg'] > 9.5)
            
            limit_up = df_daily[cond_main_no_st].copy()
            
            print(f"Found {len(limit_up)} limit up stocks (Main Board, No ST) on {date}")
            
            # 获取前一日行情，计算连板
            from datetime import timedelta
            # 获取最近的一个交易日
            # 注意：Tushare trade_cal 返回通常是降序（最近的在前），或者取决于 start/end
            # 我们显式按 cal_date 排序以防万一
            # 这里我们取过去1个月的日历就够了，不用取一整年
            start_date_cal = (datetime.strptime(date, '%Y%m%d') - timedelta(days=30)).strftime('%Y%m%d')
            cal = self.pro.trade_cal(exchange='', start_date=start_date_cal, end_date=date, is_open='1')
            
            # 确保按日期升序排列
            cal = cal.sort_values('cal_date').reset_index(drop=True)
            
            if not cal.empty and len(cal) >= 2:
                # cal[-1] 应该是 date (如果 date 是交易日)
                # cal[-2] 是上一交易日
                prev_trade_date = cal.iloc[-2]['cal_date']
                print(f"Checking previous trade date: {prev_trade_date}")
                
                df_prev = self.pro.daily(trade_date=prev_trade_date)
                
                # 合并基础信息以过滤 ST
                df_prev = pd.merge(df_prev, base_info[['ts_code', 'name']], on='ts_code', how='left')
                
                # 筛选昨日也涨停的股票 (Smart Filter: 仅主板非ST)
                p_is_bj = df_prev['ts_code'].str.match(r'^(8|43)')
                p_is_kc_cy = df_prev['ts_code'].str.match(r'^(688|300|301)')
                p_is_main = ~(p_is_bj | p_is_kc_cy)
                p_is_st = df_prev['name'].str.contains('ST', case=False, na=False)
                
                p_cond_main_no_st = p_is_main & (~p_is_st) & (df_prev['pct_chg'] > 9.5)
                
                prev_limit_up = df_prev[p_cond_main_no_st]['ts_code'].tolist()
                
                # 今涨停 且 昨涨停
                two_ban_candidates = limit_up[limit_up['ts_code'].isin(prev_limit_up)].copy()
                print(f"Candidates (>=2 ban): {len(two_ban_candidates)}")
                
                if len(cal) >= 3:
                     prev_2_date = cal.iloc[-3]['cal_date']
                     print(f"Checking pre-prev trade date: {prev_2_date}")
                     df_prev_2 = self.pro.daily(trade_date=prev_2_date)
                     # 合并基础信息以过滤 ST
                     df_prev_2 = pd.merge(df_prev_2, base_info[['ts_code', 'name']], on='ts_code', how='left')
                     
                     # 同理筛选前前日涨停 (仅主板非ST)
                     p2_is_bj = df_prev_2['ts_code'].str.match(r'^(8|43)')
                     p2_is_kc_cy = df_prev_2['ts_code'].str.match(r'^(688|300|301)')
                     p2_is_main = ~(p2_is_bj | p2_is_kc_cy)
                     p2_is_st = df_prev_2['name'].str.contains('ST', case=False, na=False)
                     
                     p2_cond_main_no_st = p2_is_main & (~p2_is_st) & (df_prev_2['pct_chg'] > 9.5)
                     
                     prev_2_limit_up = df_prev_2[p2_cond_main_no_st]['ts_code'].tolist()
                     
                     # 排除掉前前日也涨停的 -> 剩下就是正好2连板
                     two_ban_exact = two_ban_candidates[~two_ban_candidates['ts_code'].isin(prev_2_limit_up)].copy()
                     print(f"Exact 2-ban: {len(two_ban_exact)}")
                     
                     if two_ban_exact.empty:
                         return pd.DataFrame()

                     # 构造 limit 字段
                     two_ban_exact['limit'] = 2
                     # 补充 name/industry 字段 (已经merge过name，这里补充industry)
                     # 注意之前merge只取了name，这里如果之前merge覆盖了columns可能会有冲突，
                     # 但limit_up是copy出来的，我们重新merge完整信息比较稳妥
                     base_info_full = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
                     # drop name from two_ban_exact to avoid suffix (it has name from previous merge)
                     if 'name' in two_ban_exact.columns:
                         two_ban_exact = two_ban_exact.drop(columns=['name'])
                         
                     two_ban_exact = pd.merge(two_ban_exact, base_info_full, on='ts_code', how='left')
                     
                     return two_ban_exact
            
            return pd.DataFrame()

        except Exception as e:
            print(f"Error getting limit up stocks for {date}: {e}")
            return pd.DataFrame()

    def get_two_ban_stocks(self, date):
        """
        获取指定日期的2连板股票
        :param date: 日期格式 'YYYYMMDD'
        :return: DataFrame
        """
        limit_up_df = self.get_limit_up_stocks(date)
        
        if limit_up_df.empty:
            print(f"No limit up stocks found for {date}")
            return pd.DataFrame()

        # 筛选2连板股票 (limit_amount >= 2, 或者通过 consecutive_limit_up_days 字段如果有的话)
        # limit_list 接口返回字段包含 limit (连板数)
        
        # 筛选 limit 字段为 2 的股票
        two_ban_df = limit_up_df[limit_up_df['limit'] == 2].copy()
        
        return two_ban_df

    def save_stocks_to_file(self, df, date):
        """
        保存股票数据到文件
        :param df: DataFrame
        :param date: 日期 'YYYYMMDD'
        """
        if df.empty:
            print("No stocks to save.")
            return

        # 创建目录结构 data/twoban/YYYYMMDD
        base_dir = os.path.join(os.getcwd(), 'data', 'twoban', date)
        os.makedirs(base_dir, exist_ok=True)

        # 保存为 CSV
        file_path_csv = os.path.join(base_dir, f"{date}_2ban_stocks.csv")
        df.to_csv(file_path_csv, index=False, encoding='utf-8-sig')
        print(f"Saved {len(df)} stocks to {file_path_csv}")

        # 保存为 TXT (人类可读)
        file_path_txt = os.path.join(base_dir, f"{date}_2ban_stocks.txt")
        with open(file_path_txt, 'w', encoding='utf-8') as f:
            f.write(f"Date: {date}\n")
            f.write(f"Total: {len(df)}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Code':<10} {'Name':<10} {'Industry':<15} {'Reason'}\n")
            f.write("-" * 50 + "\n")
            
            for _, row in df.iterrows():
                # 注意：limit_list 接口可能不包含行业和原因，这里尽量提取
                ts_code = row.get('ts_code', '')
                name = row.get('name', '')
                industry = row.get('industry', 'N/A') # limit_list 通常有 industry
                reason = row.get('limit_reason', 'N/A') # limit_list 通常包含板的原因
                
                f.write(f"{ts_code:<10} {name:<10} {industry:<15} {reason}\n")
        
        print(f"Saved txt report to {file_path_txt}")

    def run(self, date=None):
        if not date:
            date = datetime.now().strftime('%Y%m%d')
        
        print(f"Fetching 2-ban stocks for {date}...")
        df = self.get_two_ban_stocks(date)
        
        if not df.empty:
            print(f"Found {len(df)} 2-ban stocks.")
            self.save_stocks_to_file(df, date)
        else:
            print("No 2-ban stocks found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch 2-ban stocks")
    parser.add_argument("--date", type=str, help="Date in YYYYMMDD format", default=None)
    args = parser.parse_args()
    
    service = TwoBanService()
    service.run(args.date)
