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

    def get_limit_up_stocks(self, date):
        """
        获取指定日期的涨停板股票
        使用 limit_list_d 接口
        :param date: 日期格式 'YYYYMMDD'
        :return: DataFrame
        """
        try:
            print(f"Fetching limit up data for {date} via limit_list_d...")
            # 尝试使用 limit_list_d 接口
            # limit_type='U' 表示涨停
            df = self.pro.limit_list_d(trade_date=date, limit_type='U')
            
            if df.empty:
                print(f"No limit up stocks found for {date} via limit_list_d.")
                return pd.DataFrame()

            # 过滤板块
            if 'ts_code' in df.columns:
                # 过滤北交所 (8xx, 43x, 92x, .BJ)
                df = df[~df['ts_code'].str.contains(r'\.BJ$', regex=True)]
                df = df[~df['ts_code'].str.match(r'^(8|43|92)')]
                
                # 过滤科创板 (688)
                df = df[~df['ts_code'].str.startswith('688')]
                
                # 过滤创业板 (300, 301)
                df = df[~df['ts_code'].str.startswith(('300', '301'))]

            # 补全基础信息 (name, industry)
            # limit_list_d 可能包含 name, 但通常不包含 industry
            # 获取所有涉及的股票代码
            ts_codes = df['ts_code'].tolist()
            
            if ts_codes:
                # 批量获取基础信息
                # 注意: stock_basic 可能会返回所有股票，我们可以只通过 merge 匹配
                # 只需要 industry，name 如果 limit_list_d 有就用它的，没有就用 stock_basic 的
                fields = 'ts_code,industry'
                if 'name' not in df.columns:
                    fields += ',name'
                
                base_info = self.pro.stock_basic(exchange='', list_status='L', fields=fields)
                
                # Merge
                df = pd.merge(df, base_info, on='ts_code', how='left')

            # 过滤ST (接口说明不提供ST统计，为了保险再次检查)
            if 'name' in df.columns:
                df = df[~df['name'].str.contains('ST', case=False, na=False)]
            
            return df

        except Exception as e:
            print(f"Error getting limit up stocks for {date}: {e}")
            # 如果是积分不足或其他API错误，打印详细信息
            return pd.DataFrame()

    def get_two_ban_stocks(self, date):
        """
        获取指定日期的2连板股票
        :param date: 日期格式 'YYYYMMDD'
        :return: DataFrame
        """
        limit_up_df = self.get_limit_up_stocks(date)
        
        if limit_up_df.empty:
            return pd.DataFrame()

        # 筛选2连板股票
        # limit_list_d 接口返回字段通常包含 limit_times (连板数)
        # 注意: 接口文档显示字段名为 limit_times，而不是 limit
        
        target_col = 'limit_times'
        if target_col not in limit_up_df.columns:
             # 尝试查找其他可能的列名
             if 'limit' in limit_up_df.columns and limit_up_df['limit'].iloc[0] in [1, 2, 3, '1', '2', '3']:
                  target_col = 'limit'
             else:
                  print(f"Warning: '{target_col}' column not found in limit_list_d result. Columns: {limit_up_df.columns}")
                  return pd.DataFrame()
        
        print(f"Using column '{target_col}' for consecutive limit up count.")
        
        # 筛选连板数为 2 的股票
        try:
            # 尝试转为数值进行比较
            two_ban_df = limit_up_df[limit_up_df[target_col].astype(int) == 2].copy()
        except:
             # 如果转换失败，尝试字符串比较
             two_ban_df = limit_up_df[limit_up_df[target_col].astype(str) == '2'].copy()
        
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
                ts_code = row.get('ts_code', '')
                name = row.get('name', '')
                industry = row.get('industry', 'N/A')
                reason = row.get('limit_reason', 'N/A')
                
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
