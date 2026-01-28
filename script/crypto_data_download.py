"""
Download BTC aggTrades data from 2020 to 2025
This script downloads raw aggregated trade data for BTCUSDT from Binance Vision
"""
from src.data_loader.raw_data_downloader import RawDataDownloader

def main():
    """Download BTC aggTrades data from 2020-01-01 to 2025-12-31"""
    
    # Initialize downloader
    downloader = RawDataDownloader()
    
    # Download parameters
    symbol = "BTCUSDT"
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    
    # Download aggregated trades
    # print(f"\n🚀 开始下载 {symbol} AggTrades 数据")
    print(f"   时间范围: {start_date} 至 {end_date}")
    
    # downloader.download_agg_trades(
    #     symbol=symbol,
    #     start_date=start_date,
    #     end_date=end_date
    # )
    
    downloader.download_klines(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="5m",
    )
    
    # print(f"\n✅ {symbol} AggTrades 下载完成！")
    print(f"   数据已保存至: {downloader.base_dir}")


if __name__ == "__main__":
    main()
