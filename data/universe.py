# cluster_qpo/data/universe.py

SP100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "UNH",
    "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG", "JNJ", "HD", "MRK",
    "COST", "ABBV", "CVX", "CRM", "BAC", "NFLX", "AMD", "PEP", "KO", "TMO",
    "ACN", "MCD", "CSCO", "WMT", "ABT", "LIN", "DHR", "TXN", "ADBE", "NEE",
    "PM", "NKE", "ORCL", "QCOM", "UPS", "MS", "RTX", "AMGN", "LOW", "INTU",
    "INTC", "GS", "HON", "IBM", "CAT", "ELV", "AMAT", "SPGI", "DE", "MDLZ",
    "GE", "AXP", "ISRG", "BKNG", "ADI", "TJX", "PLD", "VRTX", "SYK",
    "C", "REGN", "CB", "CI", "SCHW", "ZTS", "MO", "BDX", "EOG", "SO",
    "CME", "DUK", "NOC", "AON", "BSX", "ITW", "PNC", "WM", "F", "GM",
    "USB", "GD", "EMR", "APD", "PSA", "NSC", "SLB", "HCA", "MCO", "CL"
]

SP500 = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "CSCO", "ACN", "AMD",
    "QCOM", "TXN", "IBM", "INTU", "NOW", "AMAT", "MU", "LRCX", "ADI", "KLAC",
    "SNPS", "CDNS", "PANW", "FTNT", "ANSS", "MCHP", "NXPI", "ON", "SWKS", "TER",
    "TYL", "ZBRA", "KEYS", "GEN", "AKAM", "JNPR", "NTAP", "WDC", "STX", "HPQ",
    "HPE", "DELL", "CTSH", "GLW", "APH", "TEL", "MSI", "ROP", "TDY", "FFIV",
    "PTC", "ENPH", "FSLR", "GRMN", "EPAM", "PAYC", "PAYX", "ADP", "FIS", "GPN",
    "JKHY", "MA", "V", "PYPL", "WU", "VRSN", "TRMB", "CDW", "SMCI", "IT",

    # Communication Services
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR",
    "OMC", "IPG", "EA", "TTWO", "WBD", "NWSA", "FOXA", "LYV", "MTCH",

    # Healthcare
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN",
    "ISRG", "BSX", "SYK", "VRTX", "ELV", "MDT", "GILD", "CI", "REGN", "ZTS",
    "BDX", "HCA", "CVS", "MCK", "EW", "IDXX", "A", "IQV", "CNC", "HUM",
    "BIIB", "RMD", "MTD", "DXCM", "WAT", "ALGN", "ZBH", "BAX", "STE", "COO",
    "HOLX", "PODD", "VTRS", "CAH", "MOH", "INCY", "RVTY", "TECH", "UHS", "DVA",

    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
    "SCHW", "C", "AXP", "CB", "PGR", "MMC", "ICE", "CME", "AON", "USB",
    "PNC", "TFC", "AIG", "MET", "TRV", "AFL", "ALL", "PRU", "COF", "MCO",
    "BK", "AMP", "DFS", "FITB", "STT", "WTW", "GPN", "SYF", "NTRS", "RF",
    "HBAN", "CFG", "KEY", "CINF", "L", "MKTX", "RJF", "BRO", "ACGL", "FDS",

    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ORLY",
    "CMG", "MAR", "GM", "F", "HLT", "ROST", "YUM", "AZO", "LVS", "EBAY",
    "DHI", "LEN", "NVR", "PHM", "ULTA", "BBY", "APTV", "GRMN", "DPZ", "EXPE",
    "RL", "TPR", "POOL", "WYNN", "MGM", "CCL", "RCL", "NCLH", "HAS", "WHR",

    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "TGT",
    "KMB", "GIS", "SYY", "STZ", "KHC", "ADM", "KDP", "HSY", "MKC", "CHD",
    "CLX", "TAP", "K", "SJM", "CAG", "CPB", "CASY", "CHRW", "CTVA", "CTAS",

    # Industrials
    "CAT", "RTX", "HON", "UNP", "UPS", "BA", "GE", "DE", "LMT", "ADP",
    "ETN", "NOC", "GD", "ITW", "EMR", "CSX", "NSC", "WM", "FDX", "PH",
    "TT", "CARR", "JCI", "PCAR", "CMI", "RSG", "ODFL", "OTIS", "XYL", "AME",
    "ROK", "IR", "FAST", "VRSK", "PAYX", "DOV", "BR", "HWM", "TXT", "SNA",
    "IEX", "J", "PWR", "GWW", "LDOS", "EFX", "EXPD", "EWO", "EWK", "JBHT",

    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "OKE", "HES", "BKR", "HAL", "DVN", "FANG", "CTRA", "TRGP", "EQT",

    # Utilities
    "NEE", "SO", "DUK", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG",
    "WEC", "ES", "AWK", "DTE", "PPL", "EIX", "FE", "AEE", "CMS", "CNP",
    "ATO", "NI", "LNT", "EVRG", "PNW",

    # Real Estate
    "PLD", "AMT", "EQIX", "PSA", "CCI", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "VICI", "SBAC", "EXR", "VTR", "IRM", "INVH", "MAA", "ESS", "UDR",
    "ARE", "KIM", "HST", "REG", "BXP",

    # Materials
    "LIN", "APD", "SHW", "FCX", "ECL", "NEM", "NUE", "DOW", "DD", "PPG",
    "VMC", "MLM", "CTVA", "IFF", "ALB", "CF", "FMC", "MOS", "AVY", "PKG",
    "BALL", "IP", "SEE", "AMCR", "LYB"
]

# De-duplicate while preserving order (some tickers appear in multiple
# sector lists above, e.g. MA/V, GPN, PAYX, CTVA)
SP500 = list(dict.fromkeys(SP500))

NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "HDFC.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS",
    "SBILIFE.NS", "TECHM.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
    "JSWSTEEL.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "CIPLA.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "BRITANNIA.NS",
    "NESTLEIND.NS", "EICHERMOT.NS", "GRASIM.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS",
    "BPCL.NS", "HINDALCO.NS", "INDUSINDBK.NS", "SBIN.NS", "TATACONSUM.NS",
    "UPL.NS", "M&M.NS", "ADANIENT.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS"
]
