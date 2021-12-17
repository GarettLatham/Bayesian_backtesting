from sklearn.linear_model import LinearRegression
# pandas and numpy are used for data manipulation

import math
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
# matplotlib and seaborn are used for plotting graphs
import time
#import matplotlib.pyplot as plt

#import seaborn
# fix_yahoo_finance is used to fetch data
import mysql.connector
import yfinance as yf
import backtrader as bt
#from tpot import TPOTClassifier
#from sklearn.model_selection import TimeSeriesSplit

# Read data
def main():
    try:
        db = mysql.connector.connect(user='', password='',
                          host='localhost',
                          database='stocks')
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        #execute SQL query using execute() method.
        cursor.execute("lock tables stocks read")
        cursor.execute("select Ticker, exchange from stocks where (exchangeDisplay='NASDAQ' OR exchangeDisplay='NYSE MKT' OR exchangeDisplay='NYSE' OR exchangeDisplay='OTC Market' OR exchangeDisplay='NASDAQ GIDS') AND (Type='S' OR Type='E') AND NOT (Ticker LIKE '%^%') AND Ticker > 'PALL'")
        #Fetch a single row using fetchone() method.
        sql_columns = np.array(cursor.fetchall())
        cursor.execute("unlock tables")
        db.commit()
        print(sql_columns.tolist())
        # cursor.execute("lock tables tick read")
        #cursor.execute("select tick_datetime from tick")
        #Fetch a single row using fetchone() method.
        #sql_index = np.array(cursor.fetchall())
        #cursor.execute("unlock tables")
        #db.commit()
        #disconnect from server
        cursor.close()
        db.close()
    except Exception as e:
        print("Caught exception when connecting to stream\n" + str(e))
        #continue
        #print(sql_columns)
        
    for symbol, exchange in sql_columns.tolist():

        try:
            X = yf.download(symbol,'1980-11-18','2021-05-17')
        
            #print(sql_columns)
                #X.columns = "symbol" + "_" + X.columns.values

            X['date'] = X.index
            X['day_of_week'] = X['date'].dt.dayofweek
            #X['Next_Close'] = X['GLD_Close'].shift(-1)
            # Group by the `Symbol` column, then grab the `Close` column.
            #Close_groups = X.groupby('Adj Close')
            # Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
            #Close_groups = Close_groups.transform(lambda x : np.sign(x.diff()))
            #X['direction'] = np.where(X['Adj Close'].shift(-1) > X['Adj Close'], "1", np.where(X['Adj Close'].shift(-1) < X['Adj Close'],"0", np.where(X['Adj Close'].shift(-1) == X['Adj Close'],"2", "3")))
            X['direction'] = np.where(X['Adj Close'].shift(-1) > X['Adj Close'], "1", np.where(X['Adj Close'].shift(-1) < X['Adj Close'],"0", np.where(X['Adj Close'].shift(-1) == X['Adj Close'],"2", "3")))
            #X['label'] = X['Adj Close'].transform(lambda x : np.sign(x.diff()))
            #X['log_ret'] = np.log(X['Adj Close']) - np.log(X['Adj Close'].shift(-1))
            X['log_ret'] = X['Adj Close'].pct_change().shift(-1).abs()
            #X['log_ret'] = np.where(X['label'] == 1, X['Adj Close'].pct_change(), np.where(X['label'] == -1, (-1*X['Adj Close'].pct_change()), 2))
            X['p0'] = np.where(X['Adj Close'] > X['Adj Close'].shift(1), 1, np.where(X['Adj Close'] < X['Adj Close'].shift(1),"0", np.where(X['Adj Close'] == X['Adj Close'].shift(1),"2", "3")))
            #X['p1'] = np.where(X['Adj Close'].shift(1) > X['Adj Close'].shift(2), 1, 0)
            #X['p2'] = np.where(X['Adj Close'].shift(1) > X['Adj Close'].shift(3), 1, 0)
            #X['p3'] = np.where(X['Adj Close'].shift(1) > X['Adj Close'].shift(4), 1, 0)
            X['pattern'] = X['p0'].shift(1).astype(str) + X['p0'].astype(str)# + X['p2'].astype(str) #+ X['p2'].astype(str)
            #X['pattern'] = X['p1'].astype(str) + X['p0'].astype(str)# + X['p2'].astype(str) #+ X['p2'].astype(str)
            #X['label'] = np.where(X['pattern'] == '00', 1,-1)
            #print(X.head(20))
            # One way to do Bayes Thereom will see which is faster later
            #pattern_probs = X.groupby('pattern').size().div(len(X))
            #three_pattern_df = X.groupby(['pattern', 'label']).size().div(len(X)).div(pattern_probs, axis=0, level='pattern').reset_index()
            #print(pattern_probs)
            
            three_pattern_df = X.groupby(['pattern', 'direction']).size().agg({'count': lambda x: x, 'prop':lambda x: x / x.sum(level=0)}).unstack(level=0)
            #print(three_pattern_df)
            
            
            #three_pattern_df.columns = three_pattern_df.columns.get_level_values(0three_pattern_df.columns = three_pattern_df.columns.get_level_values(0))
            three_pattern_ret_df = X.groupby(['pattern', 'direction']).agg({"log_ret": ["mean", "std"]})
            #three_pattern_ret_df.columns = three_pattern_ret_df.columns.get_level_values(0)
            #three_pattern_df.join(three_pattern_ret_df)
            final_df = pd.merge(three_pattern_df, three_pattern_ret_df, how='inner', left_index = True, right_index = True).reset_index()
            final_df.rename(columns={final_df.columns[4]: "log_mean_ret" }, inplace = True)
            final_df.rename(columns={final_df.columns[5]: "log_std_ret" }, inplace = True)
            #final_df['expected_value'] = np.where(np.logical_and(final_df['pattern'] == final_df['pattern'].shift(-1), final_df['label'] == -1), ((-1*final_df['log_mean_ret']) * final_df['prop']) - ((final_df['log_mean_ret'].shift(-1)) * final_df['prop'].shift(-1)), np.where(np.logical_and(final_df['pattern'] == final_df['pattern'].shift(1), final_df['label'] == 1), (final_df['log_mean_ret'] * final_df['prop']) - ((-1*final_df['log_mean_ret'].shift(1)) * final_df['prop'].shift(1)), 0))
            final_df = final_df[~final_df.pattern.str.contains("2")]
            final_df = final_df[~final_df.pattern.str.contains("3")]
            final_df = final_df[~final_df.pattern.str.contains("nan")]
            final_df = final_df[~final_df.direction.str.contains("2")]
            final_df = final_df[~final_df.direction.str.contains("3")]
            
            final_df['expected_value'] = np.where(np.logical_and(final_df['pattern'] == final_df['pattern'].shift(-1), final_df['direction'] == "0"), ((final_df['log_mean_ret']) * final_df['prop']) - ((final_df['log_mean_ret'].shift(-1)) * final_df['prop'].shift(-1)), np.where(np.logical_and(final_df['pattern'] == final_df['pattern'].shift(1), final_df['direction'] == "1"), (final_df['log_mean_ret'] * final_df['prop']) - ((final_df['log_mean_ret'].shift(1)) * final_df['prop'].shift(1)), 0))
            final_df = final_df[(final_df['expected_value'] > 0)]
            final_df = final_df[(final_df['count'] > 200)]
            #final_df = final_df[~final_df.direction.str.contains("0")]
            final_df['symbol'] = str(symbol)
            final_df['exchange'] = str(exchange)
            #final_df.drop('index', axis = 1, inplace=True)
            final_df.set_index("symbol", inplace=True)
            

            final_df[['pnl', 'sharpe', 'vwr', 'sqn', 'dd_len', 'dd_perc', 'dd_money', 'dd_max_len', 'dd_max_perc', 'dd_max_money']] = final_df.apply(backtest, train_X=X, axis=1, result_type="expand")
            #X = X[X.index > '2020-02-20']
            #print(X)
            #final_df[['covid_pnl', 'covid_sharpe', 'covid_vwr', 'covid_sqn', 'covid_dd_len', 'covid_dd_perc', 'covid_dd_money', 'covid_dd_max_len', 'covid_dd_max_perc', 'covid_dd_max_money']] = final_df.apply(backtest, train_X=X, axis=1, result_type="expand")
            print(final_df)
            #print(three_pattern_ret_df)
            with open("/home/user0/Desktop/laptop_backup/stocks_coc1.csv", 'a') as f:
                final_df.to_csv(f, mode='a', header=f.tell()==0)
    
            #time.sleep(100)
            X.pop('p0')
            #X.pop('p1')
            #X.pop('p2')
            del X
            del three_pattern_df
            del final_df
            #X.pop('p3')
            #print(X)
                
        except Exception as e:
            print(str(e))
            continue
          
def backtest(row, train_X):

    train_X['label'] = np.where(train_X['pattern'] == row['pattern'], 1,-1)
    train_X = ohlc_adj(train_X)
    #Variable for our starting cash
    startcash = 10000

    #Create an instance of cerebro
    cerebro = bt.Cerebro(maxcpus=31)
    cerebro.broker.set_coc(True)
    #Set commissions
    comminfo = FixedCommisionScheme()
    cerebro.broker.addcommissioninfo(comminfo)

    '''
    Note that the defaults for the scheme assume the account currency is the counter currency. If your account currency is the same as the base currency, you will need to initialize it like this:
    comminfo = forexSpreadCommisionScheme(spread=10, acc_counter_currency=False)
    '''
  

    #Add our strategy
    #cerebro.addstrategy(Mltests)

    #Get oanda datframe
    #dataframe = get_data()
    #predictions = train_X['pred'].to_dict()
    #train_X.pop('pred')
    #Pass it to the backtrader datafeed and add it to the cerebro
    #data = bt.feeds.PandasData(dataname=train_X)
    data = PandasData_Label(dataname=train_X)

    #Add the data to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(startcash)
    cerebro.addsizer(maxRiskSizer)

    cerebro.add_signal(bt.SIGNAL_LONGSHORT, MySignal)
    #cerebro.add_signal(bt.SIGNAL_SHORT, MySignal)
    # Analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='myvwr', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='myta')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mydd')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='mysqn')
    # Run over everything
    thestrats = cerebro.run()
    thestrat = thestrats[0]
    #cerebro.run()
    #cerebro.plot()

    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - startcash

    #print('Final Portfolio Value: ${}'.format(portvalue))
    #print('P/L: ${}'.format(pnl))
    sharpeanalyzer = thestrat.analyzers.mysharpe.get_analysis()
    vwranalyzer = thestrat.analyzers.myvwr.get_analysis()
    sqnanalyzer = thestrat.analyzers.mysqn.get_analysis()
    #print('VWR: ', list(vwranalyzer.values())[0])
    #print('SQN: ', sqnanalyzer)
    tradeanalyzer = thestrat.analyzers.myta.get_analysis()
    drawdownanalyzer = thestrat.analyzers.mydd.get_analysis()
    #print('Drawdown: ', round(drawdownanalyzer.drawdown,2))
    return round(pnl,2), round(list(sharpeanalyzer.values())[0],2), round(list(vwranalyzer.values())[0],2), round(sqnanalyzer.sqn,2), drawdownanalyzer.len, round(drawdownanalyzer.drawdown,2), round(drawdownanalyzer.moneydown,2), drawdownanalyzer.max.len, round(drawdownanalyzer.max.drawdown,2), round(drawdownanalyzer.max.moneydown,2)

def ohlc_adj(dat):
    """
    :param dat: pandas DataFrame with stock data, including "Open", "High", "Low", "Close", and "Adj Close", with "Adj Close" containing adjusted closing prices
 
    :return: pandas DataFrame with adjusted stock data
 
    This function adjusts stock data for splits, dividends, etc., returning a data frame with
    "Open", "High", "Low" and "Close" columns. The input DataFrame is similar to that returned
    by pandas Yahoo! Finance API.
    """
    return pd.DataFrame({"Open": dat["Open"] * dat["Adj Close"] / dat["Close"],
                       "High": dat["High"] * dat["Adj Close"] / dat["Close"],
                       "Low": dat["Low"] * dat["Adj Close"] / dat["Close"],
                       "Close": dat["Adj Close"],
                       "Volume": dat["Volume"],
                       "direction": dat["direction"],
                       "p0": dat["p0"],
                       "pattern": dat["pattern"],
                       "label": dat["label"]})
                       
class LongOnly(bt.Sizer):
    params = (('stake', 5),)

    def _getsizing(self, comminfo, cash, data, isbuy):
      if isbuy:
          return self.p.stake

      # Sell situation
      position = self.broker.getposition(data)
      if not position.size:
          return 0  # do not sell if nothing is open

      return self.p.stake
      
class maxRiskSizer(bt.Sizer):
    '''
    Returns the number of shares rounded down that can be purchased for the
    max rish tolerance
    '''
    params = (('risk', 0.99),)

    def __init__(self):
        if self.p.risk > 1 or self.p.risk < 0:
            raise ValueError('The risk parameter is a percentage which must be'
                'entered as a float. e.g. 0.5')

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy == True:
            size = math.floor((cash * self.p.risk) / data[0])
        else:
            size = math.floor((cash * self.p.risk) / data[0]) * -1
        return size

class FixedCommisionScheme(bt.CommInfoBase):
    '''
    This is a simple fixed commission scheme
    '''
    params = (
        ('commission', 0),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        )

    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission


class PandasData_Label(bt.feeds.PandasData):

    # Add a 'pe' line to the inherited ones from the base class
    lines = ('label',)

    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('label', -1),)

class MySignal(bt.Indicator):
    lines = ('signal',)
    #params = (('period', 30),)

    def __init__(self):
        #print(self.data[0])
        #sleep(10)
        self.lines.signal = self.data.label
        
if __name__ == '__main__':
		main()
