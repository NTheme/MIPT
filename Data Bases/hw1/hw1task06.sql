SELECT dt, high_price, vol
  FROM coins
  WHERE dt >= '2018-01-01' AND dt <= '2018-12-31' AND avg_price > 0.001 AND symbol = 'DOGE'
