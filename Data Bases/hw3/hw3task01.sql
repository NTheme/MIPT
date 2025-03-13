SELECT RANK() OVER (ORDER BY db.vol DESC) AS rank, *
FROM (SELECT coins.dt AS dt,
             SUM (coins.vol) AS vol
      FROM coins
      GROUP BY dt) AS db
LIMIT 10;
