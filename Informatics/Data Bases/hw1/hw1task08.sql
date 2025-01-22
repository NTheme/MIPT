SELECT UPPER(coins.full_nm) AS full_name, coins.dt, coins.high_price AS price
FROM (
    SELECT full_nm, MIN(dt) AS dt
    FROM (SELECT coins.full_nm, coins.dt
        FROM (
            SELECT full_nm, MAX(high_price) AS price
            FROM coins
            GROUP BY full_nm
            ) dbd JOIN coins ON coins.full_nm = dbd.full_nm AND coins.high_price = dbd.price) dd
    GROUP BY full_nm
    ) db JOIN coins ON coins.full_nm = db.full_nm AND coins.dt = db.dt
ORDER BY price DESC, full_name;
