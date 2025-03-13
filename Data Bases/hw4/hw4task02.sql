WITH RECURSIVE query AS(
    SELECT 1 AS memid, 'gg'::VARCHAR AS firstname, 'gg'::VARCHAR AS surname
    UNION SELECT db.memid,
                 db.firstname,
                 db.surname
    FROM cd.members AS db, query
    WHERE query.memid = db.recommendedby)
SELECT * FROM query
WHERE memid != 1
ORDER BY memid;
