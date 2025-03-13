SELECT facid,
       EXTRACT (MONTH FROM CAST(starttime AS DATE)) AS month,
       SUM(slots) AS slots
FROM cd.bookings
WHERE EXTRACT (YEAR FROM CAST(starttime AS DATE)) = 2012
GROUP BY facid, EXTRACT (MONTH FROM CAST(starttime AS DATE))
UNION (SELECT facid, NULL AS month, sum(slots) as slots
    FROM cd.bookings
    WHERE EXTRACT (YEAR FROM CAST(starttime AS DATE)) = 2012
GROUP BY facid
ORDER BY facid)
UNION (SELECT NULL AS facid, NULL AS month, sum(slots) as slots
    FROM cd.bookings
    WHERE EXTRACT (YEAR FROM CAST(starttime AS DATE)) = 2012)
ORDER BY facid, month;
