SELECT facid,
       EXTRACT (MONTH FROM CAST(starttime AS DATE)) AS month,
       SUM(slots) AS total_slots
FROM cd.bookings
WHERE EXTRACT (YEAR FROM CAST(starttime AS DATE)) = 2012
GROUP BY facid, EXTRACT (MONTH FROM CAST(starttime AS DATE))
ORDER BY facid, month;
