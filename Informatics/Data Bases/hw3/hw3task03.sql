SELECT firstname, surname, ROUND(SUM(slots)::NUMERIC / 20) * 10 AS hours, RANK() OVER(ORDER BY ROUND(SUM(slots)::NUMERIC / 20) * 10 DESC) AS rank
FROM cd.members
JOIN cd.bookings b on members.memid = b.memid
GROUP BY firstname, surname
ORDER BY rank, surname, firstname;
