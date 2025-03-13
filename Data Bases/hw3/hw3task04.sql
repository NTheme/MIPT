SELECT name,
       CASE
           WHEN db.rev = 1 THEN
               'high'
           WHEN db.rev = 2 THEN
               'average'
           ELSE
               'low'
           END AS revenue
FROM (SELECT name, NTILE(3) OVER(ORDER BY SUM(CASE
                    WHEN cd.bookings.memid = 0 THEN
                        cd.facilities.guestcost * cd.bookings.slots
                    ELSE
                        cd.facilities.membercost * cd.bookings.slots
                    END) DESC) AS rev
FROM cd.facilities, cd.bookings
GROUP BY name
ORDER BY rev, name) AS db;
