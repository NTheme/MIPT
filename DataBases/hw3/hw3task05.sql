SELECT name, row_number() OVER(ORDER BY SUM(CASE
                    WHEN cd.bookings.memid = 0 THEN
                        cd.facilities.guestcost * cd.bookings.slots
                    ELSE
                        cd.facilities.membercost * cd.bookings.slots
                    END) DESC) AS rank
FROM cd.facilities, cd.bookings
GROUP BY name
ORDER BY rank, name
LIMIT 3;
