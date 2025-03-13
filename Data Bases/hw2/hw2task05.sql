SELECT surname, firstname, cd.members.memid, MIN(starttime) AS starttime
FROM cd.members
    JOIN cd.bookings ON cd.members.memid = cd.bookings.memid
WHERE starttime >= '2012-09-01'
GROUP BY cd.members.memid, surname, firstname
ORDER BY memid;
