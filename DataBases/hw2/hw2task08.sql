SELECT DISTINCT firstname || ' ' || members.surname AS member, c.name AS facility 
FROM cd.members 
    JOIN cd.bookings b on b.memid = cd.members.memid 
    JOIN cd.facilities c ON c.facid = b.facid 
WHERE c.name LIKE 'Tennis Court%' 
ORDER BY member;
