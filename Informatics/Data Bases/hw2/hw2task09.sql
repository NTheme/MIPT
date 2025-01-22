SELECT * FROM (SELECT cd.members.firstname || ' ' || cd.members.surname AS member,
                cd.facilities.name AS facility,
                CASE
                    WHEN cd.members.memid = 0 THEN
                        cd.facilities.guestcost * cd.bookings.slots
                    ELSE
                        cd.facilities.membercost * cd.bookings.slots
                    END AS cost
FROM cd.members, cd.facilities, cd.bookings
WHERE cd.members.memid = cd.bookings.memid AND
    cd.bookings.facid = cd.facilities.facid AND
    CAST(cd.bookings.starttime AS DATE) = '2012-09-14') AS db
WHERE cost > 30
ORDER BY cost DESC, member, facility;
