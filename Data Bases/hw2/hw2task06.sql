SELECT DISTINCT cd.members.firstname, cd.members.surname
FROM cd.members
    JOIN cd.members AS db ON db.recommendedby = cd.members.memid
ORDER BY surname, firstname;
