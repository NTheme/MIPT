SELECT cd.members.firstname AS memfname, cd.members.surname AS memsname, db.firstname AS recfname, db.surname AS recsname
FROM cd.members
    LEFT JOIN cd.members AS db ON db.memid = cd.members.recommendedby
ORDER BY memsname, memfname;
