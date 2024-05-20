SELECT cd.members.firstname || ' ' || cd.members.surname AS member,
       (SELECT db.firstname || ' ' || db.surname 
        FROM cd.members as db 
        WHERE cd.members.recommendedby = db.memid) AS recommender
FROM cd.members
ORDER BY member, recommender;
