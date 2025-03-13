SELECT name,
    CASE
        WHEN monthlymaintenance > 100 THEN
           'expensive'
       else
       'cheap'
       END as cost
FROM cd.facilities;
