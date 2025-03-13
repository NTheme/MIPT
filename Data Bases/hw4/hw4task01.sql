WITH RECURSIVE fibonacci AS(
    SELECT 0::numeric AS prefibnumber, 1::numeric AS curfibnumber, 0 AS nth
    UNION SELECT curfibnumber AS prefibnumber,
                 curfibnumber + prefibnumber AS curfibnumber,
                 nth + 1 AS nth
    FROM fibonacci
    WHERE nth < 99)
SELECT nth, curfibnumber AS value FROM fibonacci;
