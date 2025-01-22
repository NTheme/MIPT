CREATE OR REPLACE VIEW v_rec_level_partition_info AS
WITH RECURSIVE Partitions AS (
SELECT
    parent_namespace.nspname AS parent_schema,
    parent_class.    relname AS parent_table,
    child_namespace. nspname AS child_schema,
    child_class.     relname AS child_table,
    1                        AS part_level
FROM pg_inherits
JOIN pg_class parent_class         ON pg_inherits. inhparent     = parent_class.    oid
JOIN pg_class child_class          ON pg_inherits. inhrelid      = child_class.     oid
JOIN pg_namespace parent_namespace ON parent_class.relnamespace  = parent_namespace.oid
JOIN pg_namespace child_namespace  ON child_class. relnamespace  = child_namespace. oid
WHERE parent_class.relkind = 'p'
    UNION ALL
    SELECT
        pt.parent_schema         AS parent_schema,
        pt.parent_table          AS parent_table,
        child_namespace.nspname  AS child_schema,
        child_class.    relname  AS child_table,
        pt.part_level + 1        AS part_level
    FROM Partitions pt
    JOIN pg_inherits pg_inherits ON pg_inherits.inhparent = (
            SELECT pg_class.oid 
            FROM pg_class
            JOIN pg_namespace ON pg_class.relnamespace = pg_namespace.oid
            WHERE pg_namespace.nspname = pt.child_schema AND pg_class.relname = pt.child_table)
    JOIN pg_class child_class         ON pg_inherits.inhrelid     = child_class.    oid
    JOIN pg_namespace child_namespace ON child_class.relnamespace = child_namespace.oid
)
SELECT *
FROM Partitions;
