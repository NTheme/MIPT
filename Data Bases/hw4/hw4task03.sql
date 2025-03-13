CREATE OR REPLACE VIEW v_first_level_partition_info AS
SELECT
    parent_namespace.nspname AS parent_schema,
    parent_class.relname     AS parent_table,
    child_namespace.nspname  AS child_schema,
    child_class.relname      AS child_table
FROM pg_inherits
JOIN pg_class parent_class         ON pg_inherits.inhparent      = parent_class.oid
JOIN pg_class child_class          ON pg_inherits.inhrelid       = child_class.oid
JOIN pg_namespace parent_namespace ON parent_class.relnamespace  = parent_namespace.oid
JOIN pg_namespace child_namespace  ON child_class.relnamespace   = child_namespace.oid
WHERE parent_class.relkind = 'p' AND
      child_class.relkind = 'p' AND
      NOT EXISTS (SELECT *
                  FROM information_schema.views
                  WHERE table_schema = 'public' AND
                        table_name = 'v_first_level_partition_info');
