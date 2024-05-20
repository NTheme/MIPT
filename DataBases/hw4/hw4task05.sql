CREATE OR REPLACE VIEW v_used_size_per_user AS
WITH user_schema_table_size AS (
        SELECT
            pg_tables.tableowner AS table_owner,
            pg_tables.schemaname AS schema_name,
            pg_tables.tablename  AS table_name,
            pg_size_pretty(pg_relation_size(quote_ident(pg_tables.schemaname) || '.' || quote_ident(pg_tables.tablename))) AS table_size
        FROM pg_catalog.pg_tables),
    user_schema_total_size AS (
        SELECT
            pg_tables.tableowner AS table_owner,
            pg_tables.schemaname AS schema_name,
            pg_size_pretty(SUM(pg_relation_size(quote_ident(pg_tables.schemaname) || '.' || quote_ident(pg_tables.tablename)))) AS used_per_schema_user_total_size
        FROM pg_catalog.pg_tables
        GROUP BY pg_tables.tableowner, pg_tables.schemaname),
    user_total_size AS (
        SELECT
            pg_tables.tableowner AS table_owner,
            pg_size_pretty(SUM(pg_relation_size(quote_ident(pg_tables.schemaname) || '.' || quote_ident(pg_tables.tablename)))) AS used_user_total_size
        FROM pg_catalog.pg_tables
        GROUP BY pg_tables.tableowner)

SELECT
    user_schema_table_size.table_owner,
    user_schema_table_size.schema_name,
    user_schema_table_size.table_name,
    user_schema_table_size.table_size,
    user_schema_total_size.used_per_schema_user_total_size,
    user_total_size.used_user_total_size
FROM user_schema_table_size
JOIN user_schema_total_size ON user_schema_table_size.table_owner = user_schema_total_size.table_owner AND
                               user_schema_table_size.schema_name = user_schema_total_size.schema_name
JOIN user_total_size ON user_schema_table_size.table_owner = user_total_size.table_owner;
