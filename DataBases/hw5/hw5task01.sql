CREATE OR REPLACE FUNCTION count_non_volatile_days(full_nm_var TEXT) RETURNS INTEGER AS
$$
DECLARE
    cnt    INTEGER;
    exists BOOLEAN;
BEGIN
    SELECT EXISTS(SELECT 1 FROM coins WHERE LOWER(full_nm) = LOWER(full_nm_var)) INTO exists;

    IF NOT exists THEN
        RAISE EXCEPTION 'Crypto currency with name "%" is absent in database!', full_nm_var USING ERRCODE = '02000';
    END IF;

    SELECT COUNT(*)
    INTO cnt
    FROM coins
    WHERE LOWER(full_nm) = LOWER(full_nm_var)
      AND open_price = close_price AND open_price = high_price AND open_price = low_price;

    RETURN cnt;
END;
$$ LANGUAGE plpgsql;
