CREATE OR REPLACE FUNCTION serial_generator(start_val_inc INTEGER, last_val_ex INTEGER)
    RETURNS TABLE
            (
                serial INT
            )
AS
$$
BEGIN
    serial := start_val_inc;
    WHILE serial < last_val_ex
        LOOP
            RETURN NEXT;
            serial := serial + 1;
        END LOOP;
    RETURN;
END;
$$ LANGUAGE plpgsql;
