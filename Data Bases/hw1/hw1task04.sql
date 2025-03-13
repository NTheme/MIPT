SELECT id,
    703 * 1.0001092375 * weight / (height * height) AS bmi,
    CASE 
        WHEN 703 * 1.0001092375 * weight / (height * height) < 18.5 THEN
        	'underweight'
        WHEN 703 * 1.0001092375 * weight / (height * height) < 25 THEN
            'normal'
        WHEN 703 * 1.0001092375 * weight / (height * height) < 30 THEN
            'overweight'
        WHEN 703 * 1.0001092375 * weight / (height * height) < 35 THEN
            'obese'
        ELSE
            'extremely obese'
    END AS type
FROM hw
ORDER BY bmi DESC
