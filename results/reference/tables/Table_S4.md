Table S4. Predictor definitions and documented source scope, including the distinction between later and earlier meteorological input generations and the remaining provenance limits.

| Field | Definition and unit | Documented construction | Interpretation / source scope |
| --- | --- | --- | --- |
| AOD | Observed QA-cell value, otherwise supplied 9 × 9 fill; unitless | Downstream raw/fill/used merge invariant; historical USA/CAN source census | Complete raw-granule reconstruction unavailable |
| Population | GHSL 2020; people per 0.1° cell; log10(1+population) | Historical USA/CAN grid and specified future inputs | Other historical raster-to-field mappings incomplete |
| blh | Monthly mean boundary-layer height, m | Historical USA/CAN grid; regional ERA5 records | Source generations remain distinct |
| wind10 | hypot(monthly mean u10,v10), m s⁻¹ | Historical USA/CAN grid; regional ERA5 records | Not the monthly mean of hourly wind-speed magnitude |
| tp | MODA mean daily accumulated precipitation × 1000, mm day⁻¹ | Historical USA/CAN grid; regional ERA5 records | Legacy mm month⁻¹ comments do not describe these values |
| t2m | Monthly mean 2-m temperature: K − 273.15, °C | Historical USA/CAN grid; regional ERA5 records | Source-generation equivalence not established for every country |
| rh | Monthly-temperature/dewpoint Magnus expression; clipped 1–100% | 17.67/243.5 Magnus coefficients in historical code and later regional helper | Monthly fields; earlier regional helper hash unresolved |
| vent | blh × wind10, m² s⁻¹ | Historical USA/CAN grid; regional ERA5 records | Product of monthly fields; not mean hourly ventilation |
| MET fill | Only nonfinite values filled; prior finite values unchanged | The documented regional fill generation | Does not establish identical generation in every selected table |
