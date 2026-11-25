# Third-party data and redistribution

## Scope and package contents

The repository contains author-owned aggregate summaries and figure-panel data prepared for replay. It does **not** contain third-party raw datasets, row-level station-month observations, or the row-level table `monthly_matched_station_months.csv`. It also excludes model caches, manuscript review reports, and credentials. A derived result's inclusion does not grant rights to the observations or products from which it was derived.

The source-data review checked official provider terms and records on 2026-11-16; it was not a legal opinion and did not download or redistribute the source datasets. `data/summary/source_manifest.json` records the included author-owned source summaries. The SatPM product manifest, when present in the checkout, is `data/external_manifest/satpm_monthly_files.json`. A hash documents a particular file; it does not establish a license or authorize redistribution.

## License boundaries

- **Original code:** MIT, including the archived project code snapshots and portable adapters; see the repository-root `LICENSE`.
- **Author-owned derived summaries and reference tables/figures:** CC BY 4.0, as scoped in `LICENSE-DATA.md`. Attribution, indication of changes, and no implied endorsement are required.
- **Third-party inputs:** retain the applicable upstream terms. Neither repository license grants additional rights to raw data, provider code, or provider graphics.

In particular, **SatPM V6.GL.03 is CC BY 4.0, with no non-commercial restriction stated for that version**. Do not extend this finding to another SatPM version or to other input products.

## Sources reviewed and packaging decisions

| Source or product | Source-based finding | Package handling |
|---|---|---|
| NASA MAIAC MCD19A2 Collection 6.1 | The LP DAAC product record identifies DOI [10.5067/MODIS/MCD19A2.061](https://doi.org/10.5067/MODIS/MCD19A2.061); LP DAAC/LAADS usage guidance permits reuse/redistribution of current products and requests citation. | Cite the exact collection and pin the granules/date range for a rerun. No raw granules are bundled. |
| NASA VIIRS/FIRMS active-fire data | Official NASA access and citation guidance was reviewed, but a product-specific redistribution license for the exact project extract was not verified. | Do not label the extract CC BY or redistribute raw detections until the exact product terms are checked. |
| ERA5 / Copernicus CDS | The reviewed ERA5 hourly single-level product page states CC BY and lists DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47). Copernicus attribution and disclaimer requirements apply, and product-specific terms can override general terms. | Record the exact product, version, fields, years, and access date. The manuscript's country-specific meteorology provenance is not established as uniformly ERA5. |
| GHSL 2020 epoch | GHSL use conditions state CC BY 4.0, with attribution, change indication, and no implied endorsement. | Identify the exact product and release behind each input before claiming that this finding covers it. |
| GFAS / CAMS | Official GFAS documentation requires acceptance of the applicable CAMS data license and specifies attribution/disclaimer language for public distribution and modified products. | Record the product and accepted terms. Do not call an input CC BY unless its exact terms say so. |
| SatPM V6.GL.03 | The official [V6.GL.03 page](https://www.satpm.org/v6-gl-03) explicitly states CC BY 4.0 and 1998–2024 coverage. The license permits commercial reuse subject to its conditions. | Cite this exact version and listed paper; no non-commercial restriction is stated for V6.GL.03. Do not generalize to other versions. |
| US EPA AQS / CSN | EPA guidance says AQS ambient monitoring data are public domain; EPA describes CSN records as reported to AQS. | Apply this only to observations actually obtained from AQS. Cite source and access date; this does not cover third-party graphics or other sources. |
| Canada NAPS | The official [NAPS dataset record](https://open.canada.ca/data/en/dataset/1b36a356-defd-4813-acea-47bc3abd859b?lang=en) lists the Open Government Licence – Canada, which permits reuse subject to attribution. | Cite the Government of Canada record and license. Confirm inherited/provincial records are covered by that source or clear them separately. |
| OpenAQ-derived cohorts | OpenAQ terms require compliance with upstream provider terms and attribution. Location-specific license and time-period metadata vary; there is no blanket license for all locations. | For a rerun, resolve and record the provider and license for every used location and period. Do not redistribute row-level observations until this is done. |
| Queensland monitoring | The reviewed [Air Quality Monitoring – 2020 record](https://www.data.qld.gov.au/dataset/air-quality-monitoring-2020) lists CC BY 4.0 for the described hourly 2020 observations. | Apply only if the project file matches this exact resource. Verify other years and archived sources separately. |

These findings rely on official records and policies. Additional primary policy references are [NASA/LAADS](https://ladsweb.modaps.eosdis.nasa.gov/learn/), [FIRMS](https://firms.modaps.eosdis.nasa.gov/web-services/), [Copernicus terms](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products), [GHSL](https://human-settlement.emergency.copernicus.eu/GHSLhowToCite.php), [GFAS](https://confluence.ecmwf.int/pages/viewpage.action?pageId=88257765), [EPA AQS permission guidance](https://www.epa.gov/outdoor-air-quality-data/do-i-need-request-permission-use-monitoring-data-and-graphics-airdata), [Canada licence](https://open.canada.ca/en/open-government-licence-canada), and [OpenAQ terms](https://docs.openaq.org/about/terms). These policy links are not a substitute for checking the exact files used in a future rerun.

## Before redistributing or making a raw-data rerun

Resolve and record the exact product IDs, versions, years, access dates, source URLs/DOIs, file hashes, and attribution/disclaimer text for each input. In particular, verify the exact FIRMS products, the OpenAQ provider/location licenses, the GHSL product behind each 2020-epoch input, and whether any Queensland input matches the cited 2020 record. Keep raw data outside this repository unless the exact source terms permit redistribution.
