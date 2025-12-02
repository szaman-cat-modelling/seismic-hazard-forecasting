This project uses earthquake data from the AFAD Earthquake Department for the years 1990 to 2025. The raw files are not included in this repository because AFAD data should be downloaded directly from their website.

To run the preprocessing workflow, download the CSV from AFAD and save it in this folder with the name:

data_AFAD.csv

The cleaning notebook will use this file to create the final Turkey earthquake catalogue. The processing steps include filtering to the years 1990 to 2025, converting dates, removing duplicates and keeping only events inside a buffered Turkey boundary. The script will also export two files:

turkey_catalog_clean.geojson  
turkey_catalog_1990_2025.csv

These files are generated automatically, so they do not need to be stored in the repository.

Download AFAD catalogue from https://deprem.afad.gov.tr

Download Turkey boundaries (GADM level 1) from https://gadm.org
