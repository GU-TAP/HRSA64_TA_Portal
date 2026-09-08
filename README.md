# HRSA64 TA Portal & Monthly Report Pipeline

Code base for the TA portal supporting the HRSA64 project *"Ending the HIV Epidemic in the U.S. — Technical Assistance Provider."*

## Repository contents

- [`HRSA64.py`](https://github.com/GU-TAP/HRSA64_TA_Portal/blob/main/HRSA64.py) — main script for the TA portal
- [`Monthly_report_pipeline`](https://github.com/GU-TAP/HRSA64_TA_Portal/tree/main/Monthly_report_pipeline) — monthly report pipeline

The TA portal is deployed on Streamlit: **https://hrsagutap.streamlit.app/**

## Data sources

Two Google Sheets are used for data management:

| Sheet | Link |
|---|---|
| TA Portal | [Open](https://docs.google.com/spreadsheets/d/1qmP6SAtFGyc6v-KEsUO3-eX_tTQ06vWlNa3cCZFHg8M) |
| Additional TA activities | [Open](https://docs.google.com/spreadsheets/d/1NxnuTv74Nka2GQob61SWNHbs2vqI-QKgYXiyfYbLCrI) |

## File storage

Uploads from jurisdictions, TA provider interactions and deliveries, travel spend authorization forms, GSA exemption forms, and similar files are stored in Google Drive:

- [Shared Drive folder](https://drive.google.com/drive/folders/1Q9dMMdyfEGWFVv2_CbHbJVMHXOST3OYf)

## Running the monthly report

1. Clone this repository to your local machine.
2. Download the latest data sheets into the repository folder.
3. Open a terminal and change into that directory:

```bash
   cd path/to/HRSA64_TA_Portal
   python report.py
```

An HTML file stamped with today's date is generated in the same folder.

## Adding users

All login credentials are stored in `secrets.toml` on the Streamlit platform — never in this repository. To add a person, follow this format:

```toml
[users."EMAIL"."ROLE"]
password = "xxxx"
name = "xxx"
```

Valid roles: `Coordinator`, `Assignee/Staff`, `Research Assistant`.