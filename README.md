# AutoRA: Unlocking Smarter Research with Automated Paper Search and Analysis

![](resources/003-logo.png)

## Features

- [x] Fetch the complete conference proceedings from the DBLP and ACL Anthology.
- [x] Perform semantic search based on your keywords of interest with the help of LLMs.
- [x] Perform the semantic search from a Flask-based portal.
- [x] Flag the interested papers with red background.
- [ ] Improve the presentation of the Flask-based portal by better formatting the special characters.
- [ ] Improve the user-friendliness of the system by automating everything except (1) creating the environment, and (2) `python app.py`.
- [ ] Add (1) a selector to filter the papers based on years and venues, (2) a checker box or slider on how many papers to return.
- [ ] Add a progress bar that tracks the browsing progress and use it start from what is left.
- [ ] Maintain a history of previous queries.


![](resources/004-demo.png)

## Data Sources

| Conference              | Source        | Note |
| ----------------------- | ------------- | ---- |
| NLP Conferences | ACL Anthology |      |
| All Other Conferences and Preprints | The arXiv Metadata Hosted on [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download) |      |

In the future, we may be able to obtain the full bibliography for each of the conferences listed below:

| Conference              | Source        | Note |
| ----------------------- | ------------- | ---- |
| ICSE | IEEE |  |
| ESEC                    | ACM           |      |
| FSE                     | ACM           |      |
| ISSTA                   | ACM           |      |
| MSR                     | IEEE and ACM  |      |
| ASE | IEEE and ACM | |
| ISSRE | IEEE | |
| ICSME | IEEE | |