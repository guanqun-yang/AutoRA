# AutoRA: Unlocking Smarter Research with Automated Paper Search and Analysis

![](resources/003-logo.png)
## Environment

- The code was developed on a gaming laptop equipped with an RTX-2060 GPU (6GB VRAM) running Ubuntu 20.04. While the code should be adaptable to other hardware configurations, compatibility may vary.
- Run the following to install the library:

```bash
conda create --name autora python==3.10
pip install -e .
python app.py
```

## Features

 - [x] Retrieve complete conference proceedings from DBLP and the ACL Anthology.
 - [x] Conduct semantic searches using `sentence_transformers` embeddings based on your chosen keywords.
 - [x] Access the semantic search through a Flask-based portal.
 - [x] Highlight selected papers with a red background.
 - [ ] Enhance the Flask portal's presentation by properly formatting special characters.
 - [x] Increase system user-friendliness by automating all tasks except (1) environment setup and (2) running `python app.py`.
 - [ ] Add features to filter papers by year and venue, and include a checkbox or slider to control the number of papers returned.
 - [ ] Implement a progress bar that tracks browsing and resumes from where you left off.
 - [ ] Maintain a history of previous search queries.


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