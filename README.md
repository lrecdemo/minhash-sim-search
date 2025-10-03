# MinHash-LSH Text Clustering Application

A Python application for clustering similar texts using MinHash with Locality Sensitive Hashing (LSH). The system uses configurable Jaccard similarity thresholds to identify and group related documents efficiently. This code accompanies the paper "CorpusClues: Scalable Unsupervised Similarity Search for
Historical Texts Using MinHash-LSH", submitted for LREC2026.  

## Quick start

### Installation

#### Running from source
1. Clone the repository
2. Install dependencies: ```pip install -r requirements.txt```
3. Run the application" ```streamlit run streamlit_app.py```
4. Access the interface at http://localhost:8501


#### Running from Docker
1. Clone the repository
2. Run ```docker-compose up --build -d```

### Usage

#### Data prepoaration
Your CSV file must contain a column named 'text' with the documents to cluster. Additional columns are preserved in the output.

Example CSV structure:
```
id,text
1,"The quck brown fox jumps over the lazy dog."
8,"Packd my box with five dozen liquor jugs."
14,"Sphinx of blck quartz judges my vow."
3,"A quick brown fx jumps over the lazy dog."
11,"Hw vexingly quick daft zebras jumped."
16,"The fve boxing wizards jump quickly!"
7,"Pac my boxes with five dozen liquor jugs."
2,"The quick brown fox jumps over the lazy dogs."
13,"Sphinx of black quartz jdge my vow"
9,"How vexingy quick daft zebras jump."
15,"The fiv boxing wizards jump quickly."
5,"Pack my box with five dozen liquor jugs."
4,"The quick brown fox jumped over the lazy dog."
10,"Hw vexingly quck daft zebras jump!"
6,"Pack my box with fiv dozen liquor jugs"
17,"Five boxing wizards jmp quickly."
12,"Sphin of black quartz judge my vow."
```

#### Configuration

Similarity Threshold:
- 0.1-0.3: Permissive clustering, larger groups, catches loose similarities
- 0.4-0.6: Moderate clustering, balanced approach
- 0.7-0.9: Strict clustering, smaller groups, only strong similarities

Text Pattern Size (Shingle Size):
- 2 characters: `"qu"`, `"ui"`, `"ic"`, `"ck"` from "quick"
- 3 characters: `"qui"`, `"uic"`, `"ick"` from "quick"
- 4 characters: `"quic"`, `"uick"` from "quick"

Text Preprocessing Options:
- Lowercase conversion: Makes comparison case-insensitive
- Diacritics removal: Treats accented and unaccented characters as identical
- Punctuation removal: Ignores punctuation marks in similarity calculation