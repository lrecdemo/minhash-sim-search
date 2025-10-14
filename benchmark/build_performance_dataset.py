#!/usr/bin/env python3
import requests
import xml.etree.ElementTree as ET
import pandas as pd

def main():
    base_url = "https://raw.githubusercontent.com/alekkeersmaekers/duke-nlp/master/xml/"
    xml_files = [
        "Papyri_Accounts.xml", "Papyri_Administration.xml", "Papyri_Contracts1.xml",
        "Papyri_Contracts2.xml", "Papyri_Contracts3.xml", "Papyri_Declarations1.xml",
        "Papyri_Declarations2.xml", "Papyri_Labels.xml", "Papyri_Letters1.xml",
        "Papyri_Letters2.xml", "Papyri_Letters3.xml", "Papyri_Lists1.xml",
        "Papyri_Lists2.xml", "Papyri_Lists3.xml", "Papyri_Lists4.xml",
        "Papyri_Other.xml", "Papyri_Paraliterary.xml", "Papyri_Pronouncements.xml",
        "Papyri_Receipts1.xml", "Papyri_Receipts2.xml", "Papyri_Reports.xml"
    ]

    rows = []
    for filename in xml_files:
        url = base_url + filename
        response = requests.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for sent in root.findall('sentence'):
            sentence_id = sent.attrib.get('id')
            document_id = sent.attrib.get('document_id')
            filename_attr = sent.attrib.get('filename')
            period_min = sent.attrib.get('period_min')
            period_max = sent.attrib.get('period_max')
            genre = sent.attrib.get('genre')
            place = sent.attrib.get('place')

            words = []
            for word in sent.findall('word'):
                form = word.attrib.get('regularized')
                if form:
                    words.append(form)
            sentence_text = ' '.join(words)

            rows.append({
                'sentence_id': sentence_id,
                'document_id': document_id,
                'filename': filename_attr,
                'period_min': period_min,
                'period_max': period_max,
                'genre': genre,
                'place': place,
                'sentence': sentence_text
            })

    df = pd.DataFrame(rows)
    df_no_gaps = df[~df['sentence'].str.contains(r'\|gap=', regex=True)]
    df_no_gaps = df_no_gaps.reset_index(drop=True)
    df_no_gaps['sentence'] = df_no_gaps['sentence'].str.replace(r'\[0\]$', '', regex=True)
    df_no_gaps.to_csv('papyri_sentences_no_gaps_clean.csv', index=False)
    print("✅ CSV saved: papyri_sentences_no_gaps_clean.csv")

if __name__ == "__main__":
    main()
