import os
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
import json
import re


data = []


# --------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# --------------------------------
# FESTIVAL KEYWORDS
# --------------------------------

festival_keywords = [
    "holi", "diwali", "ram navmi", "janmashtami",
    "eid", "bakrid", "christmas", "dussehra",
    "guru nanak jayanti", "mahavir jayanti"
]


# --------------------------------
# PDF EXTRACTION
# --------------------------------

print("Extracting PDFs...")

pdf_folder = "data/pdfs"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        path = os.path.join(pdf_folder, file)

        try:

            with pdfplumber.open(path) as pdf:

                for page_num, page in enumerate(pdf.pages):

                    # -------- Normal text --------

                    text = page.extract_text()

                    if text:

                        cleaned = clean_text(text)

                        if cleaned:

                            data.append({
                                "text": cleaned,
                                "source": file,
                                "url": path,
                                "page": page_num + 1
                            })


                    # -------- Table extraction --------

                    tables = page.extract_tables()

                    if tables:

                        for table in tables:

                            for row in table:

                                row = [
                                    str(cell).strip()
                                    for cell in row
                                    if cell
                                ]

                                if len(row) < 2:
                                    continue


                                # Handle multiple calendar formats

                                date = None
                                event = None

                                if len(row) == 2:

                                    date = row[0]
                                    event = row[1]

                                elif len(row) >= 3:

                                    date = row[1]
                                    event = row[2]


                                if not date or not event:
                                    continue


                                event_clean = clean_text(event)


                                # Detect festival

                                if any(
                                    f in event_clean.lower()
                                    for f in festival_keywords
                                ):

                                    sentence = (
                                        f"{event_clean} festival occurs on "
                                        f"{date} according to the academic calendar."
                                    )

                                else:

                                    sentence = (
                                        f"{event_clean} occurs on "
                                        f"{date} according to the academic calendar."
                                    )


                                sentence = clean_text(sentence)


                                if sentence:

                                    data.append({
                                        "text": sentence,
                                        "source": file,
                                        "url": path,
                                        "page": page_num + 1
                                    })


        except Exception as e:

            print(
                "Error reading PDF:",
                file,
                e
            )


# --------------------------------
# HTML EXTRACTION
# --------------------------------

print("Extracting HTML files...")

html_folder = "data/html"

for file in os.listdir(html_folder):

    if file.endswith(".html"):

        path = os.path.join(
            html_folder,
            file
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                soup = BeautifulSoup(
                    f,
                    "html.parser"
                )


                # Remove unwanted HTML elements

                for tag in soup(
                    [
                        "script",
                        "style",
                        "nav",
                        "footer",
                        "header"
                    ]
                ):

                    tag.decompose()


                text = soup.get_text()

                cleaned = clean_text(text)


                if cleaned:

                    data.append({
                        "text": cleaned,
                        "source": file,
                        "url": path,
                        "page": 1
                    })


        except Exception as e:

            print(
                "Error reading HTML:",
                file,
                e
            )


# --------------------------------
# EXCEL EXTRACTION
# --------------------------------

print("Extracting Excel files...")

excel_folder = "data/excel"

for file in os.listdir(excel_folder):

    if file.endswith(".xlsx") or file.endswith(".xls"):

        path = os.path.join(
            excel_folder,
            file
        )

        try:

            df = pd.read_excel(path)

            text = df.to_string(
                index=False
            )

            cleaned = clean_text(text)


            if cleaned:

                data.append({
                    "text": cleaned,
                    "source": file,
                    "url": path,
                    "page": 1
                })


        except Exception as e:

            print(
                "Error reading Excel:",
                file,
                e
            )


# --------------------------------
# SAVE OUTPUT
# --------------------------------

os.makedirs(
    "processed",
    exist_ok=True
)


output_file = "processed/extracted_raw.json"


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\nExtraction completed successfully!")

print(
    "Total extracted records:",
    len(data)
)

print(
    "Saved to:",
    output_file
)