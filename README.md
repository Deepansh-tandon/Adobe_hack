## 📄 PDF Title & Heading Extractor

This project extracts **titles** and **headings** (H1, H2, H3) from PDF files using layout analysis, NLP (spaCy), and heuristics. It's containerized using Docker for easy deployment and consistent performance.

---

### 🚀 Features

* Extracts the main document **title** and **structured headings**
* Uses **spaCy** (`en_core_web_md`) for semantic understanding
* Ignores **headers**, **footers**, and **table content**
* Assigns heading levels (`H1`, `H2`, `H3`) using font size, boldness, and numbering
* Optionally generates **visual debug images**

---

## 🧱 Project Structure

```
project/
├── Dockerfile
├── requirements.txt
├── process_pdf.py
├── input/                # Place your PDF files here
└── output/               # Extracted JSONs and visual outputs
```

---

## 🐳 Getting Started with Docker

### 1. Build the Docker Image

```bash
docker build -t pdf-heading-extractor .
```

### 2. Run the Container

Mount `input/` and `output/` directories to persist data:

```bash
docker run \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  pdf-heading-extractor
```

### 3. Optional Flags

You can pass CLI flags defined in the script:

| Flag                | Description                                 |
| ------------------- | ------------------------------------------- |
| `--verbose`         | Enables DEBUG logging                       |
| `--debug-visualize` | Saves annotated images of headings detected |
| `--keep-duplicates` | Keeps headings with the same text           |

Example:

```bash
docker run \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  pdf-heading-extractor --verbose --debug-visualize
```

---

## 🧪 Example Output

For a given `sample.pdf`:

```json
{
  "title": "The Future of AI Research",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Background and Motivation", "page": 2},
    {"level": "H2", "text": "Methodology", "page": 3},
    {"level": "H1", "text": "Conclusion", "page": 6}
  ]
}
```

---

## 📦 Dependencies

Installed via `requirements.txt`:

```txt
pdfplumber
spacy
scikit-learn
numpy
pillow
```

Also installs `en_core_web_md` during build.

---

## 📁 Output

* Extracted structured JSON per PDF in `output/`
* Optional debug images in `output/debug_images/` (when `--debug-visualize` is used)

---

## 📌 Notes

* The container runs the script automatically on all PDFs inside `/app/input`
* Ideal for automation pipelines or batch processing

---