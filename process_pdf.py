import os
import json
import pandas as pd
import joblib
import sys
import numpy as np
import fitz  # PyMuPDF for title extraction
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import re
from collections import defaultdict
import argparse
import logging

# --- CONFIGURATION & SETUP ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Add the main directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'main'))

try:
    from feature_extraction import main as extract_features
    logging.info("Successfully imported feature_extraction module")
except ImportError:
    logging.warning("Could not import feature_extraction module. Creating a fallback version.")
    def extract_features(pdf_path):
        """Fallback feature extraction using PyMuPDF"""
        import fitz
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                "text": span["text"].strip(),
                                "x0": span["bbox"][0],
                                "y0": span["bbox"][1],
                                "x1": span["bbox"][2],
                                "y1": span["bbox"][3],
                                "font_size": span["size"],
                                "font": span["font"],
                                "page": page_num + 1,
                                "is_bold": bool("Bold" in span["font"] or span["flags"] & 2**4),
                                "is_centred": False,  # Will be calculated later
                                "width": span["bbox"][2] - span["bbox"][0],
                                "height": span["bbox"][3] - span["bbox"][1],
                                "font_size_rank": 1,  # Will be calculated later
                                "center_x_norm": (span["bbox"][0] + span["bbox"][2]) / 2 / page.rect.width,
                                "y_norm": span["bbox"][1] / page.rect.height,
                                "num_spans": 1,
                                "spacing_above": 0,
                                "spacing_below": 0,
                                "is_list_item": False,
                                "is_section_header": False
                            })
        
        doc.close()
        return [block for block in text_blocks if block["text"]]

class OptimizedPostProcessor:
    def __init__(self):
        # Configurable thresholds
        self.max_title_words = 12
        self.min_heading_words = 2
        self.max_heading_words = 15
        self.footer_threshold = 50  # pixels from bottom
        self.title_case_threshold = 0.7
        self.noun_ratio_threshold = 0.4
        self.stopword_ratio_threshold = 0.3
        
        # Pre-compiled regex patterns
        self.punctuation_pattern = re.compile(r"[.!?]$")
        self.end_punctuation_pattern = re.compile(r"[.!?;:]$")
        
        # Efficient stopword set
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'
        }
        
        # Common verbs (no POS tagging needed)
        self.common_verbs = {
            'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
            'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must'
        }
        
        self.heading_levels = ['Title', 'H1', 'H2', 'H3', 'H4', 'body']

    def process(self, blocks, page_height=842):
        """Optimized processing pipeline"""
        if not blocks:
            return []

        seen_title = False
        processed_blocks = []
        prev_heading_level = None

        for block in blocks:
            text = block.get('text', '').strip()
            if not text:
                continue

            current_level = block.get('level', 'body')
            page = block.get('page', 1)
            y0 = block.get('y0', 0)
            word_count = len(text.split())
            
            # ---- Fast Text Checks ----
            is_allcaps = text.isupper() and any(c.isalpha() for c in text)
            ends_with_punct = bool(self.punctuation_pattern.search(text))
            
            # Rule 1: Title handling
            if current_level == 'Title':
                if seen_title or page > 1:
                    current_level = 'H1'
                else:
                    seen_title = True
            
            # Rule 2: Length-based demotion
            if current_level != 'body' and word_count > self.max_heading_words:
                current_level = 'body'
            
            # Rule 3: All-caps promotion
            if is_allcaps and word_count <= 5 and current_level not in ['Title', 'H1']:
                current_level = 'H1'
            
            # Rule 4: Punctuation demotion
            if ends_with_punct and current_level != 'body':
                current_level = 'body'
            
            # Rule 5: Footer detection
            if y0 > (page_height - self.footer_threshold):
                current_level = 'footer'
            
            # ---- Smart Heading Detection ----
            if current_level == 'body' and self.min_heading_words <= word_count <= self.max_heading_words:
                # Fast title case check
                title_case_score = sum(1 for word in text.split() if word and word[0].isupper()) / word_count
                
                # Fast verb detection
                words_lower = text.lower().split()
                verb_score = sum(1 for word in words_lower if word in self.common_verbs) / word_count
                
                # Stopword ratio
                stopword_ratio = sum(1 for word in words_lower if word in self.stopwords) / word_count
                
                # Heading likelihood score
                heading_score = (
                    0.4 * (title_case_score > self.title_case_threshold) +
                    0.3 * (stopword_ratio < self.stopword_ratio_threshold) +
                    0.3 * (verb_score < 0.2)
                )
                
                if heading_score >= 0.7:
                    current_level = self._suggest_heading_level(block)
            
            # Update block
            block['level'] = current_level
            processed_blocks.append(block)
            
            # Track heading hierarchy
            if current_level != 'body':
                prev_heading_level = current_level

        return self._validate_hierarchy(processed_blocks)

    def _suggest_heading_level(self, block):
        """Efficient heading level suggestion"""
        font_size = block.get('font_size', 12)
        word_count = len(block['text'].split())
        
        if font_size > 18 or word_count <= 3:
            return 'H1'
        elif font_size > 14 or word_count <= 6:
            return 'H2'
        else:
            return 'H3'

    def _validate_hierarchy(self, blocks):
        """Lightweight hierarchy validation"""
        current_level_idx = 0
        
        for block in blocks:
            level = block['level']
            if level == 'body':
                continue
                
            try:
                level_idx = self.heading_levels.index(level)
            except ValueError:
                continue
                
            # Prevent skipping more than one level
            if level_idx > current_level_idx + 1:
                new_level = self.heading_levels[current_level_idx + 1]
                block['level'] = new_level
                level_idx = current_level_idx + 1
                
            current_level_idx = level_idx
            
        return blocks

def extract_title_with_fitz(pdf_path, top_region=0.7, min_title_length=5):
    """
    Extracts the most probable title from a PDF by analyzing the top portion of the first page.
    """
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return ""
        
        first_page = doc[0]
        page_height = first_page.rect.height
        crop_area = fitz.Rect(0, 0, first_page.rect.width, page_height * top_region)
        
        # Extract text blocks with basic formatting info
        blocks = first_page.get_text("blocks", clip=crop_area)
        doc.close()
        
        # Filter valid candidates
        candidates = []
        for block in blocks:
            x0, y0, x1, y1, text, block_type, _ = block
            text = text.strip()
            if (block_type == 0 and  # Only text blocks
                len(text) >= min_title_length and
                not text.isdigit() and
                not text.startswith(('http://', 'https://'))):
                
                candidates.append({
                    "text": text,
                    "y_pos": y0,
                    "height": y1 - y0,  # Block height as proxy for font size
                    "width": x1 - x0
                })
        
        if not candidates:
            return ""
        
        # Prepare features for clustering (position + size)
        features = np.array([
            [c["y_pos"]/page_height, c["height"]*10]  # Normalized features
            for c in candidates
        ])
        
        # Cluster using DBSCAN
        features = StandardScaler().fit_transform(features)
        clustering = DBSCAN(eps=0.3, min_samples=1).fit(features)
        
        # Score clusters (higher = better title candidate)
        best_title = ""
        best_score = -1
        
        for cluster_id in set(clustering.labels_):
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_items = [candidates[i] for i in cluster_indices]
            
            # Calculate cluster score
            avg_height = np.mean([c["height"] for c in cluster_items])
            min_y_pos = np.min([c["y_pos"] for c in cluster_items])
            text_length = np.mean([len(c["text"]) for c in cluster_items])
            
            score = (avg_height * 0.6 +        # Size matters most
                     (1 - min_y_pos/page_height) * 0.3 +  # Prefer higher position
                     text_length * 0.1)         # Longer text preferred
            
            if score > best_score:
                best_score = score
                # Combine all text in cluster (sorted top-to-bottom)
                cluster_items.sort(key=lambda x: x["y_pos"])
                best_title = " ".join([item["text"] for item in cluster_items])
        
        return best_title.strip()
    
    except Exception as e:
        logging.warning(f"Title extraction error: {e}")
        return ""

def add_simple_nlp_features(text):
    """
    Add simple NLP features without requiring NLTK downloads
    """
    if not text:
        return {
            'stopword_ratio': 0.0,
            'capitalization_ratio': 0.0,
            'punctuation_count': 0,
            'verb_count': 0
        }
    
    # Simple stopwords list
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
    
    words = text.lower().split()
    if not words:
        return {
            'stopword_ratio': 0.0,
            'capitalization_ratio': 0.0,
            'punctuation_count': 0,
            'verb_count': 0
        }
    
    # Stopword ratio
    stopword_count = sum(1 for word in words if word in stopwords)
    stopword_ratio = stopword_count / len(words)
    
    # Capitalization ratio
    original_words = text.split()
    capitalized_count = sum(1 for word in original_words if word and word[0].isupper())
    capitalization_ratio = capitalized_count / len(original_words) if original_words else 0.0
    
    # Punctuation count
    punctuation_chars = '.,!?;:()[]{}"\'-'
    punctuation_count = sum(1 for char in text if char in punctuation_chars)
    
    # Simple verb detection (basic approach)
    common_verbs = {'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'go', 'goes', 'went', 'gone', 'see', 'sees', 'saw', 'seen', 'come', 'comes', 'came', 'make', 'makes', 'made', 'take', 'takes', 'took', 'taken', 'get', 'gets', 'got', 'gotten', 'know', 'knows', 'knew', 'known', 'think', 'thinks', 'thought', 'say', 'says', 'said', 'tell', 'tells', 'told', 'find', 'finds', 'found', 'give', 'gives', 'gave', 'given', 'show', 'shows', 'showed', 'shown', 'work', 'works', 'worked', 'call', 'calls', 'called', 'try', 'tries', 'tried', 'ask', 'asks', 'asked', 'need', 'needs', 'needed', 'feel', 'feels', 'felt', 'become', 'becomes', 'became', 'become', 'leave', 'leaves', 'left', 'put', 'puts', 'put', 'mean', 'means', 'meant', 'keep', 'keeps', 'kept', 'let', 'lets', 'let', 'begin', 'begins', 'began', 'begun', 'seem', 'seems', 'seemed', 'help', 'helps', 'helped', 'talk', 'talks', 'talked', 'turn', 'turns', 'turned', 'start', 'starts', 'started', 'might', 'could', 'would', 'should', 'may', 'can', 'will', 'shall', 'must'}
    verb_count = sum(1 for word in words if word in common_verbs)
    
    return {
        'stopword_ratio': stopword_ratio,
        'capitalization_ratio': capitalization_ratio,
        'punctuation_count': punctuation_count,
        'verb_count': verb_count
    }

def process_pdf_with_xgboost(pdf_path, xgboost_model_path="./xgboost_model.joblib", args=None):
    """
    Process a PDF file using the XGBoost model for classification
    """
    logging.info(f"📄 Processing {pdf_path}...")
    
    # 1. Extract the document title first
    document_title = extract_title_with_fitz(pdf_path)
    if document_title:
        logging.info(f"Detected document title: {document_title}")
    
    # 2. Extract features from PDF
    text_blocks = extract_features(pdf_path)
    
    if not text_blocks:
        logging.error("No text blocks extracted from PDF")
        return {"title": document_title, "outline": []}
    
    logging.debug(f"Extracted {len(text_blocks)} text blocks")
    
    # 3. Add simple NLP features
    for block in text_blocks:
        nlp_features = add_simple_nlp_features(block["text"])
        block.update(nlp_features)
    
    # 4. Convert to DataFrame
    df = pd.DataFrame(text_blocks)
    
    # 5. Check if XGBoost model exists, if not fall back to rule-based processing
    if not os.path.exists(xgboost_model_path):
        logging.warning(f"XGBoost model not found at {xgboost_model_path}. Using rule-based fallback.")
        return process_pdf_fallback(df, document_title, args)
    
    # 6. Preprocess features (same as training)
    df_for_prediction = df.copy()
    
    # Drop unnecessary columns for XGBoost prediction
    drop_cols = ['text', 'font', 'page', 'font_size', 'width', 'x1', 'y1', 'center_x_norm', 
                 'y_norm', 'num_spans', 'spacing_above', 'spacing_below', 'is_list_item', 'is_section_header']
    df_for_prediction = df_for_prediction.drop(columns=[c for c in drop_cols if c in df_for_prediction.columns], errors='ignore')
    
    # Convert boolean and object columns to integers/floats
    bool_cols = df_for_prediction.select_dtypes(include='bool').columns
    df_for_prediction[bool_cols] = df_for_prediction[bool_cols].astype(int)
    
    # Convert object columns to numeric where possible
    for col in df_for_prediction.select_dtypes(include='object').columns:
        if col == 'is_bold' or col == 'is_centred':
            df_for_prediction[col] = df_for_prediction[col].astype(int)
        else:
            try:
                df_for_prediction[col] = pd.to_numeric(df_for_prediction[col], errors='coerce')
            except:
                df_for_prediction = df_for_prediction.drop(columns=[col])
    
    # Drop rows with missing values
    df_for_prediction = df_for_prediction.dropna()
    
    if df_for_prediction.empty:
        logging.warning("No valid data after preprocessing. Using fallback.")
        return process_pdf_fallback(df, document_title, args)
    
    # 7. Load XGBoost model
    try:
        xgb_model = joblib.load(xgboost_model_path)
        logging.debug("XGBoost model loaded successfully")
    except Exception as e:
        logging.warning(f"Error loading XGBoost model: {e}. Using fallback.")
        return process_pdf_fallback(df, document_title, args)
    
    # 8. Make predictions
    features = [col for col in df_for_prediction.columns]
    X = df_for_prediction[features]
    
    # Predict using XGBoost model
    predictions = xgb_model.predict(X)
    probabilities = xgb_model.predict_proba(X)
    
    # Add predictions to original DataFrame
    df['predicted_level'] = predictions
    df['confidence'] = probabilities.max(axis=1)  # Max probability as confidence
    
    # 9. Map numeric levels to readable labels
    level_mapping = {
        0: "H1",
        1: "H2", 
        2: "H3",
        3: "Other", 
        4: "Title"
    }
    
    df['predicted_label'] = df['predicted_level'].map(level_mapping)
    
    # 10. Apply OptimizedPostProcessor for better results
    processor = OptimizedPostProcessor()
    
    # Convert DataFrame to format expected by processor
    blocks_for_processing = []
    for _, row in df.iterrows():
        block = {
            'text': row['text'],
            'level': row['predicted_label'],
            'page': row.get('page', 1),
            'y0': row.get('y0', 0),
            'font_size': row.get('font_size', 12)
        }
        blocks_for_processing.append(block)
    
    # Process blocks with optimized rules
    optimized_blocks = processor.process(blocks_for_processing)
    
    # Update DataFrame with optimized results
    for i, optimized_block in enumerate(optimized_blocks):
        if i < len(df):
            df.iloc[i, df.columns.get_loc('predicted_label')] = optimized_block['level']
    
    # 11. Create final structured output
    return create_structured_output(df, document_title)

def process_pdf_fallback(df, document_title, args):
    """
    Enhanced fallback processing when XGBoost model is not available
    Uses rule-based classification with rich features from feature extraction
    """
    logging.info("Using enhanced rule-based fallback processing")
    
    # Calculate document-wide statistics for better classification
    avg_font_size = df['font_size'].mean()
    std_font_size = df['font_size'].std()
    large_font_threshold = avg_font_size + std_font_size
    medium_font_threshold = avg_font_size + (std_font_size * 0.5)
    
    # Enhanced rule-based classification
    for idx, row in df.iterrows():
        text = row['text'].strip()
        font_size = row.get('font_size', 12)
        font_size_rank = row.get('font_size_rank', 10)
        is_bold = row.get('is_bold', False)
        is_centred = row.get('is_centred', False)
        is_section_header = row.get('is_section_header', False)
        is_list_item = row.get('is_list_item', False)
        word_count = len(text.split())
        capitalization_ratio = row.get('capitalization_ratio', 0)
        stopword_ratio = row.get('stopword_ratio', 0)
        
        # Calculate heading score
        heading_score = 0
        confidence = 0.5
        
        # Font size scoring
        if font_size >= large_font_threshold:
            heading_score += 3
            confidence += 0.2
        elif font_size >= medium_font_threshold:
            heading_score += 2
            confidence += 0.15
        
        # Font size rank scoring (lower rank = larger font)
        if font_size_rank <= 2:
            heading_score += 2
            confidence += 0.15
        elif font_size_rank <= 4:
            heading_score += 1
            confidence += 0.1
        
        # Bold and centering
        if is_bold:
            heading_score += 2
            confidence += 0.1
        if is_centred:
            heading_score += 1
            confidence += 0.1
        
        # Section header detection
        if is_section_header:
            heading_score += 3
            confidence += 0.2
        
        # Word count (headings are typically shorter)
        if word_count <= 3:
            heading_score += 2
        elif word_count <= 6:
            heading_score += 1
        elif word_count > 15:
            heading_score -= 2
        
        # Capitalization patterns
        if capitalization_ratio > 0.6:  # Title case
            heading_score += 1
        elif capitalization_ratio == 1.0 and word_count <= 8:  # All caps, short
            heading_score += 2
        
        # Stopword ratio (headings typically have fewer stopwords)
        if stopword_ratio < 0.2:
            heading_score += 1
        
        # List items are generally not headings
        if is_list_item:
            heading_score -= 2
        
        # Classify based on heading score
        if heading_score >= 6:
            df.loc[idx, 'predicted_label'] = 'H1'
            confidence = min(0.9, confidence + 0.2)
        elif heading_score >= 4:
            df.loc[idx, 'predicted_label'] = 'H2'
            confidence = min(0.8, confidence + 0.15)
        elif heading_score >= 2:
            df.loc[idx, 'predicted_label'] = 'H3'
            confidence = min(0.7, confidence + 0.1)
        else:
            df.loc[idx, 'predicted_label'] = 'Other'
        
        df.loc[idx, 'confidence'] = confidence
    
    return create_structured_output(df, document_title)

def create_structured_output(df, document_title):
    """
    Create structured JSON output compatible with original format
    """
    outline = []
    
    # Filter for headings only
    heading_types = ['H1', 'H2', 'H3', 'H4']
    headings_df = df[df['predicted_label'].isin(heading_types)].copy()
    
    # Sort by page and position
    headings_df = headings_df.sort_values(['page', 'y0'])
    
    # Convert to outline format
    for _, row in headings_df.iterrows():
        outline.append({
            "level": row['predicted_label'],
            "text": row['text'].strip(),
            "page": int(row.get('page', 1))
        })
    
    return {
        "title": document_title,
        "outline": outline
    }

# --- FEATURE 8: CLI JSON VALIDATION MODE ---
def validate_json_files(output_dir):
    logging.info(f"--- Running JSON Validation in '{output_dir}' ---")
    json_files = [f for f in os.listdir(output_dir) if f.lower().endswith(".json")]
    if not json_files:
        logging.warning("No JSON files found to validate.")
        return
    
    total_errors = 0
    for fname in json_files:
        path = os.path.join(output_dir, fname)
        errors = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'title' not in data: errors.append("Missing 'title' key.")
            if 'outline' not in data: errors.append("Missing 'outline' key.")
            elif not isinstance(data['outline'], list): errors.append("'outline' is not a list.")
            else:
                for i, item in enumerate(data['outline']):
                    if not isinstance(item, dict): errors.append(f"Outline item {i} is not a dictionary.")
                    else:
                        if 'level' not in item: errors.append(f"Item {i} missing 'level'.")
                        if 'text' not in item: errors.append(f"Item {i} missing 'text'.")
                        if 'page' not in item: errors.append(f"Item {i} missing 'page'.")
        except json.JSONDecodeError:
            errors.append("Invalid JSON format.")
        except Exception as e:
            errors.append(f"An unexpected error occurred: {e}")
        
        if errors:
            logging.error(f"❌ Validation FAILED for {fname}:")
            for err in errors:
                logging.error(f"  - {err}")
            total_errors += 1
        else:
            logging.info(f"✅ Validation PASSED for {fname}")
            
    logging.info(f"--- Validation Complete. Found {total_errors} file(s) with errors. ---")

# --- SCRIPT EXECUTION ---
def main():
    parser = argparse.ArgumentParser(description="Extract titles and headings from PDF documents using XGBoost.")
    parser.add_argument("--input", dest="input_dir", default="./input", help="Input directory for PDFs.")
    parser.add_argument("--output", dest="output_dir", default="./output", help="Output directory for JSONs.")
    parser.add_argument("--model", dest="model_path", default="./xgboost_model.joblib", help="Path to XGBoost model file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose DEBUG logging.")
    parser.add_argument("--debug-visualize", action="store_true", help="Generate annotated debug images.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep duplicate heading text across pages.")
    parser.add_argument("--validate-json", action="store_true", help="Run in validation mode on the output directory.")
    args = parser.parse_args()

    if args.verbose: 
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.validate_json:
        validate_json_files(args.output_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logging.error(f"❌ No PDF files found in '{args.input_dir}' directory.")
        return

    logging.info(f"🚀 Found {len(pdf_files)} PDF(s) to process...")
    
    for fname in pdf_files:
        try:
            pdf_path = os.path.join(args.input_dir, fname)
            result = process_pdf_with_xgboost(pdf_path, args.model_path, args)
            output_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + ".json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logging.info(f"✅ Successfully extracted structure to {output_path}")
        except Exception as e:
            logging.error(f"❌ Failed on {fname}: {e}", exc_info=args.verbose)
            
    print("\n🎉 Extraction complete.")

if __name__ == "__main__":
    main()