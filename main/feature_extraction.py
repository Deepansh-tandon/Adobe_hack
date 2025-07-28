import fitz
from collections import Counter
import numpy as np
import json
import pandas as pd
import re

def merge_close_spans(spans, distance_threshold=2.0):
    """
    Merge spans that are very close together to fix fragmented text
    """
    if not spans:
        return []
    
    merged = []
    current_group = [spans[0]]
    
    for i in range(1, len(spans)):
        prev_span = current_group[-1]
        curr_span = spans[i]
        
        # Calculate horizontal distance between spans
        distance = curr_span["bbox"][0] - prev_span["bbox"][2]
        
        # Check if spans are close enough and have similar formatting
        if (distance <= distance_threshold and 
            abs(curr_span.get("size", 12) - prev_span.get("size", 12)) <= 1 and
            curr_span.get("font", "") == prev_span.get("font", "")):
            current_group.append(curr_span)
        else:
            # Process current group and start new one
            if current_group:
                merged_span = create_merged_span(current_group)
                if merged_span:
                    merged.append(merged_span)
            current_group = [curr_span]
    
    # Don't forget the last group
    if current_group:
        merged_span = create_merged_span(current_group)
        if merged_span:
            merged.append(merged_span)
    
    return merged

def create_merged_span(span_group):
    """
    Create a single span from a group of spans
    """
    if not span_group:
        return None
    
    # Merge text with spaces
    merged_text = " ".join(span["text"] for span in span_group if span["text"].strip())
    
    if not merged_text.strip():
        return None
    
    # Use properties from the first span
    first_span = span_group[0]
    
    # Calculate merged bounding box
    x0 = min(span["bbox"][0] for span in span_group)
    y0 = min(span["bbox"][1] for span in span_group)
    x1 = max(span["bbox"][2] for span in span_group)
    y1 = max(span["bbox"][3] for span in span_group)
    
    return {
        "text": merged_text,
        "font": first_span.get("font", "unknown"),
        "size": first_span.get("size", 12.0),
        "flags": first_span.get("flags", 0),
        "bbox": [x0, y0, x1, y1]
    }

def extract_text_spans(pdf_path):
    spans = []
    font_sizes = []
    doc = fitz.open(pdf_path)

    if not (0 < doc.page_count <= 200):
        raise ValueError("PDF must be between 1 and 200 pages.")

    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height
        page_width = page.rect.width

        # Use dict mode to get detailed font information
        page_dict = page.get_text("dict")
        
        page_spans = []

        for block in page_dict["blocks"]:
            if "lines" not in block:  # Skip image blocks
                continue
                
            for line in block["lines"]:
                # Collect all spans in a line to potentially merge them
                line_spans = []
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    line_spans.append(span)
                
                # Merge spans that are very close together (likely fragmented text)
                if line_spans:
                    merged_spans = merge_close_spans(line_spans)
                    
                    for span in merged_spans:
                        text = span["text"].strip()
                        if not text or len(text) <= 1:
                            continue

                        # Extract font information
                        font_name = span.get("font", "unknown")
                        font_size = span.get("size", 12.0)
                        font_flags = span.get("flags", 0)
                        
                        # Check if bold (flag 16 = bold)
                        is_bold = bool(font_flags & 2**4)
                        
                        # Get bounding box
                        bbox = span["bbox"]
                        x0, y0, x1, y1 = bbox
                        
                        width = x1 - x0
                        center_x_norm = (x0 + width / 2) / page_width
                        word_count = len(text.split())
                        token_density = word_count / width if width > 0 else 0
                        y_norm = y0 / page_height

                        span_dict = {
                            "text": text,
                            "font": font_name,
                            "font_size": font_size,
                            "size": font_size,
                            "page": page_num,
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            "width": width,
                            "center_x_norm": center_x_norm,
                            "token_density": token_density,
                            "is_centred": abs(center_x_norm - 0.5) <= 0.05,
                            "is_bold": is_bold,
                            "y_norm": y_norm
                        }

                        page_spans.append(span_dict)
                        font_sizes.append(font_size)

        # Sort page spans by vertical position
        page_spans.sort(key=lambda x: x["y0"])
        for i, s in enumerate(page_spans):
            if i > 0:
                s["spacing_above"] = s["y0"] - page_spans[i - 1]["y1"]
            else:
                s["spacing_above"] = None
            if i < len(page_spans) - 1:
                s["spacing_below"] = page_spans[i + 1]["y0"] - s["y1"]
            else:
                s["spacing_below"] = None

        spans.extend(page_spans)

    # Font size ranking
    unique_sizes = sorted(set(round(sz, 2) for sz in font_sizes), reverse=True)
    size_rank_map = {sz: rank + 1 for rank, sz in enumerate(unique_sizes)}

    for span in spans:
        span["font_size_rank"] = size_rank_map.get(round(span["font_size"], 2), -1)

    doc.close()
    return spans

def is_natural_break(current_span, next_span, avg_line_height):
    """
    Determine if there should be a natural break between spans
    """
    # Large vertical gap (more than 1.5x average line height)
    vertical_gap = next_span["y0"] - current_span["y1"]
    if vertical_gap > avg_line_height * 1.5:
        return True
    
    # Different font sizes (significant difference)
    if abs(current_span["font_size"] - next_span["font_size"]) > 2:
        return True
    
    # Different formatting (bold vs non-bold)
    if current_span["is_bold"] != next_span["is_bold"]:
        return True
    
    # Text patterns that indicate new sections
    current_text = current_span["text"].strip()
    next_text = next_span["text"].strip()
    
    # Current line ends with period, colon, or exclamation and next starts with capital
    if (re.search(r'[.!:]$', current_text) and 
        re.match(r'^[A-Z]', next_text)):
        return True
    
    # Next line starts with bullet point, number, or common section headers
    if re.match(r'^(\d+\.|\d+\)|\*|•|-|[A-Z][a-z]*:)', next_text):
        return True
    
    # Current line is very short (likely a heading or label)
    if len(current_text.split()) <= 3 and current_text.endswith(':'):
        return True
    
    return False

def detect_list_items(spans):
    """
    Detect list items and similar structured content
    """
    list_patterns = [
        r'^\d+\.',  # Numbered lists: 1. 2. 3.
        r'^\d+\)',  # Numbered lists: 1) 2) 3)
        r'^[a-z]\.',  # Lettered lists: a. b. c.
        r'^[a-z]\)',  # Lettered lists: a) b) c)
        r'^[•\*\-]',  # Bullet points
        r'^\d+\s+(cup|tablespoon|teaspoon|pound|ounce)',  # Recipe ingredients
        r'^(Ingredients?|Instructions?|Directions?|Steps?):?$',  # Section headers
    ]
    
    for span in spans:
        text = span["text"].strip()
        span["is_list_item"] = any(re.match(pattern, text, re.IGNORECASE) for pattern in list_patterns)
        span["is_section_header"] = re.match(r'^(Ingredients?|Instructions?|Directions?|Steps?):?$', text, re.IGNORECASE) is not None
    
    return spans

def group_spans_into_blocks_moderate(spans):
    """
    Moderate grouping approach that preserves structure while reducing fragmentation
    """
    if not spans:
        return []
    
    # Add list item detection
    spans = detect_list_items(spans)
    
    # Calculate average line height for the document
    line_heights = []
    for span in spans:
        line_heights.append(span["y1"] - span["y0"])
    avg_line_height = np.mean(line_heights) if line_heights else 12
    
    grouped_blocks = []
    spans_by_page = {}
    
    # Group spans by page
    for span in spans:
        page = span["page"]
        if page not in spans_by_page:
            spans_by_page[page] = []
        spans_by_page[page].append(span)
    
    for page_num, page_spans in spans_by_page.items():
        # Sort by vertical position, then horizontal
        page_spans.sort(key=lambda x: (x["y0"], x["x0"]))
        
        current_block = []
        
        for i, span in enumerate(page_spans):
            if not current_block:
                current_block.append(span)
                continue
            
            last_span = current_block[-1]
            
            # Check if we should start a new block
            should_break = False
            
            # Always break on section headers
            if span.get("is_section_header", False):
                should_break = True
            
            # Break if previous was a section header
            elif last_span.get("is_section_header", False):
                should_break = True
            
            # Break on natural breaks
            elif is_natural_break(last_span, span, avg_line_height):
                should_break = True
            
            # Break if horizontal alignment is very different (different columns)
            elif abs(span["x0"] - last_span["x0"]) > 50:
                should_break = True
            
            # Special case: keep list items separate but allow grouping of continuation text
            elif (last_span.get("is_list_item", False) and 
                  span.get("is_list_item", False)):
                should_break = True
            
            if should_break:
                # Finalize current block
                if current_block:
                    grouped_blocks.append(current_block)
                current_block = [span]
            else:
                # Add to current block
                current_block.append(span)
        
        # Don't forget the last block
        if current_block:
            grouped_blocks.append(current_block)
    
    # Convert grouped spans to block dictionaries
    text_blocks = []
    for block in grouped_blocks:
        if not block:
            continue
            
        # Combine text with proper spacing
        texts = []
        for span in block:
            text = span["text"].strip()
            if text:
                texts.append(text)
        
        if not texts:
            continue
            
        combined_text = " ".join(texts)
        
        # Get block-level properties
        font_sizes = [s["font_size"] for s in block]
        widths = [s["width"] for s in block]
        token_densities = [s["token_density"] for s in block]
        x0s = [s["x0"] for s in block]
        x1s = [s["x1"] for s in block]
        y0s = [s["y0"] for s in block]
        y1s = [s["y1"] for s in block]
        center_x_norms = [s["center_x_norm"] for s in block]
        y_norms = [s["y_norm"] for s in block]
        
        # Get the most common font name in the block
        font_names = [s["font"] for s in block]
        most_common_font = Counter(font_names).most_common(1)[0][0] if font_names else "unknown"
        
        block_dict = {
            "text": combined_text,
            "page": block[0]["page"],
            "font": most_common_font,
            "font_size": np.mean(font_sizes),
            "font_size_rank": block[0].get("font_size_rank", -1),
            "is_bold": any(s["is_bold"] for s in block),
            "is_centred": all(s["is_centred"] for s in block),
            "width": np.mean(widths),
            "x0": min(x0s),
            "x1": max(x1s),
            "y0": min(y0s),
            "y1": max(y1s),
            "center_x_norm": np.mean(center_x_norms),
            "token_density": np.mean(token_densities),
            "y_norm": np.mean(y_norms),
            "num_spans": len(block),
            "spacing_above": block[0].get("spacing_above"),
            "spacing_below": block[-1].get("spacing_below"),
            "is_list_item": any(s.get("is_list_item", False) for s in block),
            "is_section_header": any(s.get("is_section_header", False) for s in block),
        }
        
        text_blocks.append(block_dict)
    
    return text_blocks

def clean_fragmented_text(text):
    """
    Clean fragmented and repeated text from PDF extraction
    """
    if not text:
        return text
    
    # Remove excessive whitespace first
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle fragmented words like "r equest" -> "request"
    # Look for single characters followed by spaces that might be word fragments
    text = re.sub(r'\b([a-z])\s+([a-z]{2,})', r'\1\2', text, flags=re.IGNORECASE)
    
    # Handle cases like "Pr oposal" -> "Proposal"
    text = re.sub(r'\b([A-Z][a-z]?)\s+([a-z]{2,})', r'\1\2', text)
    
    # Remove repeated word patterns like "RFP: R RFP: R"
    words = text.split()
    cleaned_words = []
    
    i = 0
    while i < len(words):
        word = words[i]
        
        # Skip if this word and the next few words form a repetitive pattern
        if i < len(words) - 1:
            # Check for immediate repetition
            if word == words[i + 1]:
                # Skip repeated words
                while i < len(words) - 1 and words[i] == words[i + 1]:
                    i += 1
                cleaned_words.append(word)
                i += 1
                continue
        
        # Check for fragmented repetitions like "quest f quest f"
        if i < len(words) - 3:
            pattern_length = 2
            if (words[i:i+pattern_length] == words[i+pattern_length:i+2*pattern_length]):
                # Found a repeating pattern, keep only one instance
                cleaned_words.extend(words[i:i+pattern_length])
                i += 2 * pattern_length
                continue
        
        cleaned_words.append(word)
        i += 1
    
    # Join and do final cleanup
    result = ' '.join(cleaned_words)
    
    # Remove remaining single character artifacts between words
    result = re.sub(r'\b[a-z]\b(?=\s+[A-Z])', '', result, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def clean_repeated_words(text):
    """
    Legacy function - now calls the improved cleaning function
    """
    return clean_fragmented_text(text)

# Main execution
def main(pdf_path):
    spans = extract_text_spans(pdf_path)

    # Apply text cleaning to all span texts after extraction
    for span in spans:
        span["text"] = clean_fragmented_text(span["text"])

    # Use the moderate grouping approach
    text_blocks = group_spans_into_blocks_moderate(spans)
    return text_blocks