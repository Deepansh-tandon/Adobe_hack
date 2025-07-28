# PDF Heading Detection System - Technical Approach

## Overview

This document outlines the technical approach for an advanced PDF heading detection and document structure extraction system. The solution combines machine learning techniques with rule-based processing to accurately identify and classify document headings while providing robust fallback mechanisms.

## Problem Statement

The challenge is to automatically extract structured information from PDF documents, specifically:
- **Document titles** from the first page
- **Heading hierarchy** (H1, H2, H3, etc.) throughout the document
- **Page-level localization** of each heading
- **JSON output format** for downstream processing

### Key Challenges
1. **PDF complexity**: Varied layouts, fonts, and formatting styles
2. **Text fragmentation**: OCR artifacts and rendering inconsistencies
3. **Heading ambiguity**: Distinguishing headings from body text
4. **Hierarchy detection**: Proper level assignment (H1 vs H2 vs H3)
5. **Cross-document consistency**: Working across different document types

## Architecture Overview

```
Input PDFs → Feature Extraction → Classification → Post-processing → JSON Output
     ↓              ↓                    ↓              ↓             ↓
  PyMuPDF     Advanced Text        XGBoost Model    Hierarchy      Structured
  Parsing     Analysis &           (Primary) +      Validation     Document
              Typography           Rule-based       & Cleaning     Outline
              Features             (Fallback)
```

## Core Components

### 1. Feature Extraction Engine (`main/feature_extraction.py`)

#### Text Span Processing
- **PyMuPDF Integration**: Direct PDF parsing for accurate text positioning
- **Smart Text Merging**: Combines fragmented text spans using proximity analysis
- **Font Attribute Extraction**: Size, weight, family, and formatting flags

#### Advanced Text Analysis
```python
Features Extracted:
├── Typography
│   ├── Font size ranking (relative importance)
│   ├── Bold/italic detection
│   └── Font family normalization
├── Positioning
│   ├── Normalized coordinates (x0, y0, x1, y1)
│   ├── Center alignment detection
│   └── Vertical spacing analysis
├── Content Analysis
│   ├── Token density calculation
│   ├── Word count optimization
│   └── Text pattern recognition
└── Structural Elements
    ├── List item detection
    ├── Section header identification
    └── Table content filtering
```

#### Text Cleaning Pipeline
- **Fragmentation Repair**: "r equest" → "request"
- **Repetition Removal**: Eliminates OCR duplicates
- **Unicode Normalization**: Consistent character encoding
- **Whitespace Optimization**: Proper spacing and line breaks

### 2. Title Extraction System

#### Clustering-Based Approach
The system uses **DBSCAN clustering** to identify title candidates:

```python
Title Detection Pipeline:
1. Extract top 70% of first page
2. Filter valid text candidates
3. Create feature vectors [y_position, font_height]
4. Apply StandardScaler normalization
5. DBSCAN clustering (eps=0.3, min_samples=1)
6. Score clusters by:
   - Average font height (60% weight)
   - Position superiority (30% weight)
   - Text length (10% weight)
```

#### Scoring Algorithm
```python
score = (avg_height * 0.6 + 
         (1 - min_y_pos/page_height) * 0.3 + 
         text_length * 0.1)
```

### 3. Classification System

#### Primary: XGBoost Model
- **Training Features**: 15+ extracted features including typography, positioning, and NLP attributes
- **Model Size**: 292KB (well under 200MB requirement)
- **Output Classes**: Title, H1, H2, H3, Other
- **Confidence Scoring**: Probability distribution analysis

#### Fallback: Enhanced Rule-Based System
When XGBoost model is unavailable, the system uses a sophisticated scoring algorithm:

```python
Heading Score Calculation:
├── Font Analysis (40% weight)
│   ├── Size relative to document average
│   ├── Font size ranking (1-5 scale)
│   └── Bold formatting detection
├── Content Analysis (35% weight)
│   ├── Word count optimization (2-15 words ideal)
│   ├── Capitalization patterns
│   ├── Stopword ratio analysis
│   └── Section header patterns
├── Positioning (15% weight)
│   ├── Text centering detection
│   └── Vertical spacing analysis
└── Contextual Factors (10% weight)
    ├── List item penalties
    └── Punctuation analysis
```

#### Classification Thresholds
- **H1**: Score ≥ 6 (High confidence headings)
- **H2**: Score ≥ 4 (Medium confidence headings)  
- **H3**: Score ≥ 2 (Lower confidence headings)
- **Other**: Score < 2 (Body text, captions, etc.)

### 4. Post-Processing Pipeline

#### OptimizedPostProcessor
Advanced rule-based refinement system:

```python
Processing Rules:
├── Title Handling
│   ├── Single title per document
│   └── Page 1 restriction
├── Length Validation
│   ├── Maximum 15 words for headings
│   └── Minimum 2 words requirement
├── Format Analysis
│   ├── All-caps promotion (≤5 words)
│   ├── Punctuation demotion
│   └── Footer detection
├── Smart Detection
│   ├── Title case scoring
│   ├── Verb ratio analysis
│   └── Stopword optimization
└── Hierarchy Validation
    ├── Level sequence enforcement
    └── Logical progression (H1→H2→H3)
```

#### Hierarchy Validation
Ensures logical heading progression:
- Prevents level jumping (H1 → H3 becomes H1 → H2)
- Maintains document structure consistency
- Corrects classification errors through context analysis

### 5. Output Generation

#### JSON Structure
```json
{
    "title": "Document Title",
    "outline": [
        {
            "level": "H1",
            "text": "Section Heading",
            "page": 1
        }
    ]
}
```

## Natural Language Processing Features

### Simple NLP Pipeline
Designed for offline operation without heavy dependencies:

```python
NLP Features:
├── Stopword Analysis
│   ├── Custom stopword dictionary (60+ words)
│   ├── Ratio calculation
│   └── Heading likelihood scoring
├── Capitalization Patterns
│   ├── Title case detection
│   ├── All-caps identification
│   └── Mixed case analysis
├── Linguistic Features
│   ├── Verb detection (100+ common verbs)
│   ├── Punctuation counting
│   └── Sentence structure analysis
└── Content Classification
    ├── Technical terminology
    ├── Section indicators
    └── List pattern recognition
```

## Docker Containerization

### Multi-Stage Architecture
```dockerfile
Base Image: python:3.11-slim (AMD64)
├── System Dependencies
│   ├── GCC/G++ for compilation
│   └── Package build tools
├── Python Environment
│   ├── All requirements.txt packages
│   ├── XGBoost model inclusion
│   └── Feature extraction modules
├── Application Layer
│   ├── process_pdf.py (main script)
│   ├── main/ directory (feature extraction)
│   └── Configuration files
└── Runtime Configuration
    ├── Volume mounting (/app/input ↔ /app/output)
    ├── Network isolation (--network none)
    └── Offline operation capability
```

### Performance Optimizations
- **Layer Caching**: Requirements installed before code copy
- **Image Size**: Optimized dependency installation
- **Offline Capability**: All models and dependencies included
- **Platform Compatibility**: Explicit AMD64 targeting

## Algorithm Performance

### Computational Complexity
- **Feature Extraction**: O(n) where n = number of text spans
- **Title Detection**: O(k log k) where k = candidate count
- **Classification**: O(1) per text block (constant time)
- **Post-processing**: O(m) where m = number of headings

### Memory Usage
- **Base Requirements**: ~200MB for Python environment
- **Model Size**: 292KB (XGBoost model)
- **Runtime Memory**: ~50-100MB per document
- **Total Container Size**: ~800MB

### Processing Speed
- **Small PDFs** (1-5 pages): 1-3 seconds
- **Medium PDFs** (10-20 pages): 3-8 seconds  
- **Large PDFs** (50+ pages): 10-30 seconds
- **Throughput**: ~10-20 documents per minute

## Quality Assurance

### Robustness Features
1. **Graceful Degradation**: XGBoost → Rule-based → Basic extraction
2. **Error Handling**: Comprehensive exception management
3. **Input Validation**: PDF format and size verification
4. **Output Consistency**: Standardized JSON format

### Testing Scenarios
- **Document Types**: Academic papers, reports, manuals, forms
- **Layout Variations**: Single/multi-column, complex formatting
- **Language Support**: Primarily English, extensible for others
- **Quality Metrics**: Precision, recall, and F1-score tracking

## Future Enhancements

### Technical Improvements
1. **Deep Learning Integration**: Transformer-based models for better context
2. **Multi-language Support**: Extended NLP capabilities
3. **Image Processing**: Logo and diagram recognition
4. **Table Structure**: Enhanced table content extraction

### Performance Optimizations
1. **Parallel Processing**: Multi-threaded document processing
2. **Caching System**: Feature extraction result caching
3. **Model Compression**: Smaller model variants
4. **Memory Optimization**: Streaming PDF processing

### Advanced Features
1. **Custom Training**: Domain-specific model fine-tuning
2. **Interactive Validation**: User feedback integration
3. **Batch Processing**: Large-scale document processing
4. **API Integration**: REST API for service deployment

## Dependencies

### Core Libraries
```python
├── PyMuPDF (fitz): PDF parsing and text extraction
├── pandas: Data manipulation and analysis
├── scikit-learn: Machine learning utilities
├── numpy: Numerical computations
├── xgboost: Gradient boosting classification
├── joblib: Model serialization
└── pillow: Image processing support
```

### System Requirements
- **Python**: 3.11+ for optimal compatibility
- **Memory**: Minimum 1GB RAM recommended
- **Storage**: ~800MB for full container
- **CPU**: AMD64/x86_64 architecture

## Deployment Considerations

### Production Readiness
- **Container Isolation**: Full network isolation capability
- **Resource Limits**: Configurable CPU and memory constraints
- **Logging**: Comprehensive operation logging
- **Monitoring**: Performance metrics tracking

### Scalability
- **Horizontal Scaling**: Multiple container instances
- **Load Balancing**: Document distribution strategies
- **Queue Management**: Batch processing queues
- **Result Aggregation**: Combined output handling

---

*This approach document provides a comprehensive overview of the PDF heading detection system architecture, implementation details, and technical considerations for production deployment.* 