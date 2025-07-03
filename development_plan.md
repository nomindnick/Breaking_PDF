# PDF Splitter Application - Development Plan

## Executive Summary

This document outlines the development plan for a PDF splitter application that intelligently identifies and separates individual documents within large, multi-document PDF files. The application serves dual purposes: immediate utility as a standalone tool for legal professionals, and as a foundational component for a future RAG-based construction claims assistant.

## Project Objectives

### Primary Objective (Phase 1)
Build a robust, user-friendly PDF splitter application that:
- Processes both searchable and image-based PDFs from CPRA requests and similar sources
- Uses multi-signal detection (LLM, visual, heuristic) to identify document boundaries
- Provides a web interface for manual review and adjustment of split points
- Processes documents efficiently (< 5 seconds per page)
- Can be easily distributed to colleagues or deployed as a web application

### Secondary Objective (Future Phase)
Create modular, reusable components that will integrate into a larger RAG-based system for:
- Construction claims analysis and management
- Document ingestion pipeline for vector embedding
- Intelligent document categorization and retrieval
- Agentic AI assistance for claims processing

## Business Case

### Immediate Value
- **Time Savings**: Automates manual document separation that currently takes hours
- **Consistency**: Reduces human error in document boundary identification
- **Scalability**: Enables processing of larger document sets efficiently
- **Firm-wide Utility**: Benefits attorneys across practice areas dealing with multi-document PDFs

### Strategic Value
- **Foundation for AI Integration**: Modular architecture supports future RAG system development
- **Competitive Advantage**: Advanced document processing capabilities
- **Knowledge Management**: Better document organization improves case preparation and discovery
- **Technology Leadership**: Positions firm as innovative in legal tech adoption

## Technical Architecture

### System Overview
```
User Upload → Preprocessing → Multi-Signal Detection → Manual Review → Document Splitting
     ↓              ↓                    ↓                 ↓              ↓
   Web UI    →  OCR Engine  →    LLM + Visual +    →   UI Preview  →   Final PDFs
                               Heuristic Signals
```

### Modular Architecture

#### Directory Structure
```
pdf_splitter/
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logging.py             # Centralized logging
│   └── exceptions.py          # Custom exception classes
├── preprocessing/
│   ├── __init__.py
│   ├── pdf_handler.py         # PDF loading, validation, metadata extraction
│   ├── ocr_processor.py       # Multi-engine OCR processing
│   ├── text_extractor.py      # Text extraction from searchable PDFs
│   └── tests/
│       ├── __init__.py
│       ├── test_pdf_handler.py
│       ├── test_ocr_processor.py
│       └── test_text_extractor.py
├── detection/
│   ├── __init__.py
│   ├── base_detector.py       # Abstract base class for detectors
│   ├── llm_detector.py        # LLM-based boundary detection
│   ├── visual_detector.py     # Visual pattern analysis
│   ├── heuristic_detector.py  # Rule-based detection
│   ├── signal_combiner.py     # Multi-signal combination and scoring
│   └── tests/
│       ├── __init__.py
│       ├── test_llm_detector.py
│       ├── test_visual_detector.py
│       ├── test_heuristic_detector.py
│       └── test_signal_combiner.py
├── splitting/
│   ├── __init__.py
│   ├── pdf_splitter.py        # PDF document splitting logic
│   ├── file_manager.py        # Output file naming and organization
│   └── tests/
│       ├── __init__.py
│       ├── test_pdf_splitter.py
│       └── test_file_manager.py
├── api/
│   ├── __init__.py
│   ├── routes.py              # FastAPI route definitions
│   ├── models.py              # Pydantic models for requests/responses
│   ├── dependencies.py        # Dependency injection for FastAPI
│   ├── middleware.py          # Custom middleware (CORS, logging, etc.)
│   └── tests/
│       ├── __init__.py
│       ├── test_routes.py
│       └── test_models.py
├── frontend/
│   ├── templates/
│   │   ├── base.html          # Base template
│   │   ├── upload.html        # File upload interface
│   │   ├── processing.html    # Progress tracking page
│   │   └── review.html        # Boundary review and adjustment
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css
│   │   ├── js/
│   │   │   ├── upload.js
│   │   │   ├── progress.js
│   │   │   └── review.js
│   │   └── images/
├── test_files/
│   ├── README.md              # Test data documentation
│   ├── sample_32_page.pdf     # Non-OCR test PDF
│   ├── sample_32_page_ocr.pdf # OCR'd test PDF
│   ├── ground_truth.json      # Expected boundaries and document types
│   └── additional_samples/    # Additional test cases
├── scripts/
│   ├── benchmark_ocr.py       # OCR engine performance testing
│   ├── validate_setup.py      # Environment validation
│   └── deploy.py              # Deployment automation
├── docs/
│   ├── api_documentation.md   # Auto-generated API docs
│   ├── user_guide.md          # End-user documentation
│   └── deployment_guide.md    # IT/deployment instructions
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Local development stack
├── .gitignore                 # Git ignore rules
├── .env.example               # Environment variable template
├── pytest.ini                # Testing configuration
└── main.py                    # Application entry point
```

### Core Components Detail

#### 1. Core Module (`core/`)
**Purpose**: Shared utilities and configuration across all modules
- **config.py**: Environment-specific settings, API keys, file paths
- **logging.py**: Centralized logging configuration and utilities
- **exceptions.py**: Custom exception classes for different error types

#### 2. Preprocessing Module (`preprocessing/`)
**Purpose**: Document preparation and text extraction
- **pdf_handler.py**:
  - PDF file validation and loading
  - Metadata extraction (page count, embedded text detection)
  - Page-level processing coordination
- **ocr_processor.py**:
  - Multi-engine OCR support (PaddleOCR primary, EasyOCR/Tesseract fallback)
  - Intelligent OCR necessity detection
  - Performance optimization and caching
- **text_extractor.py**:
  - Direct text extraction from searchable PDFs
  - Quality assessment of extracted text
  - Coordinate mapping for visual analysis

#### 3. Detection Module (`detection/`)
**Purpose**: Multi-signal document boundary identification
- **base_detector.py**: Abstract interface for all detection methods
- **llm_detector.py**:
  - Local LLM integration (Ollama dev, Transformers prod)
  - Context overlap analysis (30% strategy)
  - Prompt engineering and response parsing
- **visual_detector.py**:
  - Layout analysis and formatting changes
  - Header/footer pattern detection
  - Whitespace and visual break identification
- **heuristic_detector.py**:
  - Date pattern changes
  - Sender/recipient analysis
  - Subject line and document type detection
  - Page numbering pattern analysis
- **signal_combiner.py**:
  - Weighted scoring of all detection signals
  - Confidence calculation and threshold management
  - Final boundary decision logic

#### 4. Splitting Module (`splitting/`)
**Purpose**: Final document separation and output management
- **pdf_splitter.py**:
  - Page range extraction from source PDF
  - Metadata preservation
  - Quality validation of split documents
- **file_manager.py**:
  - Automatic file naming based on content analysis
  - Manual naming interface support
  - Output organization and cleanup

#### 5. API Module (`api/`)
**Purpose**: Web service interface and business logic coordination
- **routes.py**: RESTful endpoints for all operations
- **models.py**: Request/response data validation with Pydantic
- **dependencies.py**: Shared dependencies and dependency injection
- **middleware.py**: Cross-cutting concerns (CORS, authentication, logging)

#### 6. Frontend Module (`frontend/`)
**Purpose**: User interface and user experience
- **Templates**: Jinja2 templates for server-side rendering
- **Static Assets**: CSS, JavaScript, and images
- **Progressive Enhancement**: HTMX for dynamic updates without full page reloads

### Technology Stack

#### Backend
- **Framework**: FastAPI (async support, automatic documentation)
- **PDF Processing**: PyMuPDF (fast, reliable)
- **OCR**: PaddleOCR primary, EasyOCR/Tesseract fallback
- **LLM**: Transformers library (production), Ollama (development)
- **Database**: SQLite (development), PostgreSQL (production)

#### Frontend
- **Primary**: HTML/CSS/JavaScript with HTMX
- **Alternative**: Streamlit (rapid prototyping option)

#### Infrastructure
- **Development**: Local laptop (32GB RAM, CPU-only)
- **Deployment**: Docker containers, cloud-ready architecture

## Development Strategy

### Modular Development Approach
Build and perfect one module at a time before moving to the next:

1. **Preprocessing Module** (Weeks 1-2)
   - Focus on OCR processor until "rock solid"
   - Handle both searchable and image PDFs
   - Optimize for speed and accuracy

2. **LLM Detection Module** (Weeks 3-4) - **IN PROGRESS - Experimental Phase**
   - ✅ Base architecture established (BaseDetector abstract class)
   - ✅ Experimentation framework with Ollama integration
   - ✅ Multiple prompt templates created
   - 🔬 Testing models: Llama3 (8B), Gemma3, Phi4-mini, Phi3-mini
   - 🔬 Testing strategies: context_overlap, type_first, chain_of_thought
   - 📊 Target: >95% F1 score before implementing other detectors

3. **Visual & Heuristic Detection** (Week 5)
   - Implement complementary detection methods
   - Test against diverse document types

4. **Signal Combination** (Week 6)
   - Develop weighted scoring algorithm
   - Validate against ground truth data
   - Fine-tune for optimal accuracy

5. **Integration & Frontend** (Weeks 7-8)
   - Build user interface
   - Implement progress tracking
   - User testing and refinement

6. **Testing & Deployment** (Week 9)
   - Comprehensive testing with diverse PDFs
   - Package for distribution
   - Deploy cloud version

### Quality Assurance Strategy
- **Unit Testing**: Each module has comprehensive test suite in co-located `/tests` directories
- **Integration Testing**: End-to-end testing with real PDFs
- **Performance Testing**: Validate < 5 seconds per page target
- **User Acceptance Testing**: Colleague feedback and iteration

### Version Control & Documentation
- **Git Workflow**: Feature branches, clear commit messages, tagged releases
- **Documentation**: Comprehensive README, API docs, user guides
- **Code Quality**: Type hints, docstrings, linting with black/flake8

## Performance Requirements

### Processing Speed
- **Target**: < 5 seconds per page total processing time
- **Breakdown**:
  - OCR: 1-2 seconds per page (when needed)
  - LLM Detection: 1-2 seconds per boundary check
  - Visual/Heuristic: < 0.5 seconds per page
  - Splitting: < 0.1 seconds per page

### Resource Constraints
- **Memory**: Efficient within 32GB RAM constraint
- **CPU**: Optimized for CPU-only processing
- **Storage**: Minimal temporary file usage, efficient cleanup

### Accuracy Targets
- **Boundary Detection**: > 90% accuracy vs ground truth
- **OCR Quality**: Readable text extraction from both PDF types
- **Error Handling**: Graceful degradation when components fail

## Deployment Options

### Option 1: Standalone Application
- **Target**: Individual attorney desktops
- **Technology**: PyInstaller executable
- **Benefits**: No IT involvement, full control, offline operation
- **Distribution**: Internal file sharing, simple installation

### Option 2: Firm Network Deployment
- **Target**: Shared firm infrastructure
- **Technology**: Docker container on internal servers
- **Benefits**: Centralized maintenance, shared resources
- **Access**: Web interface accessible firm-wide

### Option 3: Cloud Deployment
- **Target**: External web application
- **Technology**: AWS/Azure with container orchestration
- **Benefits**: Scalability, accessibility, professional deployment
- **Considerations**: Data security, compliance requirements

### Hybrid Approach (Recommended)
Start with standalone application for immediate use, architect for cloud deployment:
- Develop with containerization in mind
- Environment-specific configuration
- Modular deployment (can run locally or in cloud)

## Future Integration Strategy

### RAG System Architecture Preparation
The PDF splitter components are designed for seamless integration into a larger RAG system:

#### Document Ingestion Pipeline
```
PDF Splitter → Document Classification → Text Chunking → Vector Embedding → Storage
     ↓               ↓                      ↓              ↓              ↓
  Individual    Construction-Specific   Semantic Chunks  Vector DB    Retrieval
  Documents     Document Types         for RAG          (Pinecone)    System
```

#### Reusable Components
- **OCR Processor**: Direct integration for any document processing
- **LLM Interface**: Extensible for various NLP tasks
- **Visual Detection**: Useful for document classification
- **API Framework**: Foundation for RAG system endpoints

#### Data Flow Integration
- **Input**: Split documents become RAG system inputs
- **Metadata**: Document boundaries inform chunking strategy
- **Classification**: Document types guide embedding approaches
- **Quality**: OCR text quality affects RAG performance

## Risk Management

### Technical Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| OCR performance on poor scans | High | Multiple engine fallback, user review interface |
| LLM latency exceeds targets | Medium | Batch processing, async operations, model optimization |
| Memory constraints with large files | Medium | Page-by-page processing, efficient garbage collection |
| Accuracy below requirements | High | Multiple signal sources, manual review capability |

### Business Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Low colleague adoption | Medium | Focus on UX, clear value demonstration, training |
| Maintenance burden | Low | Comprehensive documentation, automated testing |
| Scope creep | Medium | Clear phase boundaries, modular architecture |
| Technology obsolescence | Low | Standard technologies, modular design |

## Success Metrics

### Technical Success
- [ ] Processes 32-page test PDF in < 160 seconds (5 sec/page)
- [ ] Achieves > 90% boundary detection accuracy vs ground truth
- [ ] Successfully handles both searchable and image-based PDFs
- [ ] Demonstrates reliable error handling and recovery

### Business Success
- [ ] Saves > 2 hours per multi-document PDF processing task
- [ ] Adopted by at least 3 colleagues within first month
- [ ] Zero data loss or corruption incidents
- [ ] Positive user feedback on interface and reliability

### Strategic Success
- [ ] Components successfully integrated into future RAG system
- [ ] Architecture supports cloud deployment without major refactoring
- [ ] Firm gains reputation for innovative legal technology use
- [ ] Foundation established for additional AI-powered legal tools

## Next Steps

### Immediate Actions (This Week)
1. Set up development environment using setup guide
2. Create basic project structure with co-located testing
3. Install and test PaddleOCR vs current OCR solution
4. Review 32-page test PDF and JSON ground truth format

### Phase 1 Kickoff (Next Week)
1. Begin OCR processor development
2. Implement PDF text detection logic
3. Create basic OCR processing pipeline
4. Establish testing framework with real PDF data

### Milestone Reviews
- **Week 2**: OCR processor complete and validated
- **Week 4**: LLM detection working with acceptable accuracy
- **Week 6**: Multi-signal combination achieving target accuracy
- **Week 8**: Complete application with functional UI
- **Week 9**: Production-ready deployment

This development plan provides a clear roadmap while maintaining flexibility for technical discoveries and requirement refinements during implementation.
