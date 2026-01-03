# Prism

**Transform academic papers into adorable manga with AI**

<p align="center">
  <img src="docs/demo1.png" width="45%" alt="Demo Panel 1"/>
  <img src="docs/demo2.png" width="45%" alt="Demo Panel 2"/>
</p>

<p align="center">
  <img src="docs/demo3.png" width="45%" alt="Demo Panel 3"/>
  <img src="docs/demo4.png" width="45%" alt="Demo Panel 4"/>
</p>

Prism is an open-source tool that converts complex academic papers (PDFs) into engaging, easy-to-understand manga-style comics. Powered by Gemini's image generation capabilities, it makes learning fun and accessible for everyone.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Next.js](https://img.shields.io/badge/next.js-14+-black.svg)

---

## Features

- **PDF to Manga Conversion** - Upload any academic paper and get a beautifully illustrated manga
- **Multiple Art Styles** - Choose from 3 unique manga themes:
  - **Chibikawa** - Original cute characters (kumo, nezu, papi) with consistent design across all panels
  - **Chiikawa** - Cute, simple characters with soft pastel colors (Nagano style)
  - **Studio Ghibli** - Dreamy watercolor atmosphere (Spirited Away style)
- **Character Consistency** - Reference images ensure characters look the same throughout the entire manga
- **AI-Powered Storyboarding** - Intelligent breakdown of complex concepts into visual panels with proper story ordering
- **Multi-Language Support** - Generate manga in English, Chinese (中文), or Japanese (日本語)
- **CJK Text Optimization** - Dynamic batch sizing for clear, readable Chinese/Japanese text
- **Real-time Generation Progress** - Watch your manga come to life panel by panel

### Example Output

<p align="center">
  <img src="docs/demo5.png" width="100%" alt="Full Manga Example"/>
</p>

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Upload PDF │ ──▶ │   Analyze   │ ──▶ │ Storyboard  │ ──▶ │  Generate   │
│             │     │  (English)  │     │  + Translate│     │ Manga Panels│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Upload** - Drop your PDF academic paper
2. **Analyze** - AI extracts and understands the content in English (for accuracy)
3. **Storyboard** - Creates a visual narrative, then translates dialogue to target language
4. **Generate** - Gemini renders each manga panel with consistent character designs

### Three-Step Translation Pipeline

For optimal quality, Prism uses a three-step process:
1. **English Analysis** - Technical content is analyzed in English for accuracy
2. **English Storyboard** - Panels and dialogue are created in English first
3. **Translation** - Final dialogue is translated to the target language (zh-CN, ja-JP)

This ensures technical terms are understood correctly before translation.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenRouter API key (for Gemini access)

### Installation

```bash
# Clone the repository
git clone https://github.com/raucvr/Prism.git
cd Prism

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### Configuration

1. Copy the example config file:
```bash
cp config/api_config.yaml.example config/api_config.yaml
```

2. Set your OpenRouter API key in `config/api_config.yaml`:
```yaml
api_key: "your-openrouter-api-key"
```

Or use environment variable:
```bash
# Windows
set OPENROUTER_API_KEY=your-openrouter-api-key

# Linux/macOS
export OPENROUTER_API_KEY=your-openrouter-api-key
```

### Running

```bash
# Terminal 1 - Start backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Start frontend
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
prism/
├── backend/
│   ├── engines/           # AI engine implementations
│   │   ├── base.py        # Abstract base classes
│   │   └── openrouter.py  # OpenRouter API wrapper
│   ├── services/
│   │   ├── pdf_parser.py      # PDF text extraction
│   │   ├── storyboarder.py    # Storyboard generation
│   │   └── manga_generator.py # Image generation with character consistency
│   ├── routes/            # API endpoints
│   └── main.py            # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js app router
│   │   ├── components/    # React components
│   │   └── store/         # Zustand state management
│   └── package.json
├── config/
│   ├── api_config.yaml.example  # Config template
│   └── character_images/        # Reference images for Chibikawa theme
│       ├── kumo.jpeg
│       ├── nezu.jpeg
│       └── papi.jpeg
└── README.md
```

## Technical Details

### Character Consistency

For the Chibikawa theme, Prism ensures character consistency by:
1. **Reference Images** - Loading character design images for each API call
2. **Explicit Mapping** - Telling the model exactly which image corresponds to which character
3. **Low Temperature** - Using temperature=0.3 to reduce randomness
4. **Negative Prompts** - Explicitly forbidding character design deviations

### Dynamic Batch Sizing

For CJK languages (Chinese, Japanese), Prism dynamically adjusts batch size:
- Long dialogue (>200 chars): 1 panel per batch
- Medium dialogue (>100 chars): 2 panels per batch
- Short dialogue: 4 panels per batch (2x2 grid)

This ensures text remains readable even with complex characters.

### Story Ordering

All panels are sorted by `panel_number` after generation to ensure the story flows correctly, even if the AI generates them out of order.

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/config` | Get current configuration |
| `POST` | `/api/manga/from-pdf` | Full pipeline: PDF to manga |
| `POST` | `/api/manga/storyboard` | Generate storyboard from PDF |
| `GET` | `/api/manga/progress` | Get generation progress |

### Example: Generate Manga from PDF

```bash
curl -X POST http://localhost:8000/api/manga/from-pdf \
  -F "file=@paper.pdf" \
  -F "theme=chibikawa" \
  -F "language=zh-CN"
```

## Configuration Options

### Manga Settings

| Option | Default | Description |
|--------|---------|-------------|
| `default_style` | `full_color_manga` | Art style |
| `default_theme` | `chibikawa` | Default manga theme |
| `temperature` | `0.3` | Generation randomness (lower = more consistent) |

### Output Settings

| Option | Default | Description |
|--------|---------|-------------|
| `image_format` | `png` | Output image format |
| `image_quality` | `95` | PNG quality |
| `max_width` | `1024` | Maximum image width |
| `max_height` | `1536` | Maximum image height |

## Supported Models

| Provider | Model | Description |
|----------|-------|-------------|
| OpenRouter | `google/gemini-2.0-flash-exp-image-generation` | Recommended - Gemini image generation |

## Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **httpx** - Async HTTP client
- **PyMuPDF** - PDF parsing
- **Pillow** - Image processing

### Frontend
- **Next.js 14** - React framework with App Router
- **TailwindCSS** - Utility-first CSS
- **Zustand** - State management
- **Framer Motion** - Animations
- **react-dropzone** - File upload

## Contributing

We welcome contributions!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [ ] Batch processing for multiple papers
- [ ] Custom character upload
- [ ] Panel layout customization
- [ ] Export to PDF/EPUB
- [ ] More art styles
- [ ] Mobile app

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenRouter](https://openrouter.ai) - API gateway for AI models
- [Chiikawa](https://twitter.com/ngnchiikawa) - Inspiration for cute art style

---

<p align="center">
  Made with &#10084; by the Prism Team
</p>

<p align="center">
  <a href="https://github.com/raucvr/Prism/issues">Report Bug</a>
  ·
  <a href="https://github.com/raucvr/Prism/issues">Request Feature</a>
</p>
