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

Prism is an open-source tool that converts complex academic papers (PDFs) into engaging, easy-to-understand manga-style comics. Powered by Nano Banana Pro (Gemini), it makes learning fun and accessible for everyone.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Next.js](https://img.shields.io/badge/next.js-14+-black.svg)

---

## Features

- **PDF to Manga Conversion** - Upload any academic paper and get a beautifully illustrated manga
- **Multiple Art Styles** - Choose from 3 unique manga themes:
  - Chibikawa - Original Cute, simple characters with soft pastel colors
  - Chiikawa - Cute, simple characters with soft pastel colors
  - Studio Ghibli - Dreamy watercolor atmosphere
- **AI-Powered Storyboarding** - Intelligent breakdown of complex concepts into visual panels
- **Multi-Language Support** - Generate manga in English, Chinese, or Japanese
- **Real-time Generation Progress** - Watch your manga come to life panel by panel

### Example Output

<p align="center">
  <img src="docs/demo5.png" width="100%" alt="Full Manga Example"/>
</p>

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Upload PDF │ ──▶ │ Parse Text  │ ──▶ │ Storyboard  │ ──▶ │ Generate    │
│             │     │ & Structure │     │ Generation  │     │ Manga Panels│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Upload** - Drop your PDF academic paper
2. **Parse** - AI extracts and understands the content
3. **Storyboard** - Creates a visual narrative with panels and dialogue
4. **Generate** - Nano Banana Pro renders each manga panel

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenRouter API key (for Nano Banana Pro access)

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

Set your OpenRouter API key as an environment variable:

```bash
# Windows
set OPENROUTER_API_KEY=your-openrouter-api-key

# Linux/macOS
export OPENROUTER_API_KEY=your-openrouter-api-key
```

The config file (`config/api_config.yaml`) uses `${OPENROUTER_API_KEY}` to read from environment.

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
│   │   ├── nano_banana.py # Google Gemini direct API
│   │   └── openrouter.py  # OpenRouter API wrapper
│   ├── services/
│   │   ├── pdf_parser.py      # PDF text extraction
│   │   ├── storyboarder.py    # Storyboard generation
│   │   └── manga_generator.py # Image generation
│   ├── routes/            # API endpoints
│   └── main.py            # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js app router
│   │   ├── components/    # React components
│   │   └── store/         # Zustand state management
│   └── package.json
├── config/
│   └── api_config.yaml    # API configuration
└── README.md
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/config` | Get current configuration |
| `POST` | `/api/config/reload` | Reload configuration |
| `POST` | `/api/generate/storyboard` | Generate storyboard from text |
| `POST` | `/api/generate/manga` | Generate manga from storyboard |
| `POST` | `/api/manga/from-pdf` | Full pipeline: PDF to manga |

### Example: Generate Manga from PDF

```bash
curl -X POST http://localhost:8000/api/manga/from-pdf \
  -F "file=@paper.pdf" \
  -F "theme=chiikawa" \
  -F "language=en-US"
```

## Configuration Options

### Manga Settings

| Option | Default | Description |
|--------|---------|-------------|
| `default_style` | `full_color_manga` | Art style |
| `panels_per_page` | `4` | Number of panels per page |
| `render_text_in_image` | `true` | Include dialogue in images |
| `default_theme` | `chiikawa` | Default manga theme |

### Output Settings

| Option | Default | Description |
|--------|---------|-------------|
| `image_format` | `png` | Output image format |
| `image_quality` | `95` | JPEG/PNG quality |
| `max_width` | `1024` | Maximum image width |
| `max_height` | `1536` | Maximum image height |

## Supported Models

Prism uses **Nano Banana Pro** (Gemini) for image generation:

| Provider | Model | Description |
|----------|-------|-------------|
| OpenRouter | `google/gemini-3-pro-image-preview` | Recommended - Nano Banana Pro via OpenRouter |
| Google | `gemini-2.0-flash-exp-image-generation` | Direct Google API (requires API key) |

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

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

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
- [ ] Collaborative editing
- [ ] Fine-tuned style models
- [ ] Mobile app

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Nano Banana Pro](https://openrouter.ai) - AI image generation
- [OpenRouter](https://openrouter.ai) - API gateway
- [Chiikawa](https://twitter.com/ngnchiikawa) - Inspiration for default art style

---

<p align="center">
  Made with &#10084; by the Prism Team
</p>

<p align="center">
  <a href="https://github.com/raucvr/Prism/issues">Report Bug</a>
  ·
  <a href="https://github.com/raucvr/Prism/issues">Request Feature</a>
</p>
