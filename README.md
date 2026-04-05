# lin020905.github.io

A cinematic-style personal portfolio website inspired by the visual structure of [fate-strange-fake.com](https://fate-strange-fake.com/), with bilingual support (中文 / English).

## Features

- Fullscreen loading interface with animated progress bar
- Hero section with dark cinematic visual style
- Fixed top navigation with responsive mobile menu
- Multi-section layout for personal showcase
  - About
  - Projects
  - Skills
  - Gallery
  - Contact
- Scroll reveal animations
- Bilingual text switching (ZH/EN)
- Responsive design for desktop and mobile

## Project Structure

```text
.
├── index.html
├── css/
│   ├── main.css
│   └── animations.css
├── js/
│   ├── i18n.js
│   └── main.js
└── README.md
```

## Local Development

Open `index.html` directly in browser, or run a static server for better testing.

Example using Python:

```bash
python -m http.server 5500
```

Then visit: `http://localhost:5500`

## GitHub Workflow (Clone → Develop → Push)

### 1) Clone your repository

```bash
git clone https://github.com/lin020905/lin020905.github.io.git
cd lin020905.github.io
```

### 2) Add / edit files locally

Put your own content into:

- `index.html` text and sections
- `css/main.css` style details
- `js/i18n.js` bilingual dictionary

### 3) Commit and push changes

```bash
git add .
git commit -m "Build portfolio website"
git push origin main
```

## Deploy to GitHub Pages

For a repository named `lin020905.github.io`, pushing to `main` will publish to:

`https://lin020905.github.io`

If not active yet:

1. Go to GitHub repository → **Settings**
2. Open **Pages**
3. Set Source to **Deploy from a branch**
4. Select branch **main** and folder **/(root)**
5. Save and wait 1–3 minutes

## How to Customize Quickly

1. Replace hero background image URL in `css/main.css` (`.hero` block)
2. Replace placeholder project cards in `index.html`
3. Update contact links in `#contact`
4. Add your own translations in `js/i18n.js`

---

If you need, I can also help you add:

- custom loading logo animation
- project detail modal pages
- image/video carousel
- automatic dark/light language preference detection
