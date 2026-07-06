---
name: ChismesitoGPT v2
colors:
  surface: '#10141a'
  surface-dim: '#10141a'
  surface-bright: '#353940'
  surface-container-lowest: '#0a0e14'
  surface-container-low: '#181c22'
  surface-container: '#1c2026'
  surface-container-high: '#262a31'
  surface-container-highest: '#31353c'
  on-surface: '#dfe2eb'
  on-surface-variant: '#bcc9c6'
  inverse-surface: '#dfe2eb'
  inverse-on-surface: '#2d3137'
  outline: '#879391'
  outline-variant: '#3d4947'
  surface-tint: '#6bd8cb'
  primary: '#6bd8cb'
  on-primary: '#003732'
  primary-container: '#29a195'
  on-primary-container: '#00302b'
  inverse-primary: '#006a61'
  secondary: '#6cd3f7'
  on-secondary: '#003543'
  secondary-container: '#269dbe'
  on-secondary-container: '#002e3b'
  tertiary: '#4ddcc6'
  on-tertiary: '#003730'
  tertiary-container: '#00a391'
  on-tertiary-container: '#00302a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#89f5e7'
  primary-fixed-dim: '#6bd8cb'
  on-primary-fixed: '#00201d'
  on-primary-fixed-variant: '#005049'
  secondary-fixed: '#b7eaff'
  secondary-fixed-dim: '#6cd3f7'
  on-secondary-fixed: '#001f28'
  on-secondary-fixed-variant: '#004e61'
  tertiary-fixed: '#6ef9e2'
  tertiary-fixed-dim: '#4ddcc6'
  on-tertiary-fixed: '#00201b'
  on-tertiary-fixed-variant: '#005047'
  background: '#10141a'
  on-background: '#dfe2eb'
  surface-variant: '#31353c'
  bg-base: '#0d1117'
  bg-surface: rgba(255,255,255,0.04)
  bg-surface-hover: rgba(255,255,255,0.08)
  border-subtle: rgba(255,255,255,0.10)
  border-active: rgba(13,148,136,0.5)
  text-primary: '#e6edf3'
  text-secondary: rgba(255,255,255,0.55)
  text-muted: rgba(255,255,255,0.35)
  gemini-blue: '#4285F4'
  deepseek-blue: '#5B8DEF'
  claude-orange: '#D97757'
  openai-green: '#74AA9C'
  sentiment-positive: '#10b981'
  sentiment-negative: '#f87171'
  sentiment-neutral: '#94a3b8'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 26px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.05em
  headline-xl-mobile:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '700'
    lineHeight: 28px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  container-margin: 2rem
  gutter-md: 1rem
  gutter-sm: 0.625rem
  stack-gap: 1.5rem
  section-padding: 1.5rem
---

## Brand & Style

The design system for the product reflects a "Refined GitHub Dark" aesthetic—combining the technical authority of an AI analysis tool with the vibrant, social energy of "chisme" (gossip/social trends). 

The style is **Corporate Modern with Glassmorphism**. It prioritizes a dark, focused environment using deep charcoal and obsidian tones, punctuated by high-energy teal and cyan gradients. The interface feels like a sophisticated digital laboratory: precise, analytical, and futuristic, yet accessible through soft rounded corners and translucent material layers.

**Key Visual Principles:**
- **Analytical Depth:** Use of dark backgrounds and subtle borders to keep the focus on data visualizations and AI insights.
- **Social Vibrancy:** Bright teal accents and active states ensure the "social" nature of the data feels alive and engaging.
- **Glassmorphism:** Strategic use of semi-transparent surfaces to create visual hierarchy without cluttering the UI.

## Colors

The palette is rooted in the **GitHub Dark** theme, utilizing `#0d1117` as the foundation. This allows the primary teal-to-cyan gradient to stand out as the functional highlight.

**Color Usage:**
- **Primary/Secondary:** Used exclusively for the main action gradient (Teal #0d9488 to Cyan #0891b2).
- **Tertiary:** A bright "glow" teal (#5eead4) reserved for active chips, links, and success states.
- **Surface Colors:** Derived from white with low opacity (4% to 8%) to create a frosted glass effect over the dark background.
- **LLM Providers:** Specific brand colors are used for badges to help users instantly identify the active intelligence model.
- **Sentiment:** Standardized Green/Red/Gray for immediate emotional data parsing.

## Typography

The system uses **Inter** exclusively to maintain a clean, highly legible, and systematic feel. The type hierarchy is intentionally compact to accommodate data-heavy dashboards.

**Implementation Rules:**
- **Headlines:** Use tighter letter spacing and heavier weights (600-700). `headline-xl` is reserved for Login and Hero titles.
- **Labels:** Small caps or bold weights at 11px/12px are used for metadata, badges, and secondary UI labels.
- **Body:** 13px is the standard for cards and chips; 14px is used for inputs to ensure readability during user interaction.
- **Responsive:** Large headers scale down on mobile devices to prevent excessive line wrapping.

## Layout & Spacing

The design system employs a **Fluid Grid** model with a focus on density for data analysis.

**Layout Structure:**
- **Grid:** A standard 12-column system for desktop, collapsing to 1 column for mobile.
- **Rhythm:** An 8px-based spacing system. Common gaps between components are 10px (gutter-sm) for tight grids like Social Chips or 16px (gutter-md) for dashboard widgets.
- **Breakpoints:**
  - **Mobile (< 768px):** Full-width containers, 1rem margins.
  - **Tablet (768px - 1024px):** 2-column grids for cards, reduced padding.
  - **Desktop (> 1024px):** Multi-column dashboard layouts with fixed sidebars if necessary.
- **Chat Layout:** The RAG chat is always visible or anchored to the bottom/side, maintaining a persistent height of 420px for the message container.

## Elevation & Depth

Hierarchy is established through **Glassmorphism** and **Tonal Layers** rather than heavy shadows.

- **Base Layer:** The global background is a flat `#0d1117`.
- **Surface Layer:** Cards and containers use `rgba(255,255,255,0.04)` with a `backdrop-filter: blur(8px)`. This creates a sense of depth and focus.
- **Active Layer:** Elements that are selected or hovered gain a subtle glow. Hover states use a higher opacity surface (`0.08`), while active states use a teal border and a low-opacity teal inner glow (`rgba(13,148,136,0.15)`).
- **Shadows:** Avoid traditional black shadows. Instead, use colored ambient glows for primary buttons: `0 2px 12px rgba(13,148,136,0.3)`.

## Shapes

The design uses a variable roundedness strategy to distinguish between different UI scales:

- **Small Components (Inputs, Chips):** Use 10px radius for a compact, professional look.
- **Standard Containers (Cards):** Use 12px radius.
- **Major Sections (Chat, Dashboard Blocks):** Use 16px radius to group large sets of information.
- **Special Elements (Login Box):** Use 24px radius to create a soft, welcoming "entry point" for the user.
- **Chat Bubbles:** Asymmetric rounding (e.g., 14px on three corners, 4px on the corner nearest the tail) to indicate the direction of the speaker.

## Components

### Buttons
- **Primary:** Linear gradient from `#0d9488` to `#0891b2`. White text, 10px radius. On hover, apply a `translateY(-1px)` transform and increase shadow intensity.
- **Secondary/Ghost:** Subtle white border (10% opacity) with a transparent background.

### Social Chips
- **States:** 
  - *Inactive:* 4% white surface, 55% opacity text, grayscale icon.
  - *Active:* 15% teal background, 50% teal border, `#5eead4` text, and full-color icon.
- **Transition:** 150ms ease-in-out for all state changes.

### Cards (Discovery Phase)
- **Structure:** 72x72px thumbnail, 2-line title clamp, and secondary metadata.
- **Selection:** Use a teal-tinted background and a high-contrast checkbox in the top-right corner.

### Inputs & Selectors
- **Style:** 10px radius, 4% white background, 10% white border.
- **Focus:** Border transitions to `accent-blue` (`#63b3ed`) with a subtle outer glow.

### Chat RAG
- **User Bubbles:** Teal-tinted background (`rgba(13,148,136,0.18)`).
- **Bot Bubbles:** Surface-colored background (`rgba(255,255,255,0.05)`).
- **Visuals:** Ensure monospaced fonts are used for code snippets or data citations within the chat.

### Data Dashboard
- **Charts:** Use Plotly with a transparent background. Sentiment colors must strictly follow the named color tokens (Positive: Green, Negative: Red, Neutral: Gray).