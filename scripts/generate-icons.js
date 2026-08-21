#!/usr/bin/env node

/**
 * PWA Icon Generator
 * Generates PNG icons from the heart.svg source at required sizes.
 * Run: node scripts/generate-icons.js
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const SIZES = [72, 96, 128, 144, 152, 192, 384, 512];
const SVG_PATH = path.join(__dirname, '..', 'public', 'heart.svg');
const ICONS_DIR = path.join(__dirname, '..', 'public', 'icons');

// Ensure icons directory exists
if (!fs.existsSync(ICONS_DIR)) {
  fs.mkdirSync(ICONS_DIR, { recursive: true });
}

// Read the SVG and create a wrapped version with background
function createIconSvg(size, bgColor = '#1e40af') {
  const padding = Math.round(size * 0.2);
  const heartSize = size - (padding * 2);

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
  <rect width="${size}" height="${size}" rx="${Math.round(size * 0.18)}" fill="${bgColor}"/>
  <g transform="translate(${padding}, ${padding})">
    <svg viewBox="0 0 24 24" width="${heartSize}" height="${heartSize}">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" fill="white" stroke="white" stroke-width="0.5"/>
    </svg>
  </g>
</svg>`;
}

// Create maskable version with more padding
function createMaskableIconSvg(size) {
  const padding = Math.round(size * 0.25);
  const heartSize = size - (padding * 2);

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
  <rect width="${size}" height="${size}" fill="#1e40af"/>
  <g transform="translate(${padding}, ${padding})">
    <svg viewBox="0 0 24 24" width="${heartSize}" height="${heartSize}">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" fill="white" stroke="white" stroke-width="0.5"/>
    </svg>
  </g>
</svg>`;
}

console.log('🫀 Generating PWA icons...');

// Check if we have a way to convert SVG to PNG
// Try using sharp, then canvas, then fall back to just saving SVGs
let converter = null;

try {
  require.resolve('sharp');
  converter = 'sharp';
  console.log('  Using sharp for SVG→PNG conversion');
} catch {
  try {
    require.resolve('canvas');
    converter = 'canvas';
    console.log('  Using canvas for SVG→PNG conversion');
  } catch {
    console.log('  No SVG→PNG converter available (sharp/canvas)');
    console.log('  Saving SVG icons as placeholders — install "sharp" for PNG generation');
  }
}

async function generateWithSharp() {
  const sharp = require('sharp');
  for (const size of SIZES) {
    const svg = Buffer.from(createIconSvg(size));
    const outPath = path.join(ICONS_DIR, `icon-${size}.png`);
    await sharp(svg).resize(size, size).png().toFile(outPath);
    console.log(`  ✓ icon-${size}.png`);
  }

  // Maskable icon
  const maskableSvg = Buffer.from(createMaskableIconSvg(512));
  await sharp(maskableSvg).resize(512, 512).png().toFile(path.join(ICONS_DIR, 'icon-512-maskable.png'));
  console.log('  ✓ icon-512-maskable.png');
}

async function generateFallback() {
  // Save SVG versions as fallback
  for (const size of SIZES) {
    const svg = createIconSvg(size);
    const outPath = path.join(ICONS_DIR, `icon-${size}.svg`);
    fs.writeFileSync(outPath, svg);
    console.log(`  ⚠ icon-${size}.svg (install "sharp" for PNG)`);
  }

  // Also save the maskable SVG
  const maskableSvg = createMaskableIconSvg(512);
  fs.writeFileSync(path.join(ICONS_DIR, 'icon-512-maskable.svg'), maskableSvg);
  console.log('  ⚠ icon-512-maskable.svg (install "sharp" for PNG)');
}

(async () => {
  try {
    if (converter === 'sharp') {
      await generateWithSharp();
    } else {
      await generateFallback();
    }
    console.log('\n✅ Icon generation complete!');
  } catch (err) {
    console.error('Error generating icons:', err.message);
    process.exit(1);
  }
})();
