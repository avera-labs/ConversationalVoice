/**
 * Generate file://-compatible caption and waveform data for expansion demos.
 * Usage: node scripts/generate-expansion-assets.mjs <example> [<example> ...]
 * Example: node scripts/generate-expansion-assets.mjs old-movie tv-talk-show
 */

import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** @typedef {{text: string, t0: number, t1: number}} Utterance */
/** @typedef {{word: string, t0: number, t1: number}} AlignedWord */
/** @typedef {{s0: Utterance[], s1: Utterance[]}} Transcript */
/** @typedef {{s0: AlignedWord[], s1: AlignedWord[]}} WordAlignment */

const PEAK_COUNT = 420;
const ANALYSIS_SAMPLE_RATE = 8000;
const EXAMPLES = /** @type {readonly string[]} */ (['old-movie', 'tv-talk-show']);

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, '..');
const requestedExamples = process.argv.slice(2);

if (requestedExamples.length === 0) {
  throw new Error(`Provide at least one example: ${EXAMPLES.join(', ')}`);
}

/**
 * Reads JSON without assuming optional metadata fields belong to the browser contract.
 * @template T
 * @param {string} filePath
 * @returns {T}
 */
function readJson(filePath) {
  return /** @type {T} */ (JSON.parse(readFileSync(filePath, 'utf8')));
}

/**
 * Treats the page's data attribute as the single source of truth for the cutoff.
 * @param {string} indexPath
 * @returns {number}
 */
function readOriginalEnd(indexPath) {
  const html = readFileSync(indexPath, 'utf8');
  const match = html.match(/\bdata-original-end="([^"]+)"/);
  const originalEnd = Number(match?.[1]);
  if (!Number.isFinite(originalEnd) || originalEnd <= 0) {
    throw new Error(`A positive data-original-end is required in ${indexPath}`);
  }
  return originalEnd;
}

/** @param {string} filePath @returns {number} */
function probeDuration(filePath) {
  const output = execFileSync(
    'ffprobe',
    ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', filePath],
    { encoding: 'utf8' },
  );
  const duration = Number(output.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error(`Could not determine duration for ${filePath}`);
  }
  return duration;
}

/** @param {string} filePath @returns {Float32Array} */
function decodeMonoAudio(filePath) {
  const buffer = execFileSync(
    'ffmpeg',
    [
      '-v', 'error', '-i', filePath, '-vn', '-ac', '1', '-ar', String(ANALYSIS_SAMPLE_RATE),
      '-f', 'f32le', 'pipe:1',
    ],
    { encoding: 'buffer', maxBuffer: 32 * 1024 * 1024 },
  );
  return new Float32Array(buffer.buffer, buffer.byteOffset, Math.floor(buffer.byteLength / 4));
}

/**
 * Samples audio against the shared full-duration clock so all tracks remain aligned.
 * The cutoff makes the original-data contract visually and audibly explicit.
 * @param {Float32Array} samples
 * @param {number} fullDuration
 * @param {number} audibleEnd
 * @returns {number[]}
 */
function buildPeaks(samples, fullDuration, audibleEnd) {
  /** @type {number[]} */
  const rawPeaks = [];
  for (let peakIndex = 0; peakIndex < PEAK_COUNT; peakIndex += 1) {
    const startTime = (peakIndex / PEAK_COUNT) * fullDuration;
    const endTime = Math.min(((peakIndex + 1) / PEAK_COUNT) * fullDuration, audibleEnd);
    if (startTime >= audibleEnd) {
      rawPeaks.push(0);
      continue;
    }
    const startSample = Math.max(0, Math.floor(startTime * ANALYSIS_SAMPLE_RATE));
    const endSample = Math.min(samples.length, Math.ceil(endTime * ANALYSIS_SAMPLE_RATE));
    let peak = 0;
    for (let sampleIndex = startSample; sampleIndex < endSample; sampleIndex += 1) {
      peak = Math.max(peak, Math.abs(samples[sampleIndex]));
    }
    rawPeaks.push(peak);
  }
  const maximumPeak = Math.max(...rawPeaks);
  if (maximumPeak === 0) return rawPeaks;
  return rawPeaks.map((peak) => Number((peak / maximumPeak).toFixed(3)));
}

/** @param {string} exampleName */
function generateExample(exampleName) {
  const exampleDirectory = join(repositoryRoot, 'assets', 'regeneration', exampleName);
  const expansionDirectory = join(exampleDirectory, 'expansion');
  const originalEnd = readOriginalEnd(join(expansionDirectory, 'index.html'));
  const speaker0Path = join(exampleDirectory, 'speaker-0.wav');
  const speaker1Path = join(exampleDirectory, 'speaker-1.wav');
  const sourcePath = join(exampleDirectory, 'source.mp4');
  const duration = probeDuration(speaker0Path);
  const speaker1Duration = probeDuration(speaker1Path);
  if (Math.abs(duration - speaker1Duration) > 0.01) {
    throw new Error(`${exampleName} speaker tracks must have matching durations.`);
  }
  if (originalEnd >= duration) {
    throw new Error(`${exampleName} originalEnd must be earlier than its speaker tracks.`);
  }

  const rawTranscript = /** @type {Transcript} */ (
    readJson(join(exampleDirectory, 'transcript.json'))
  );
  const rawAlignment = /** @type {WordAlignment} */ (
    readJson(join(exampleDirectory, 'word-alignment.json'))
  );
  const transcript = /** @type {Transcript} */ ({
    s0: rawTranscript.s0.map(({ text, t0, t1 }) => ({ text, t0, t1 })),
    s1: rawTranscript.s1.map(({ text, t0, t1 }) => ({ text, t0, t1 })),
  });
  const wordAlignment = /** @type {WordAlignment} */ ({
    s0: rawAlignment.s0.map(({ word, t0, t1 }) => ({ word, t0, t1 })),
    s1: rawAlignment.s1.map(({ word, t0, t1 }) => ({ word, t0, t1 })),
  });
  const waveformData = {
    duration: Number(duration.toFixed(6)),
    original: buildPeaks(decodeMonoAudio(sourcePath), duration, originalEnd),
    speaker0: buildPeaks(decodeMonoAudio(speaker0Path), duration, duration),
    speaker1: buildPeaks(decodeMonoAudio(speaker1Path), duration, duration),
  };

  mkdirSync(expansionDirectory, { recursive: true });
  writeFileSync(
    join(expansionDirectory, 'caption-data.js'),
    `/* Usage: load before voice-comparison.js to provide file://-compatible expansion captions. */\nwindow.DIALOGUE_TRANSCRIPT=${JSON.stringify(transcript)};\nwindow.DIALOGUE_WORD_ALIGNMENT=${JSON.stringify(wordAlignment)};\n`,
  );
  writeFileSync(
    join(expansionDirectory, 'waveform-peaks.js'),
    `/* Usage: load before voice-comparison.js to provide precomputed expansion waveforms. */\nwindow.DIALOGUE_WAVEFORM_PEAKS=${JSON.stringify(waveformData)};\n`,
  );
}

for (const exampleName of requestedExamples) {
  if (!EXAMPLES.includes(exampleName)) throw new Error(`Unknown expansion example: ${exampleName}`);
  generateExample(exampleName);
}
