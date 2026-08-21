/**
 * Interactive original/reconstructed voice comparison controller.
 * Usage: load after a page-specific waveform-peaks.js file on any voice comparison page.
 * Set data-playback-duration on .player only when the comparison uses a fixed playback window.
 * Set data-continue-speaker-audio-after-video when longer speaker tracks should
 * take over the playback clock after the source video freezes on its last frame.
 * Set data-original-end to silence an irrelevant original tail, freeze the video,
 * and hand the shared clock to the longer speaker tracks at that timestamp.
 * Set data-static-full-timeline to render the complete waveform without the
 * original-to-expansion timeline transition.
 * Set data-speaker-offset to prepend player-managed silence to both speaker
 * tracks without modifying their source audio files.
 */
(() => {
  'use strict';

  // Configuration and contracts

  /** @typedef {'original' | 'enhanced'} AudioMode */
  /** @typedef {'original' | 'speaker0' | 'speaker1'} WaveformTrack */
  /** @typedef {'speaker0' | 'speaker1'} SpeakerTrack */
  /** @typedef {'loading' | 'ready' | 'error'} ResourceState */
  /** @typedef {'automatic' | 'user'} ModeChangeSource */
  /** @typedef {{word: string, t0: number, t1: number}} AlignedWord */
  /** @typedef {{text: string, t0: number, t1: number}} Utterance */
  /** @typedef {{text: string, isHighlighted?: boolean}} ExplanationPart */
  /** @typedef {{duration: number, original: number[], speaker0: number[], speaker1: number[]}} WaveformPeakData */

  const AUTOMATIC_SWITCH_INTERVAL_MS = 3000;
  const CAPTION_CHUNK_MAXIMUM_WORDS = 8;
  const CAPTION_CHUNK_PAUSE_SECONDS = .36;
  const EXPANSION_EXPLANATION_PARTS = /** @type {readonly ExplanationPart[]} */ ([
    { text: 'AI ' },
    { text: 'expands', isHighlighted: true },
    { text: ' this segment from the real conversion.' },
  ]);
  const EXPANSION_EXPLANATION = EXPANSION_EXPLANATION_PARTS.map(({ text }) => text).join('');
  const MEDIA_SYNC_TOLERANCE_SECONDS = .12;
  const TIMELINE_EXPANSION_TRANSITION_MS = 900;
  const WAVEFORM_CENTER_Y = 15;
  const WAVEFORM_MAXIMUM_HEIGHT = 13;
  const WAVEFORM_TIME_LABEL_INTERVAL_SECONDS = 30;
  const WAVEFORM_TIME_TICK_INTERVAL_SECONDS = 10;
  const WAVEFORM_VIEWBOX_WIDTH = 1000;
  const SPEAKER_TRACKS = /** @type {const} */ (['speaker0', 'speaker1']);
  const MODE_LABELS = /** @type {Record<AudioMode, string>} */ ({
    original: 'Original audio',
    enhanced: 'Reconstructed audio',
  });

  /**
   * Fails early when the page structure and controller contract drift apart.
   * @param {ParentNode} root
   * @param {string} selector
   * @returns {Element}
   */
  function requireElement(root, selector) {
    const element = root.querySelector(selector);
    if (!element) throw new Error(`Required element is missing: ${selector}`);
    return element;
  }

  /**
   * Parses an optional positive numeric data attribute.
   * @param {string | undefined} value
   * @param {string} attributeName
   * @returns {number | null}
   */
  function parseOptionalPositiveNumber(value, attributeName) {
    if (value === undefined) return null;
    const parsedValue = Number(value);
    if (!Number.isFinite(parsedValue) || parsedValue <= 0) {
      throw new TypeError(`${attributeName} must be a positive number.`);
    }
    return parsedValue;
  }

  // DOM contract and page configuration

  const player = /** @type {HTMLElement} */ (requireElement(document, '.player'));
  const configuredPlaybackDuration = parseOptionalPositiveNumber(
    player.dataset.playbackDuration,
    'data-playback-duration',
  );
  const continueSpeakerAudioAfterVideo = player.hasAttribute('data-continue-speaker-audio-after-video');
  const staticFullTimeline = player.hasAttribute('data-static-full-timeline');
  const originalEndSeconds = parseOptionalPositiveNumber(
    player.dataset.originalEnd,
    'data-original-end',
  );
  const expansionStartSeconds = parseOptionalPositiveNumber(
    player.dataset.reconstructionEnd,
    'data-reconstruction-end',
  );
  const speakerOffsetSeconds = parseOptionalPositiveNumber(
    player.dataset.speakerOffset,
    'data-speaker-offset',
  ) || 0;
  const video = /** @type {HTMLVideoElement} */ (requireElement(player, 'video'));
  const playButton = /** @type {HTMLButtonElement} */ (requireElement(player, '.big-play'));
  const brandOutro = /** @type {HTMLButtonElement | null} */ (player.querySelector('.brand-outro'));
  const waveformDock = /** @type {HTMLElement} */ (requireElement(player, '.waveform-dock'));
  const referenceWaveform = /** @type {SVGSVGElement} */ (requireElement(waveformDock, '.dock-wave'));
  const waveformTimeTrack = /** @type {HTMLElement | null} */ (waveformDock.querySelector('.waveform-time-track'));
  const status = /** @type {HTMLElement} */ (requireElement(player, '.mode-status'));
  const statusCopy = /** @type {HTMLElement} */ (requireElement(status, '.mode-copy'));
  const captionPanel = /** @type {HTMLElement | null} */ (player.querySelector('.stage-captions'));
  const speakerAudios = /** @type {Record<SpeakerTrack, HTMLAudioElement>} */ ({
    speaker0: /** @type {HTMLAudioElement} */ (requireElement(player, '[data-speaker-track="speaker0"]')),
    speaker1: /** @type {HTMLAudioElement} */ (requireElement(player, '[data-speaker-track="speaker1"]')),
  });
  const speakerAudioElements = SPEAKER_TRACKS.map((track) => speakerAudios[track]);
  const muteButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (player.querySelectorAll('.track-mute'));
  const speakerRows = /** @type {Record<SpeakerTrack, HTMLElement>} */ ({
    speaker0: /** @type {HTMLElement} */ (requireElement(player, '.waveform-row.speaker0')),
    speaker1: /** @type {HTMLElement} */ (requireElement(player, '.waveform-row.speaker1')),
  });
  const modeButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (player.querySelectorAll('.mode-button'));
  const wheelSwitchTargets = /** @type {NodeListOf<HTMLElement>} */ (player.querySelectorAll('.stage, .mode-controls'));
  if (muteButtons.length !== SPEAKER_TRACKS.length || modeButtons.length !== 2) {
    throw new Error('Voice comparison requires two mode buttons and two speaker mute buttons.');
  }

  // Runtime state and waveform data

  /** @type {AudioMode} */
  let activeMode = 'original';
  /** @type {ResourceState} */
  let resourceState = 'loading';
  /** @type {SpeakerTrack | null} */
  let mutedSpeakerTrack = null;
  /** @type {number | null} */
  let automaticSwitchTimer = null;
  /** @type {number | null} */
  let originalCutoffTimer = null;
  /** @type {number | null} */
  let speakerStartTimer = null;
  /** @type {number | null} */
  let timelineTransitionTimer = null;
  let automaticSwitchingEnabled = true;
  /** @type {boolean} */
  let speakerContinuationActive = false;

  const rawWaveformPeaks = /** @type {WaveformPeakData} */ (
    window.DIALOGUE_WAVEFORM_PEAKS
  );
  if (!rawWaveformPeaks
    || !Number.isFinite(rawWaveformPeaks.duration)
    || rawWaveformPeaks.duration <= 0
    || !SPEAKER_TRACKS.every((track) => Array.isArray(rawWaveformPeaks[track]))
    || !Array.isArray(rawWaveformPeaks.original)) {
    throw new TypeError('Waveform peak data is missing or invalid.');
  }
  const rawWaveformLength = rawWaveformPeaks.original.length;
  if (rawWaveformLength === 0
    || ![rawWaveformPeaks.original, rawWaveformPeaks.speaker0, rawWaveformPeaks.speaker1]
      .every((peaks) => peaks.length === rawWaveformLength
        && peaks.every((peak) => Number.isFinite(peak) && peak >= 0 && peak <= 1))) {
    throw new TypeError('Waveform tracks must contain equally sized, normalized peak arrays.');
  }

  /**
   * Samples a precomputed peak track at an absolute source timestamp.
   * Out-of-range timestamps deliberately resolve to silence so waveform data
   * follows the same contract as the player-managed speaker offset.
   * @param {number[]} peaks
   * @param {number} sourceTime
   * @param {number} sourceDuration
   * @returns {number}
   */
  function sampleWaveformPeak(peaks, sourceTime, sourceDuration) {
    if (sourceTime < 0 || sourceTime >= sourceDuration) return 0;
    const sampleIndex = Math.min(
      peaks.length - 1,
      Math.floor((sourceTime / sourceDuration) * peaks.length),
    );
    return peaks[sampleIndex];
  }

  /**
   * Extends the visual timeline and shifts only speaker content to the right.
   * Resampling preserves absolute timing for the original track while adding
   * an exact-duration silent prefix to both unmodified speaker files.
   * @param {WaveformPeakData} source
   * @param {number} offsetSeconds
   * @returns {WaveformPeakData}
   */
  function applySpeakerWaveformOffset(source, offsetSeconds) {
    if (offsetSeconds === 0) return source;
    const duration = source.duration + offsetSeconds;
    const sampleCount = Math.ceil((source.original.length * duration) / source.duration);
    const sampleTimes = Array.from(
      { length: sampleCount },
      (_, index) => ((index + .5) / sampleCount) * duration,
    );
    return {
      duration,
      original: sampleTimes.map((time) => sampleWaveformPeak(source.original, time, source.duration)),
      speaker0: sampleTimes.map((time) => sampleWaveformPeak(
        source.speaker0,
        time - offsetSeconds,
        source.duration,
      )),
      speaker1: sampleTimes.map((time) => sampleWaveformPeak(
        source.speaker1,
        time - offsetSeconds,
        source.duration,
      )),
    };
  }

  const waveformPeaks = applySpeakerWaveformOffset(rawWaveformPeaks, speakerOffsetSeconds);
  if (configuredPlaybackDuration !== null && waveformPeaks.duration !== configuredPlaybackDuration) {
    throw new Error(`Waveform duration must be ${configuredPlaybackDuration}s.`);
  }
  if (originalEndSeconds !== null && !continueSpeakerAudioAfterVideo) {
    throw new Error('data-original-end requires data-continue-speaker-audio-after-video.');
  }
  if (originalEndSeconds !== null && originalEndSeconds >= waveformPeaks.duration) {
    throw new Error('data-original-end must be earlier than the waveform duration.');
  }
  if (expansionStartSeconds !== null && expansionStartSeconds >= waveformPeaks.duration) {
    throw new Error('data-reconstruction-end must be earlier than the waveform duration.');
  }
  const waveformLength = waveformPeaks.original.length;
  if (originalEndSeconds !== null) {
    // data-original-end is both an audio cutoff and a visual-data contract. A
    // non-zero original tail could otherwise imply that unavailable audio exists.
    const originalCutoffIndex = Math.ceil(
      (originalEndSeconds / waveformPeaks.duration) * waveformLength,
    );
    if (waveformPeaks.original.slice(originalCutoffIndex).some((peak) => peak > 0)) {
      throw new Error('Original waveform peaks must be silent after data-original-end.');
    }
    if (SPEAKER_TRACKS.some((track) => waveformPeaks[track]
      .slice(originalCutoffIndex)
      .every((peak) => peak === 0))) {
      throw new Error('Each speaker waveform must continue after data-original-end.');
    }
  }
  const waveformTracks = /** @type {Record<WaveformTrack, number[]>} */ ({
    original: waveformPeaks.original,
    speaker0: waveformPeaks.speaker0,
    speaker1: waveformPeaks.speaker1,
  });
  if (originalEndSeconds !== null) {
    const originalPhaseRatio = originalEndSeconds / waveformPeaks.duration;
    player.style.setProperty('--original-phase-ratio', originalPhaseRatio.toFixed(6));
    player.style.setProperty('--timeline-condensed-scale', (1 / originalPhaseRatio).toFixed(6));
    player.style.setProperty('--timeline-condensed-clip', `${((1 - originalPhaseRatio) * 100).toFixed(3)}%`);
  }
  const captionTranscript = /** @type {Record<'s0' | 's1', Utterance[]> | undefined} */ (
    window.DIALOGUE_TRANSCRIPT
  );
  const captionAlignment = /** @type {Record<'s0' | 's1', AlignedWord[]> | undefined} */ (
    window.DIALOGUE_WORD_ALIGNMENT
  );
  if (captionPanel && (!captionTranscript || !captionAlignment)) {
    throw new TypeError('Caption panel requires transcript and word-alignment data.');
  }

  /** @param {number} value @returns {number} */
  function clampRatio(value) {
    return Math.min(1, Math.max(0, value));
  }

  /** Uses the active page contract to resolve the complete shared timeline. @returns {number} */
  function getPlaybackDuration() {
    if (configuredPlaybackDuration !== null) return configuredPlaybackDuration;
    if (continueSpeakerAudioAfterVideo) {
      const speakerDurations = speakerAudioElements
        .map((audio) => audio.duration)
        .filter((duration) => Number.isFinite(duration) && duration > 0);
      if (speakerDurations.length === speakerAudioElements.length) {
        return Math.max(...speakerDurations) + speakerOffsetSeconds;
      }
    }
    if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
    return waveformPeaks.duration;
  }

  /** Returns the duration currently represented across the full waveform width. */
  function getVisibleTimelineDuration() {
    if (originalEndSeconds !== null && !player.classList.contains('is-timeline-expanded')) {
      return originalEndSeconds;
    }
    return getPlaybackDuration();
  }

  /** Returns the media position that currently owns the shared playhead. @returns {number} */
  function getPlaybackTime() {
    return speakerContinuationActive
      ? Math.max(...speakerAudioElements.map((audio) => audio.currentTime)) + speakerOffsetSeconds
      : video.currentTime;
  }

  /** Returns the timestamp where the video/original phase yields to speaker-only playback. @returns {number} */
  function getOriginalEndTime() {
    if (originalEndSeconds !== null) return originalEndSeconds;
    return Number.isFinite(video.duration) && video.duration > 0 ? video.duration : waveformPeaks.duration;
  }

  /** Prevents every render path from restoring original audio past its HTML-configured cutoff. */
  function hasOriginalExpired() {
    return originalEndSeconds !== null
      && (speakerContinuationActive || video.currentTime >= originalEndSeconds);
  }

  /** Treats the video and speaker-only continuation as one logical playback state. @returns {boolean} */
  function isPlaybackPaused() {
    if (!speakerContinuationActive) return video.paused;
    return speakerAudioElements.every((audio) => audio.paused || audio.ended);
  }

  // Waveform rendering and synchronization

  /**
   * Builds one closed, mirrored amplitude envelope from decoded audio peaks.
   * @param {number[]} peaks
   * @returns {string}
   */
  function buildWaveformPath(peaks) {
    const lastIndex = Math.max(1, peaks.length - 1);
    const upper = peaks.map((peak, index) => {
      const x = (index / lastIndex) * WAVEFORM_VIEWBOX_WIDTH;
      const y = WAVEFORM_CENTER_Y - peak * WAVEFORM_MAXIMUM_HEIGHT;
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    });
    const lower = peaks.map((peak, index) => {
      const reversedIndex = peaks.length - 1 - index;
      const x = (reversedIndex / lastIndex) * WAVEFORM_VIEWBOX_WIDTH;
      const y = WAVEFORM_CENTER_Y + peaks[reversedIndex] * WAVEFORM_MAXIMUM_HEIGHT;
      return `L${x.toFixed(2)},${y.toFixed(2)}`;
    });
    return `${upper.join(' ')} ${lower.join(' ')} Z`;
  }

  /** Draws every waveform from the precomputed peak contract. */
  function renderWaveforms() {
    player.querySelectorAll('[data-wave-track]').forEach((element) => {
      const path = /** @type {SVGPathElement} */ (element);
      const track = /** @type {WaveformTrack} */ (path.dataset.waveTrack);
      const peaks = waveformTracks[track];
      if (!Array.isArray(peaks)) throw new Error(`Missing waveform track: ${track}`);
      path.setAttribute('d', buildWaveformPath(peaks));
    });
  }

  /** Builds a ten-second ruler with labels at thirty-second intervals. */
  function renderWaveformTimeScale() {
    if (!waveformTimeTrack) return;
    const ticks = document.createDocumentFragment();
    for (
      let seconds = 0;
      seconds < waveformPeaks.duration;
      seconds += WAVEFORM_TIME_TICK_INTERVAL_SECONDS
    ) {
      const tick = document.createElement('span');
      tick.className = 'waveform-time-tick';
      tick.style.setProperty(
        '--time-position-full',
        `${((seconds / waveformPeaks.duration) * 100).toFixed(3)}%`,
      );
      if (originalEndSeconds !== null) {
        tick.style.setProperty(
          '--time-position-condensed',
          `${(clampRatio(seconds / originalEndSeconds) * 100).toFixed(3)}%`,
        );
        tick.classList.toggle('is-after-original', seconds > originalEndSeconds);
      }
      if (seconds === 0) tick.classList.add('is-start');
      if (seconds % WAVEFORM_TIME_LABEL_INTERVAL_SECONDS === 0) {
        tick.classList.add('is-labeled');
        tick.textContent = `${seconds}s`;
      }
      ticks.append(tick);
    }
    waveformTimeTrack.replaceChildren(ticks);
  }

  /**
   * Maps the shared playhead into a possibly shorter reconstructed source.
   * @param {HTMLAudioElement} audio
   * @returns {number}
   */
  function getEnhancedSyncTime(audio) {
    if (!Number.isFinite(audio.duration)) return 0;
    return Math.min(Math.max(0, video.currentTime - speakerOffsetSeconds), audio.duration);
  }

  /** Synchronizes enhanced audio while treating its missing tail as silence. */
  function synchronizeEnhancedAudio() {
    // Once the video ends, speaker audio owns the clock and must not be pulled
    // back to the frozen video's final timestamp.
    if (speakerContinuationActive) return;
    speakerAudioElements.forEach((audio) => {
      if (audio.readyState === HTMLMediaElement.HAVE_NOTHING) return;
      const targetTime = getEnhancedSyncTime(audio);
      if (video.currentTime < speakerOffsetSeconds) audio.pause();
      if (Math.abs(audio.currentTime - targetTime) > MEDIA_SYNC_TOLERANCE_SECONDS) audio.currentTime = targetTime;
    });
  }

  /**
   * Splits a transcript sentence into screen-sized spoken phrases using real
   * pauses, punctuation, and a maximum word count rather than rewriting text.
   * @param {AlignedWord[]} words
   * @returns {AlignedWord[][]}
   */
  function splitCaptionWords(words) {
    /** @type {AlignedWord[][]} */
    const chunks = [];
    /** @type {AlignedWord[]} */
    let activeChunk = [];

    words.forEach((word) => {
      const previousWord = activeChunk.at(-1);
      const followsPause = previousWord
        ? word.t0 - previousWord.t1 >= CAPTION_CHUNK_PAUSE_SECONDS
        : false;
      if (activeChunk.length > 0
        && (followsPause || activeChunk.length >= CAPTION_CHUNK_MAXIMUM_WORDS)) {
        chunks.push(activeChunk);
        activeChunk = [];
      }
      activeChunk.push(word);
      if (/[.!?…—]$/.test(word.word)) {
        chunks.push(activeChunk);
        activeChunk = [];
      }
    });
    if (activeChunk.length > 0) chunks.push(activeChunk);
    return chunks;
  }

  /** Renders homepage-derived speaker captions at the current shared playhead. */
  function renderCaptions() {
    if (!captionPanel || !captionTranscript || !captionAlignment) return;
    const captionTime = getPlaybackTime() - speakerOffsetSeconds;
    const fragment = document.createDocumentFragment();

    if (activeMode === 'enhanced') {
      SPEAKER_TRACKS.forEach((track) => {
        const speakerKey = track === 'speaker0' ? 's0' : 's1';
        const activeUtterance = captionTranscript[speakerKey]
          .find(({ t0, t1 }) => captionTime >= t0 && captionTime <= t1);
        if (!activeUtterance) return;

        const utteranceWords = captionAlignment[speakerKey]
          .filter(({ t0, t1 }) => t1 >= activeUtterance.t0 && t0 <= activeUtterance.t1);
        const activeChunk = splitCaptionWords(utteranceWords)
          .find((chunk) => captionTime >= chunk[0].t0 && captionTime <= chunk.at(-1).t1);
        if (!activeChunk) return;

        const speakerLabel = track === 'speaker0' ? 'Speaker 0' : 'Speaker 1';
        const row = document.createElement('div');
        row.className = 'stage-caption-row';
        row.setAttribute('aria-label', `${speakerLabel}: ${activeChunk.map(({ word }) => word).join(' ')}`);

        const label = document.createElement('span');
        label.className = 'stage-caption-speaker';
        label.textContent = speakerLabel;
        row.appendChild(label);

        activeChunk.forEach(({ word, t0, t1 }) => {
          const token = document.createElement('span');
          token.className = 'stage-caption-word';
          token.classList.toggle('is-active', captionTime >= t0 && captionTime <= t1);
          token.textContent = word;
          row.appendChild(token);
        });
        fragment.appendChild(row);
      });
    }

    captionPanel.replaceChildren(fragment);
    const hasCaption = captionPanel.childElementCount > 0;
    captionPanel.classList.toggle('is-visible', hasCaption);
    captionPanel.setAttribute('aria-hidden', String(!hasCaption));
  }

  // View rendering

  /** Updates speaker audibility and controls without touching playback position. */
  function renderSpeakerMuteState() {
    const isEnhanced = activeMode === 'enhanced';
    SPEAKER_TRACKS.forEach((track) => {
      speakerAudios[track].muted = !isEnhanced || mutedSpeakerTrack === track;
    });
    muteButtons.forEach((button) => {
      const track = /** @type {SpeakerTrack} */ (button.dataset.muteTrack);
      const isMuted = mutedSpeakerTrack === track;
      button.textContent = isMuted ? 'Unmute' : 'Mute';
      button.setAttribute('aria-label', `${isMuted ? 'Unmute' : 'Mute'} ${track === 'speaker0' ? 'Speaker 0' : 'Speaker 1'}`);
      button.setAttribute('aria-pressed', String(isMuted));
      button.setAttribute('aria-disabled', String(!isEnhanced));
      speakerRows[track].classList.toggle('is-muted', isMuted);
    });
  }

  /** Updates status copy for resource, mode, and dialogue-expansion phases. */
  function renderStatusCopy() {
    const explanationStartSeconds = Math.max(
      expansionStartSeconds === null ? 0 : expansionStartSeconds + speakerOffsetSeconds,
      originalEndSeconds || 0,
    );
    const isExplainingExpansion = resourceState === 'ready'
      && expansionStartSeconds !== null
      && getPlaybackTime() >= explanationStartSeconds;
    status.classList.toggle('is-explaining', isExplainingExpansion);
    const copyState = resourceState === 'loading'
      ? 'loading'
      : resourceState === 'error'
        ? 'error'
        : isExplainingExpansion ? 'expansion' : activeMode;
    if (statusCopy.dataset.copyState === copyState) return;
    statusCopy.dataset.copyState = copyState;

    if (!isExplainingExpansion) {
      statusCopy.removeAttribute('aria-label');
      statusCopy.textContent = resourceState === 'loading'
        ? 'Loading media…'
        : resourceState === 'error' ? 'Media failed to load' : MODE_LABELS[activeMode];
      return;
    }

    const explanation = document.createElement('span');
    explanation.className = 'mode-explanation';
    EXPANSION_EXPLANATION_PARTS.forEach(({ text, isHighlighted }) => {
      if (!isHighlighted) {
        explanation.append(text);
        return;
      }

      const emphasis = document.createElement('span');
      emphasis.className = 'mode-explanation-emphasis';
      emphasis.textContent = text;
      explanation.append(emphasis);
    });
    statusCopy.setAttribute('aria-label', EXPANSION_EXPLANATION);
    statusCopy.replaceChildren(explanation);
  }

  /** Reflects preload readiness without overwriting the selected audio mode. */
  function renderResourceState() {
    const isReady = resourceState === 'ready';
    player.dataset.resourceState = resourceState;
    playButton.disabled = !isReady;
    playButton.textContent = resourceState === 'loading' ? '…' : '▶';
    playButton.setAttribute('aria-label', resourceState === 'loading'
      ? 'Loading demo media'
      : resourceState === 'error' ? 'Demo media failed to load' : 'Play demo');
    renderStatusCopy();
  }

  /** Applies the current mode to status, controls, waveforms, and audible sources. */
  function renderMode() {
    const isEnhanced = activeMode === 'enhanced';
    status.dataset.mode = activeMode;
    player.classList.toggle('is-enhanced', isEnhanced);
    modeButtons.forEach((button) => {
      const isActive = button.dataset.audioMode === activeMode;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', String(isActive));
    });
    synchronizeEnhancedAudio();
    video.muted = isEnhanced || hasOriginalExpired();
    renderSpeakerMuteState();
    renderResourceState();
    renderCaptions();
    requestAnimationFrame(updateProgress);
  }

  // Resource loading

  /**
   * Resolves once a required media element exposes its timeline metadata. iOS may decline
   * to buffer future media data before a user gesture, so preload must not gate the controls.
   * @param {HTMLMediaElement} media
   * @returns {Promise<void>}
   */
  function waitUntilMetadataLoaded(media) {
    if (media.error) return Promise.reject(media.error);
    if (media.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        media.removeEventListener('loadedmetadata', handleLoadedMetadata);
        media.removeEventListener('error', handleError);
      };
      const handleLoadedMetadata = () => { cleanup(); resolve(); };
      const handleError = () => { cleanup(); reject(media.error || new Error('Media preload failed.')); };
      media.addEventListener('loadedmetadata', handleLoadedMetadata, { once: true });
      media.addEventListener('error', handleError, { once: true });
    });
  }

  /**
   * Starts lightweight metadata loading without making iOS wait for speculative buffering.
   * Actual playback remains synchronized and begins from a direct user gesture when required.
   */
  async function preloadMedia() {
    const requiredMedia = [video, ...speakerAudioElements];
    requiredMedia.forEach((media) => media.load());
    const metadataReadiness = requiredMedia.map(waitUntilMetadataLoaded);
    resourceState = 'ready';
    renderResourceState();
    void playMedia();
    try {
      await Promise.all(metadataReadiness);
      if (originalEndSeconds !== null
        && Number.isFinite(video.duration)
        && originalEndSeconds >= video.duration) {
        throw new RangeError('data-original-end must be earlier than the source video duration.');
      }
    } catch (error) {
      resourceState = 'error';
      renderResourceState();
      console.error('Required demo media metadata could not be loaded.', error);
    }
  }

  // Playback and mode state transitions

  /**
   * Toggles one speaker while enforcing the two-speaker contract: zero or one muted track.
   * @param {SpeakerTrack} track
   */
  function toggleSpeakerMute(track) {
    mutedSpeakerTrack = mutedSpeakerTrack === track ? null : track;
    renderSpeakerMuteState();
  }

  /** Clears the active interval while preserving whether automatic switching is enabled. */
  function clearAutomaticSwitchTimer() {
    if (automaticSwitchTimer === null) return;
    window.clearInterval(automaticSwitchTimer);
    automaticSwitchTimer = null;
  }

  /** Clears the media-time cutoff scheduler when playback pauses or seeks. */
  function clearOriginalCutoffTimer() {
    if (originalCutoffTimer === null) return;
    window.clearTimeout(originalCutoffTimer);
    originalCutoffTimer = null;
  }

  /** Clears the delayed speaker start whenever the video pauses or seeks. */
  function clearSpeakerStartTimer() {
    if (speakerStartTimer === null) return;
    window.clearTimeout(speakerStartTimer);
    speakerStartTimer = null;
  }

  /** Starts both speaker files at their offset-adjusted position. */
  async function startSynchronizedSpeakerAudio() {
    if (speakerContinuationActive
      || video.paused
      || video.currentTime < speakerOffsetSeconds) return;
    synchronizeEnhancedAudio();
    try {
      await Promise.all(speakerAudioElements
        .filter((audio) => !audio.ended)
        .map((audio) => audio.play()));
    } catch (error) {
      pauseMedia();
      console.error('Offset speaker playback could not start.', error);
    }
  }

  /**
   * Schedules speaker time zero at the configured shared-timeline offset.
   * A timeupdate fallback also invokes the same idempotent start path when
   * browsers throttle background timers.
   */
  function scheduleSpeakerStart() {
    clearSpeakerStartTimer();
    if (speakerOffsetSeconds === 0 || speakerContinuationActive || video.paused) return;
    const remainingMediaSeconds = speakerOffsetSeconds - video.currentTime;
    if (remainingMediaSeconds <= 0) {
      void startSynchronizedSpeakerAudio();
      return;
    }
    const playbackRate = Number.isFinite(video.playbackRate) && video.playbackRate > 0
      ? video.playbackRate
      : 1;
    speakerStartTimer = window.setTimeout(() => {
      speakerStartTimer = null;
      if (video.currentTime < speakerOffsetSeconds) {
        scheduleSpeakerStart();
        return;
      }
      void startSynchronizedSpeakerAudio();
    }, (remainingMediaSeconds / playbackRate) * 1000);
  }

  /**
   * Schedules the original-to-speaker handoff from data-original-end. The
   * timeupdate listener remains a fallback for throttled background timers.
   */
  function scheduleOriginalCutoff() {
    clearOriginalCutoffTimer();
    if (originalEndSeconds === null || speakerContinuationActive || video.paused) return;
    const remainingMediaSeconds = originalEndSeconds - video.currentTime;
    if (remainingMediaSeconds <= 0) {
      video.muted = true;
      void beginSpeakerContinuation();
      return;
    }
    const playbackRate = Number.isFinite(video.playbackRate) && video.playbackRate > 0
      ? video.playbackRate
      : 1;
    originalCutoffTimer = window.setTimeout(() => {
      originalCutoffTimer = null;
      video.muted = true;
      void beginSpeakerContinuation();
    }, (remainingMediaSeconds / playbackRate) * 1000);
  }

  /** Permanently stops the passive comparison guide after explicit user input or completion. */
  function stopAutomaticSwitching() {
    automaticSwitchingEnabled = false;
    clearAutomaticSwitchTimer();
  }

  /**
   * Changes mode through one state transition path for automatic and user input.
   * @param {AudioMode} mode
   * @param {ModeChangeSource} source
   * @returns {boolean}
   */
  function setMode(mode, source) {
    if (mode === activeMode) return false;
    if (source === 'user') stopAutomaticSwitching();
    activeMode = mode;
    renderMode();
    return true;
  }

  /** Alternates modes until playback ends or a user changes the mode. */
  function startAutomaticSwitching() {
    if (speakerContinuationActive
      || !automaticSwitchingEnabled
      || automaticSwitchTimer !== null
      || video.paused) return;
    automaticSwitchTimer = window.setInterval(() => {
      const nextMode = activeMode === 'original' ? 'enhanced' : 'original';
      setMode(nextMode, 'automatic');
    }, AUTOMATIC_SWITCH_INTERVAL_MS);
  }

  /**
   * Starts the video and both speaker sources so mode changes remain immediate.
   * Returns false when browser autoplay policy or a media error prevents a synchronized start.
   * @returns {Promise<boolean>}
   */
  async function playMedia() {
    if (resourceState !== 'ready') return false;
    clearBrandOutro();
    if (getPlaybackTime() >= getPlaybackDuration()) {
      speakerContinuationActive = false;
      video.currentTime = 0;
      speakerAudioElements.forEach((audio) => { audio.currentTime = 0; });
      if (continueSpeakerAudioAfterVideo) setMode('original', 'automatic');
    }
    synchronizeEnhancedAudio();
    renderMode();
    try {
      const playPromises = [];
      if (!speakerContinuationActive) playPromises.push(video.play());
      if (speakerContinuationActive || video.currentTime >= speakerOffsetSeconds) {
        speakerAudioElements.forEach((audio) => {
          const audioTime = speakerContinuationActive ? audio.currentTime : getEnhancedSyncTime(audio);
          if (!audio.ended && (!Number.isFinite(audio.duration) || audioTime < audio.duration)) {
            playPromises.push(audio.play());
          }
        });
      }
      await Promise.all(playPromises);
      return true;
    } catch (error) {
      // Some browsers may start muted sources while rejecting audible autoplay.
      // Pause every source so the shared playhead never enters a partial-playing state.
      pauseMedia();
      if (!(error instanceof DOMException && error.name === 'NotAllowedError')) {
        console.error('Media playback could not start.', error);
      }
      return false;
    }
  }

  /** Pauses the video and both reconstructed speaker sources. */
  function pauseMedia() {
    clearOriginalCutoffTimer();
    clearSpeakerStartTimer();
    video.pause();
    speakerAudioElements.forEach((audio) => audio.pause());
  }

  /** Cancels the active branded ending before replay or seeking. */
  function clearBrandOutro() {
    if (!brandOutro) return;
    player.classList.remove('is-outro-running');
    brandOutro.setAttribute('aria-hidden', 'true');
  }

  /** Uses the final-frame hold as the first phase of a continuous branded crossfade. */
  function beginBrandOutro() {
    if (!brandOutro) {
      completePlaybackWindow();
      return;
    }
    if (player.classList.contains('is-outro-running')) return;

    stopAutomaticSwitching();
    pauseMedia();
    player.classList.remove('is-playing');
    // The CSS delay preserves the one-second final-frame beat without a second
    // state change that could cancel the background crossfade.
    player.classList.add('is-outro-running');
    brandOutro.setAttribute('aria-hidden', 'false');
  }

  /** Toggles all synchronized sources through the video playback state. */
  function togglePlayback() {
    if (resourceState !== 'ready') return;
    if (isPlaybackPaused()) void playMedia();
    else pauseMedia();
  }

  /**
   * Preserves native Space behavior for interactive or editable controls.
   * @param {EventTarget | null} target
   * @returns {boolean}
   */
  function isKeyboardInteractiveTarget(target) {
    return target instanceof Element
      && Boolean(target.closest('button, a[href], input, select, textarea, [contenteditable="true"]'));
  }

  /**
   * Uses Space as a page-level playback shortcut without triggering scroll or key-repeat toggles.
   * @param {KeyboardEvent} event
   */
  function togglePlaybackFromKeyboard(event) {
    const isSpaceKey = event.code === 'Space' || event.key === ' ';
    if (!isSpaceKey
      || event.repeat
      || event.defaultPrevented
      || resourceState !== 'ready'
      || isKeyboardInteractiveTarget(event.target)) return;
    event.preventDefault();
    togglePlayback();
  }

  /**
   * Maps horizontal arrow keys to the two comparison modes.
   * @param {KeyboardEvent} event
   */
  function switchModeFromKeyboard(event) {
    if ((event.key !== 'ArrowLeft' && event.key !== 'ArrowRight')
      || event.repeat
      || event.defaultPrevented
      || isKeyboardInteractiveTarget(event.target)) return;
    event.preventDefault();
    setMode(event.key === 'ArrowLeft' ? 'original' : 'enhanced', 'user');
  }

  // Progress and seeking

  /** Keeps the playhead aligned to the plotted area rather than its label column. */
  function updateWaveformProgress(ratio) {
    const dockBounds = waveformDock.getBoundingClientRect();
    const waveBounds = (waveformTimeTrack ?? referenceWaveform).getBoundingClientRect();
    const left = waveBounds.left - dockBounds.left + clampRatio(ratio) * waveBounds.width;
    waveformDock.style.setProperty('--waveform-progress', `${left.toFixed(2)}px`);
  }

  /** Animates between the focused original window and the complete generated timeline. */
  function renderTimelineWindow(playbackTime) {
    if (originalEndSeconds === null) return;
    if (staticFullTimeline) {
      player.classList.add('is-timeline-expanded');
      player.classList.remove('is-timeline-transitioning');
      return;
    }
    const shouldExpandTimeline = playbackTime >= originalEndSeconds;
    if (player.classList.contains('is-timeline-expanded') === shouldExpandTimeline) return;

    player.classList.add('is-timeline-transitioning');
    player.classList.toggle('is-timeline-expanded', shouldExpandTimeline);
    if (timelineTransitionTimer !== null) window.clearTimeout(timelineTransitionTimer);
    timelineTransitionTimer = window.setTimeout(() => {
      player.classList.remove('is-timeline-transitioning');
      timelineTransitionTimer = null;
    }, TIMELINE_EXPANSION_TRANSITION_MS);
  }

  /** Updates the visual playhead and corrects meaningful media drift. */
  function updateProgress() {
    const playbackTime = getPlaybackTime();
    renderTimelineWindow(playbackTime);
    updateWaveformProgress(playbackTime / getVisibleTimelineDuration());
    synchronizeEnhancedAudio();
    renderCaptions();
    renderStatusCopy();
  }

  /** Ends the fixed comparison window without modifying the source video. */
  function completePlaybackWindow() {
    stopAutomaticSwitching();
    pauseMedia();
    clearBrandOutro();
    speakerContinuationActive = false;
    video.currentTime = 0;
    speakerAudioElements.forEach((audio) => { audio.currentTime = 0; });
    if (continueSpeakerAudioAfterVideo) setMode('original', 'automatic');
    updateProgress();
  }

  /**
   * Hands the shared playhead to the longer speaker tracks while the video
   * remains frozen on the configured final relevant frame.
   */
  async function beginSpeakerContinuation() {
    if (!continueSpeakerAudioAfterVideo || speakerContinuationActive) return;
    stopAutomaticSwitching();
    clearOriginalCutoffTimer();
    clearSpeakerStartTimer();
    // Mute before seeking or changing modes so no event handler can expose the
    // irrelevant source tail during the ownership handoff.
    video.muted = true;
    synchronizeEnhancedAudio();
    speakerContinuationActive = true;
    if (originalEndSeconds !== null) video.currentTime = originalEndSeconds;
    video.pause();
    setMode('enhanced', 'automatic');
    player.classList.add('is-playing');
    try {
      await Promise.all(speakerAudioElements
        .filter((audio) => !audio.ended)
        .map((audio) => audio.play()));
    } catch (error) {
      pauseMedia();
      console.error('Speaker continuation could not start.', error);
    }
    updateProgress();
  }

  /**
   * Seeks both sources from the shared waveform coordinate system.
   * @param {PointerEvent} event
   */
  function seekFromWaveform(event) {
    if (!Number.isFinite(video.duration)) return;
    clearBrandOutro();
    const bounds = (waveformTimeTrack ?? referenceWaveform).getBoundingClientRect();
    const ratio = clampRatio((event.clientX - bounds.left) / bounds.width);
    const targetTime = ratio * getVisibleTimelineDuration();
    const wasPlaying = !isPlaybackPaused();
    const originalEndTime = getOriginalEndTime();
    const targetsSpeakerContinuation = continueSpeakerAudioAfterVideo && targetTime >= originalEndTime;
    speakerContinuationActive = targetsSpeakerContinuation;
    if (targetsSpeakerContinuation) {
      video.muted = true;
      video.pause();
    }
    video.currentTime = Math.min(targetTime, originalEndTime);
    speakerAudioElements.forEach((audio) => {
      if (Number.isFinite(audio.duration)) {
        audio.currentTime = Math.min(
          Math.max(0, targetTime - speakerOffsetSeconds),
          audio.duration,
        );
      }
    });
    if (targetsSpeakerContinuation) setMode('enhanced', 'automatic');
    else synchronizeEnhancedAudio();
    if (wasPlaying) void playMedia();
    updateProgress();
  }

  /**
   * Maps upward/leftward wheel intent to Reconstructed and the opposite intent to Original.
   * @param {WheelEvent} event
   */
  function switchModeFromWheel(event) {
    const dominantDelta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
    if (dominantDelta === 0) return;
    event.preventDefault();
    setMode(dominantDelta > 0 ? 'original' : 'enhanced', 'user');
  }

  // Event binding and startup

  /** Connects playback, seeking, and lifecycle events. */
  function bindMediaEvents() {
    playButton.addEventListener('click', togglePlayback);
    video.addEventListener('click', togglePlayback);
    document.addEventListener('keydown', togglePlaybackFromKeyboard);
    document.addEventListener('keydown', switchModeFromKeyboard);
    window.addEventListener('resize', () => requestAnimationFrame(updateProgress));
    video.addEventListener('loadedmetadata', updateProgress);
    speakerAudioElements.forEach((audio) => audio.addEventListener('loadedmetadata', synchronizeEnhancedAudio));
    video.addEventListener('timeupdate', () => {
      if (speakerOffsetSeconds > 0
        && !speakerContinuationActive
        && video.currentTime >= speakerOffsetSeconds
        && speakerAudioElements.some((audio) => audio.paused && !audio.ended)) {
        void startSynchronizedSpeakerAudio();
      }
      if (originalEndSeconds !== null
        && !speakerContinuationActive
        && video.currentTime >= originalEndSeconds) {
        video.muted = true;
        void beginSpeakerContinuation();
        return;
      }
      if (configuredPlaybackDuration !== null && video.currentTime >= configuredPlaybackDuration) {
        completePlaybackWindow();
        return;
      }
      updateProgress();
    });
    video.addEventListener('seeking', () => {
      clearOriginalCutoffTimer();
      clearSpeakerStartTimer();
    });
    video.addEventListener('seeked', () => {
      synchronizeEnhancedAudio();
      renderMode();
      scheduleOriginalCutoff();
      scheduleSpeakerStart();
    });
    video.addEventListener('play', () => {
      player.classList.add('is-playing');
      startAutomaticSwitching();
      scheduleOriginalCutoff();
      scheduleSpeakerStart();
    });
    video.addEventListener('pause', () => {
      clearOriginalCutoffTimer();
      clearSpeakerStartTimer();
      if (continueSpeakerAudioAfterVideo && (video.ended || speakerContinuationActive)) return;
      player.classList.remove('is-playing');
      clearAutomaticSwitchTimer();
      speakerAudioElements.forEach((audio) => audio.pause());
    });
    video.addEventListener('ended', () => {
      if (continueSpeakerAudioAfterVideo) void beginSpeakerContinuation();
      else completePlaybackWindow();
    });
    video.addEventListener('ratechange', () => {
      scheduleOriginalCutoff();
      scheduleSpeakerStart();
    });
    video.addEventListener('volumechange', () => {
      if (hasOriginalExpired() && !video.muted) video.muted = true;
    });
    speakerAudioElements.forEach((audio) => {
      audio.addEventListener('timeupdate', () => {
        if (speakerContinuationActive) updateProgress();
      });
      audio.addEventListener('play', () => {
        if (speakerContinuationActive) player.classList.add('is-playing');
      });
      audio.addEventListener('pause', () => {
        if (speakerContinuationActive && isPlaybackPaused()) player.classList.remove('is-playing');
      });
      audio.addEventListener('ended', () => {
        if (speakerContinuationActive && speakerAudioElements.every((speakerAudio) => speakerAudio.ended)) {
          if (brandOutro) beginBrandOutro();
          else completePlaybackWindow();
        }
      });
    });
    brandOutro?.addEventListener('click', togglePlayback);
    waveformDock.addEventListener('pointerdown', (event) => {
      if (event.target instanceof Element && event.target.closest('.track-mute')) return;
      waveformDock.setPointerCapture(event.pointerId);
      seekFromWaveform(event);
    });
    waveformDock.addEventListener('pointermove', (event) => {
      if (event.buttons === 1 && waveformDock.hasPointerCapture(event.pointerId)) seekFromWaveform(event);
    });
  }

  /** Connects hover, activation, and wheel intent to the shared mode transition. */
  function bindModeEvents() {
    wheelSwitchTargets.forEach((target) => {
      target.addEventListener('wheel', switchModeFromWheel, { passive: false });
    });
    modeButtons.forEach((button) => {
      const mode = /** @type {AudioMode} */ (button.dataset.audioMode);
      button.addEventListener('pointerenter', (event) => {
        if (event.pointerType === 'mouse') setMode(mode, 'user');
      });
      button.addEventListener('click', () => setMode(mode, 'user'));
    });
    muteButtons.forEach((button) => {
      const track = /** @type {SpeakerTrack} */ (button.dataset.muteTrack);
      button.addEventListener('pointerdown', (event) => event.stopPropagation());
      button.addEventListener('click', (event) => {
        event.stopPropagation();
        if (activeMode !== 'enhanced') return;
        toggleSpeakerMute(track);
      });
    });
  }

  renderWaveforms();
  renderWaveformTimeScale();
  bindMediaEvents();
  bindModeEvents();
  renderMode();
  void preloadMedia();
})();
