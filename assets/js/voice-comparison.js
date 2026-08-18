/**
 * Interactive original/reconstructed voice comparison controller.
 * Usage: load after a page-specific waveform-peaks.js file on any voice comparison page.
 * Set data-playback-duration on .player only when the comparison uses a fixed playback window.
 */
(() => {
  'use strict';

  // Configuration and contracts

  /** @typedef {'original' | 'enhanced'} AudioMode */
  /** @typedef {'original' | 'speaker0' | 'speaker1'} WaveformTrack */
  /** @typedef {'speaker0' | 'speaker1'} SpeakerTrack */
  /** @typedef {'loading' | 'ready' | 'error'} ResourceState */
  /** @typedef {'automatic' | 'user'} ModeChangeSource */

  const AUTOMATIC_SWITCH_INTERVAL_MS = 3000;
  const MEDIA_SYNC_TOLERANCE_SECONDS = .12;
  const WAVEFORM_CENTER_Y = 15;
  const WAVEFORM_MAXIMUM_HEIGHT = 13;
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
  const video = /** @type {HTMLVideoElement} */ (requireElement(player, 'video'));
  const playButton = /** @type {HTMLButtonElement} */ (requireElement(player, '.big-play'));
  const waveformDock = /** @type {HTMLElement} */ (requireElement(player, '.waveform-dock'));
  const referenceWaveform = /** @type {SVGSVGElement} */ (requireElement(waveformDock, '.dock-wave'));
  const status = /** @type {HTMLElement} */ (requireElement(player, '.mode-status'));
  const statusCopy = /** @type {HTMLElement} */ (requireElement(status, '.mode-copy'));
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
  let automaticSwitchingEnabled = true;

  const waveformPeaks = /** @type {{duration: number, original: number[], speaker0: number[], speaker1: number[]}} */ (
    window.DIALOGUE_WAVEFORM_PEAKS
  );
  if (!waveformPeaks
    || !Number.isFinite(waveformPeaks.duration)
    || waveformPeaks.duration <= 0
    || !SPEAKER_TRACKS.every((track) => Array.isArray(waveformPeaks[track]))
    || !Array.isArray(waveformPeaks.original)) {
    throw new TypeError('Waveform peak data is missing or invalid.');
  }
  if (configuredPlaybackDuration !== null && waveformPeaks.duration !== configuredPlaybackDuration) {
    throw new Error(`Waveform duration must be ${configuredPlaybackDuration}s.`);
  }
  const waveformTracks = /** @type {Record<WaveformTrack, number[]>} */ ({
    original: waveformPeaks.original,
    speaker0: waveformPeaks.speaker0,
    speaker1: waveformPeaks.speaker1,
  });

  /** @param {number} value @returns {number} */
  function clampRatio(value) {
    return Math.min(1, Math.max(0, value));
  }

  /** Uses a configured comparison window, then media metadata, then waveform metadata. @returns {number} */
  function getPlaybackDuration() {
    if (configuredPlaybackDuration !== null) return configuredPlaybackDuration;
    if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
    return waveformPeaks.duration;
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

  /**
   * Maps the shared playhead into a possibly shorter reconstructed source.
   * @param {HTMLAudioElement} audio
   * @returns {number}
   */
  function getEnhancedSyncTime(audio) {
    if (!Number.isFinite(audio.duration)) return 0;
    return Math.min(video.currentTime, audio.duration);
  }

  /** Synchronizes enhanced audio while treating its missing tail as silence. */
  function synchronizeEnhancedAudio() {
    speakerAudioElements.forEach((audio) => {
      if (audio.readyState === HTMLMediaElement.HAVE_NOTHING) return;
      const targetTime = getEnhancedSyncTime(audio);
      if (Math.abs(audio.currentTime - targetTime) > MEDIA_SYNC_TOLERANCE_SECONDS) audio.currentTime = targetTime;
    });
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

  /** Reflects preload readiness without overwriting the selected audio mode. */
  function renderResourceState() {
    const isReady = resourceState === 'ready';
    player.dataset.resourceState = resourceState;
    playButton.disabled = !isReady;
    playButton.textContent = resourceState === 'loading' ? '…' : '▶';
    playButton.setAttribute('aria-label', resourceState === 'loading'
      ? 'Loading demo media'
      : resourceState === 'error' ? 'Demo media failed to load' : 'Play demo');
    statusCopy.textContent = resourceState === 'loading'
      ? 'Loading media…'
      : resourceState === 'error' ? 'Media failed to load' : MODE_LABELS[activeMode];
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
    video.muted = isEnhanced;
    renderSpeakerMuteState();
    renderResourceState();
    requestAnimationFrame(updateProgress);
  }

  // Resource loading

  /**
   * Resolves once a required media element has enough buffered data to start.
   * @param {HTMLMediaElement} media
   * @returns {Promise<void>}
   */
  function waitUntilPlayable(media) {
    if (media.error) return Promise.reject(media.error);
    if (media.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        media.removeEventListener('canplay', handleCanPlay);
        media.removeEventListener('error', handleError);
      };
      const handleCanPlay = () => { cleanup(); resolve(); };
      const handleError = () => { cleanup(); reject(media.error || new Error('Media preload failed.')); };
      media.addEventListener('canplay', handleCanPlay, { once: true });
      media.addEventListener('error', handleError, { once: true });
    });
  }

  /** Preloads every source needed for seamless original/reconstructed switching. */
  async function preloadMedia() {
    const requiredMedia = [video, ...speakerAudioElements];
    requiredMedia.forEach((media) => media.load());
    const readiness = requiredMedia.map(waitUntilPlayable);
    try {
      await Promise.all(readiness);
      resourceState = 'ready';
      renderResourceState();
    } catch (error) {
      resourceState = 'error';
      renderResourceState();
      console.error('Required demo media could not be preloaded.', error);
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
    if (!automaticSwitchingEnabled || automaticSwitchTimer !== null || video.paused) return;
    automaticSwitchTimer = window.setInterval(() => {
      const nextMode = activeMode === 'original' ? 'enhanced' : 'original';
      setMode(nextMode, 'automatic');
    }, AUTOMATIC_SWITCH_INTERVAL_MS);
  }

  /** Starts the video and both speaker sources so mode changes remain immediate. */
  async function playMedia() {
    if (resourceState !== 'ready') return;
    if (video.currentTime >= getPlaybackDuration()) {
      video.currentTime = 0;
      speakerAudioElements.forEach((audio) => { audio.currentTime = 0; });
    }
    synchronizeEnhancedAudio();
    renderMode();
    try {
      const playPromises = [video.play()];
      speakerAudioElements.forEach((audio) => {
        if (!Number.isFinite(audio.duration) || getEnhancedSyncTime(audio) < audio.duration) playPromises.push(audio.play());
      });
      await Promise.all(playPromises);
    } catch (error) {
      console.error('Media playback could not start.', error);
    }
  }

  /** Pauses the video and both reconstructed speaker sources. */
  function pauseMedia() {
    video.pause();
    speakerAudioElements.forEach((audio) => audio.pause());
  }

  /** Toggles all synchronized sources through the video playback state. */
  function togglePlayback() {
    if (resourceState !== 'ready') return;
    if (video.paused) void playMedia();
    else pauseMedia();
  }

  // Progress and seeking

  /** Keeps the playhead aligned to the plotted area rather than its label column. */
  function updateWaveformProgress(ratio) {
    const dockBounds = waveformDock.getBoundingClientRect();
    const waveBounds = referenceWaveform.getBoundingClientRect();
    const left = waveBounds.left - dockBounds.left + clampRatio(ratio) * waveBounds.width;
    waveformDock.style.setProperty('--waveform-progress', `${left.toFixed(2)}px`);
  }

  /** Updates the visual playhead and corrects meaningful media drift. */
  function updateProgress() {
    updateWaveformProgress(video.currentTime / getPlaybackDuration());
    synchronizeEnhancedAudio();
  }

  /** Ends the fixed comparison window without modifying the source video. */
  function completePlaybackWindow() {
    stopAutomaticSwitching();
    pauseMedia();
    video.currentTime = 0;
    speakerAudioElements.forEach((audio) => { audio.currentTime = 0; });
    updateProgress();
  }

  /**
   * Seeks both sources from the shared waveform coordinate system.
   * @param {PointerEvent} event
   */
  function seekFromWaveform(event) {
    if (!Number.isFinite(video.duration)) return;
    const bounds = referenceWaveform.getBoundingClientRect();
    const ratio = clampRatio((event.clientX - bounds.left) / bounds.width);
    const targetTime = ratio * getPlaybackDuration();
    video.currentTime = targetTime;
    synchronizeEnhancedAudio();
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
    video.addEventListener('loadedmetadata', updateProgress);
    speakerAudioElements.forEach((audio) => audio.addEventListener('loadedmetadata', synchronizeEnhancedAudio));
    video.addEventListener('timeupdate', () => {
      if (configuredPlaybackDuration !== null && video.currentTime >= configuredPlaybackDuration) {
        completePlaybackWindow();
        return;
      }
      updateProgress();
    });
    video.addEventListener('seeked', () => {
      synchronizeEnhancedAudio();
      renderMode();
    });
    video.addEventListener('play', () => {
      player.classList.add('is-playing');
      startAutomaticSwitching();
    });
    video.addEventListener('pause', () => {
      player.classList.remove('is-playing');
      clearAutomaticSwitchTimer();
      speakerAudioElements.forEach((audio) => audio.pause());
    });
    video.addEventListener('ended', completePlaybackWindow);
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
  bindMediaEvents();
  bindModeEvents();
  renderMode();
  void preloadMedia();
})();
