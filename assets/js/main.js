/**
 * Custom waveform players, synchronized word captions, and language UI.
 * Usage: add a .wave-player with data-waveform and an embedded audio node.
 * Load after the embedded alignment and transcript data at the end of /index.html.
 * Components are configured through their existing HTML classes and data attributes.
 */
(() => {
  'use strict';

  // Configuration and contracts

  const DEFAULT_AUDIO_DURATION_SECONDS = 15.092;
  const KEYBOARD_SEEK_STEP_SECONDS = 2;
  const RESIZE_DEBOUNCE_MS = 120;
  const WAVEFORM_DETAILED_PEAK_COUNT = 320;
  const WAVEFORM_MAXIMUM_BAR_COUNT = 320;
  const WAVEFORM_MINIMUM_BAR_COUNT = 72;
  const TRACK_SYNC_TOLERANCE_SECONDS = 0.08;
  /** @typedef {{w: string, s: number, e: number}} AlignedWord */
  /** @typedef {{s: number, e: number, text: string}} Utterance */
  /** @typedef {'speaker0' | 'speaker1'} SpeakerTrack */

  /**
   * Enforces required page-level DOM contracts during startup.
   * @template {Element} T
   * @param {ParentNode} root
   * @param {string} selector
   * @param {{new (...args: any[]): T}} elementType
   * @returns {T}
   */
  function requireElement(root, selector, elementType) {
    const element = root.querySelector(selector);
    if (!(element instanceof elementType)) throw new Error(`Required element is missing: ${selector}`);
    return element;
  }

  /** @param {string} selector @returns {Record<string, unknown[]>} */
  function readEmbeddedJson(selector) {
    const dataElement = requireElement(document, selector, HTMLScriptElement);
    const payload = JSON.parse(dataElement.textContent || 'null');
    if (!isRecord(payload)) throw new TypeError(`Embedded data must be a JSON object: ${selector}`);
    return payload;
  }

  // Page state

  const languageToggle = requireElement(document, '.language-toggle', HTMLButtonElement);
  const audioElements = /** @type {NodeListOf<HTMLAudioElement>} */ (document.querySelectorAll('audio'));
  const players = /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll('.wave-player'));
  const dualPlayerElements = /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll('.dual-track-player'));
  /** @type {DualTrackPlayer[]} */
  const dualPlayerControllers = [];
  const alignmentData = /** @type {Record<string, AlignedWord[]>} */ (readEmbeddedJson('#alignment-data'));
  const transcriptData = /** @type {Record<string, Utterance[]>} */ (readEmbeddedJson('#transcript-data'));
  /** @type {Record<string, Utterance[]>} */
  const regenerationTranscriptData = {};
  /** @type {Record<string, AlignedWord[]>} */
  const regenerationAlignmentData = {};
  /** @type {Map<string, Promise<AudioBuffer>>} */
  const waveformBufferCache = new Map();
  /** @type {WeakMap<HTMLElement, {detailedPeaks: number[], maximumHeight: number, barCount: number}>} */
  const waveformRenderState = new WeakMap();
  /** @type {number | null} */
  let waveformResizeTimer = null;
  /** @type {AudioContext | null} */
  let waveformAudioContext = null;

  // Caption data

  /** @param {unknown} value @returns {value is Record<string, unknown>} */
  function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  /**
   * Loads JSON through HTTP so the rendered captions always reflect the source files.
   * Opening index.html through file:// is intentionally unsupported because browsers block fetch there.
   * @param {string} url
   * @returns {Promise<Record<string, unknown>>}
   */
  async function fetchJsonObject(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Failed to load ${url}: HTTP ${response.status}`);
    const payload = await response.json();
    if (!isRecord(payload)) throw new TypeError(`Expected a JSON object from ${url}`);
    return payload;
  }

  /**
   * Converts the persisted transcript contract into the compact structure used by the renderer.
   * @param {Record<string, unknown>} payload
   * @param {string} prefix
   * @returns {Record<string, Utterance[]>}
   */
  function parseTranscriptPayload(payload, prefix) {
    return Object.fromEntries(['s0', 's1'].map((speakerKey) => {
      const utterances = payload[speakerKey];
      if (!Array.isArray(utterances)) throw new TypeError(`Transcript is missing array ${speakerKey}`);
      return [`${prefix}-${speakerKey}`, utterances.map((utterance) => {
        if (!isRecord(utterance)
          || typeof utterance.t0 !== 'number'
          || typeof utterance.t1 !== 'number'
          || typeof utterance.text !== 'string') {
          throw new TypeError(`Invalid transcript entry in ${speakerKey}`);
        }
        return { s: utterance.t0, e: utterance.t1, text: utterance.text };
      })];
    }));
  }

  /**
   * Converts the persisted word-alignment contract without changing its timing.
   * @param {Record<string, unknown>} payload
   * @param {string} prefix
   * @returns {Record<string, AlignedWord[]>}
   */
  function parseAlignmentPayload(payload, prefix) {
    return Object.fromEntries(['s0', 's1'].map((speakerKey) => {
      const words = payload[speakerKey];
      if (!Array.isArray(words)) throw new TypeError(`Alignment is missing array ${speakerKey}`);
      return [`${prefix}-${speakerKey}`, words.map((word) => {
        if (!isRecord(word)
          || typeof word.t0 !== 'number'
          || typeof word.t1 !== 'number'
          || typeof word.word !== 'string') {
          throw new TypeError(`Invalid alignment entry in ${speakerKey}`);
        }
        return { w: word.word, s: word.t0, e: word.t1 };
      })];
    }));
  }

  /** @param {HTMLElement} player @returns {Promise<void>} */
  async function loadExternalCaptionData(player) {
    const transcriptUrl = player.dataset.transcriptUrl;
    const alignmentUrl = player.dataset.alignmentUrl;
    if (!transcriptUrl && !alignmentUrl) return;
    if (!transcriptUrl || !alignmentUrl || !player.dataset.captionPrefix) {
      throw new Error('External captions require transcript, alignment, and prefix attributes');
    }
    const [transcriptPayload, alignmentPayload] = await Promise.all([
      fetchJsonObject(transcriptUrl),
      fetchJsonObject(alignmentUrl),
    ]);
    Object.assign(regenerationTranscriptData, parseTranscriptPayload(transcriptPayload, player.dataset.captionPrefix));
    Object.assign(regenerationAlignmentData, parseAlignmentPayload(alignmentPayload, player.dataset.captionPrefix));
  }

  // Waveform rendering

  /** @returns {AudioContext} */
  function getWaveformAudioContext() {
    if (waveformAudioContext) return waveformAudioContext;
    const AudioContextType = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextType) throw new Error('This browser does not support Web Audio decoding');
    waveformAudioContext = new AudioContextType();
    return waveformAudioContext;
  }

  /**
   * Downloads and decodes each unique URL once, even when several players share the same source.
   * @param {HTMLAudioElement} audio
   * @returns {Promise<AudioBuffer>}
   */
  function decodeAudio(audio) {
    const sourceUrl = audio.currentSrc || audio.src;
    if (!sourceUrl) return Promise.reject(new Error('Audio element has no source URL'));
    if (!waveformBufferCache.has(sourceUrl)) {
      const pendingBuffer = fetch(sourceUrl)
        .then((response) => {
          if (!response.ok) throw new Error(`Failed to load waveform audio: HTTP ${response.status}`);
          return response.arrayBuffer();
        })
        .then((encodedAudio) => getWaveformAudioContext().decodeAudioData(encodedAudio));
      waveformBufferCache.set(sourceUrl, pendingBuffer);
    }
    return waveformBufferCache.get(sourceUrl);
  }

  /**
   * Combines peak amplitude with RMS energy so transients remain visible without overstating silence.
   * @param {AudioBuffer} audioBuffer
   * @param {number} barCount
   * @returns {number[]}
   */
  function calculateWaveformPeaks(audioBuffer, barCount) {
    const sampleCount = audioBuffer.length;
    const rawLevels = Array.from({ length: barCount }, (_, barIndex) => {
      const start = Math.floor((barIndex * sampleCount) / barCount);
      const end = Math.max(start + 1, Math.floor(((barIndex + 1) * sampleCount) / barCount));
      let peak = 0;
      let squareSum = 0;
      let measuredSamples = 0;
      for (let channelIndex = 0; channelIndex < audioBuffer.numberOfChannels; channelIndex += 1) {
        const channel = audioBuffer.getChannelData(channelIndex);
        for (let sampleIndex = start; sampleIndex < Math.min(end, channel.length); sampleIndex += 1) {
          const amplitude = Math.abs(channel[sampleIndex]);
          peak = Math.max(peak, amplitude);
          squareSum += amplitude * amplitude;
          measuredSamples += 1;
        }
      }
      const rms = measuredSamples > 0 ? Math.sqrt(squareSum / measuredSamples) : 0;
      return (0.65 * peak) + (0.35 * rms);
    });
    const maximumLevel = Math.max(...rawLevels);
    if (maximumLevel <= 0) return rawLevels;
    return rawLevels.map((level) => Math.pow(level / maximumLevel, 0.72));
  }

  /** @param {HTMLElement} waveform @param {number[]} peaks @param {number} maximumHeight */
  function renderWaveformBars(waveform, peaks, maximumHeight) {
    const fragment = document.createDocumentFragment();
    peaks.forEach((peak) => {
      const bar = document.createElement('span');
      bar.style.height = `${Math.max(2, Math.round(peak * maximumHeight))}px`;
      fragment.appendChild(bar);
    });
    waveform.replaceChildren(fragment);
    waveform.removeAttribute('aria-busy');
  }

  /** @param {HTMLElement} waveform @returns {number} */
  function getWaveformBarCount(waveform) {
    const availableWidth = Math.max(1, waveform.clientWidth);
    return Math.max(
      WAVEFORM_MINIMUM_BAR_COUNT,
      Math.min(WAVEFORM_MAXIMUM_BAR_COUNT, Math.round(availableWidth / 1.75)),
    );
  }

  /** @param {number[]} detailedPeaks @param {number} barCount @returns {number[]} */
  function resampleWaveformPeaks(detailedPeaks, barCount) {
    if (barCount >= detailedPeaks.length) return detailedPeaks.slice();
    return Array.from({ length: barCount }, (_, barIndex) => {
      const start = Math.floor((barIndex * detailedPeaks.length) / barCount);
      const end = Math.max(start + 1, Math.floor(((barIndex + 1) * detailedPeaks.length) / barCount));
      let peak = 0;
      let sum = 0;
      for (let index = start; index < end; index += 1) {
        peak = Math.max(peak, detailedPeaks[index]);
        sum += detailedPeaks[index];
      }
      return (0.65 * peak) + (0.35 * (sum / (end - start)));
    });
  }

  /** Reuses the decoded buffer while adapting detail to the current rendered width. @param {HTMLElement} waveform */
  function renderResponsiveWaveform(waveform) {
    const state = waveformRenderState.get(waveform);
    if (!state) return;
    const barCount = getWaveformBarCount(waveform);
    if (state.barCount === barCount) return;
    state.barCount = barCount;
    renderWaveformBars(waveform, resampleWaveformPeaks(state.detailedPeaks, barCount), state.maximumHeight);
  }

  function refreshResponsiveWaveforms() {
    document.querySelectorAll('.waveform, .dual-waveform').forEach(renderResponsiveWaveform);
  }

  /** @param {HTMLAudioElement} audio @param {HTMLElement} waveform @param {number} maximumHeight */
  async function loadWaveform(audio, waveform, maximumHeight) {
    waveform.setAttribute('aria-busy', 'true');
    try {
      const audioBuffer = await decodeAudio(audio);
      waveformRenderState.set(waveform, {
        detailedPeaks: calculateWaveformPeaks(audioBuffer, WAVEFORM_DETAILED_PEAK_COUNT),
        maximumHeight,
        barCount: 0,
      });
      renderResponsiveWaveform(waveform);
    } catch (error) {
      console.error(error);
      waveform.removeAttribute('aria-busy');
      waveform.setAttribute('aria-label', 'Waveform failed to load');
    }
  }

  // Language and captions

  /** @returns {'en' | 'zh-CN'} */
  function readLanguage() {
    try { return localStorage.getItem('avera-language') === 'zh-CN' ? 'zh-CN' : 'en'; }
    catch { return 'en'; }
  }

  /** @param {'en' | 'zh-CN'} language */
  function applyLanguage(language) {
    const isChinese = language === 'zh-CN';
    document.documentElement.lang = language;
    document.title = 'Conversational Voice';
    languageToggle.setAttribute('aria-label', isChinese ? 'Switch to English' : '切换为中文');
    updatePlayerLabels();
    requestAnimationFrame(alignComparisonRows);
    try { localStorage.setItem('avera-language', language); } catch { /* file:// storage may be unavailable */ }
  }

  /** @param {number} seconds @returns {string} */
  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return '0:00';
    return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
  }

  /**
   * Returns the drawable width inside a waveform's horizontal padding so the bars, playhead,
   * and pointer seeking all use the same coordinate system.
   * @param {HTMLElement} waveform
   * @returns {{left: number, width: number}}
   */
  function getWaveformContentBox(waveform) {
    const style = window.getComputedStyle(waveform);
    const left = Number.parseFloat(style.paddingLeft) || 0;
    const right = Number.parseFloat(style.paddingRight) || 0;
    return { left, width: Math.max(0, waveform.clientWidth - left - right) };
  }

  /** @param {HTMLElement} waveform @param {number} ratio */
  function updateWaveformPlayhead(waveform, ratio) {
    const contentBox = getWaveformContentBox(waveform);
    const clampedRatio = Math.min(1, Math.max(0, ratio));
    waveform.style.setProperty('--playhead-position', `${contentBox.left + contentBox.width * clampedRatio}px`);
  }

  /** @param {HTMLElement} waveform @param {number} clientX @returns {number} */
  function getWaveformPointerRatio(waveform, clientX) {
    const bounds = waveform.getBoundingClientRect();
    const contentBox = getWaveformContentBox(waveform);
    if (contentBox.width <= 0) return 0;
    const contentLeft = bounds.left + waveform.clientLeft + contentBox.left;
    return Math.min(1, Math.max(0, (clientX - contentLeft) / contentBox.width));
  }

  /** @param {HTMLElement} player */
  function renderCaption(player) {
    const audio = player.querySelector('audio');
    const panel = player.querySelector('.live-caption');
    const isChinese = document.documentElement.lang === 'zh-CN';
    if (player.dataset.captionSet) {
      const fragment = document.createDocumentFragment();
      const captionSets = player.dataset.captionSet === 'regen-mix'
        ? ['regen-s0', 'regen-s1']
        : [player.dataset.captionSet];
      const firstStart = Math.min(...captionSets
        .map((captionSet) => regenerationTranscriptData[captionSet]?.[0]?.s)
        .filter(Number.isFinite));
      const captionTime = audio.paused && audio.currentTime === 0 && Number.isFinite(firstStart)
        ? firstStart
        : audio.currentTime;
      captionSets.forEach((captionSet) => {
        const activeUtterance = (regenerationTranscriptData[captionSet] || [])
          .find((utterance) => captionTime >= utterance.s && captionTime <= utterance.e);
        if (!activeUtterance) return;
        const speakerNumber = captionSet.endsWith('s0') ? '0' : '1';
        const speaker = isChinese ? `说话人 ${speakerNumber}` : `Speaker ${speakerNumber}`;
        const sentenceWords = (regenerationAlignmentData[captionSet] || [])
          .filter((word) => word.e >= activeUtterance.s && word.s <= activeUtterance.e);
        const row = document.createElement('div');
        row.className = 'caption-row';
        row.setAttribute('aria-label', `${speaker}: ${activeUtterance.text}`);
        const label = document.createElement('span');
        label.className = 'caption-speaker';
        label.textContent = speaker;
        row.appendChild(label);
        sentenceWords.forEach((word) => {
          const token = document.createElement('span');
          token.className = 'caption-word';
          if (!audio.paused && captionTime >= word.s && captionTime <= word.e) token.classList.add('active');
          token.textContent = word.w;
          row.appendChild(token);
        });
        fragment.appendChild(row);
      });
      panel.replaceChildren(fragment);
      return;
    }
    if (!player.dataset.speakers) return;
    const fragment = document.createDocumentFragment();
    player.dataset.speakers.split(',').forEach((speakerId) => {
      const utterances = transcriptData[speakerId] || [];
      const activeUtterance = utterances.find((utterance) => audio.currentTime >= utterance.s && audio.currentTime <= utterance.e);
      if (!activeUtterance) return;

      /** @type {AlignedWord[]} */
      const words = alignmentData[speakerId] || [];
      const sentenceWords = words.filter((word) => word.e >= activeUtterance.s && word.s <= activeUtterance.e);
      const row = document.createElement('div');
      row.className = 'caption-row';
      const speaker = isChinese ? `说话人 ${speakerId}` : `Speaker ${speakerId}`;
      row.setAttribute('aria-label', `${speaker}: ${activeUtterance.text}`);
      const label = document.createElement('span');
      label.className = 'caption-speaker';
      label.textContent = speaker;
      row.appendChild(label);
      sentenceWords.forEach((word) => {
        const token = document.createElement('span');
        token.className = 'caption-word';
        if (!audio.paused && audio.currentTime >= word.s && audio.currentTime <= word.e) token.classList.add('active');
        token.textContent = word.w;
        row.appendChild(token);
      });
      fragment.appendChild(row);
    });
    panel.replaceChildren(fragment);
  }

  /** @param {HTMLElement} player */
  function updatePlayer(player) {
    const audio = requireElement(player, 'audio', HTMLAudioElement);
    const bars = player.querySelectorAll('.waveform span');
    const duration = Number.isFinite(audio.duration) ? audio.duration : DEFAULT_AUDIO_DURATION_SECONDS;
    const ratio = duration > 0 ? audio.currentTime / duration : 0;
    bars.forEach((bar, index) => bar.classList.toggle('played', (index + 0.5) / bars.length <= ratio));
    player.querySelector('.wave-time').textContent = `${formatTime(audio.currentTime)} / ${formatTime(duration)}`;
    const waveform = player.querySelector('.waveform');
    waveform.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
    waveform.setAttribute('aria-valuetext', `${formatTime(audio.currentTime)} / ${formatTime(duration)}`);
    updateWaveformPlayhead(waveform, ratio);
    waveform.classList.toggle('is-playing', !audio.paused);
    renderCaption(player);
  }

  // Shared playback coordination

  /** Pauses every other example so one obvious playback action owns the page audio. @param {Iterable<HTMLAudioElement>} activeAudios */
  function pauseAllExcept(activeAudios) {
    const activeSet = new Set(activeAudios);
    audioElements.forEach((audio) => { if (!activeSet.has(audio)) audio.pause(); });
  }

  // Dual-track player

  /**
   * Owns all state and event handling for one two-speaker player.
   * The surrounding HTML is the component contract; missing controls fail loudly during setup.
   */
  class DualTrackPlayer {
    /** @param {HTMLElement} element */
    constructor(element) {
      this.element = element;
      this.audios = {
        speaker0: this.requireElement('[data-track-audio="speaker0"]', HTMLAudioElement),
        speaker1: this.requireElement('[data-track-audio="speaker1"]', HTMLAudioElement),
      };
      this.audioList = Object.values(this.audios);
      this.masterAudio = this.audios.speaker0;
      this.bothButton = this.requireElement('.dual-play-both', HTMLButtonElement);
      this.bothIcon = this.requireElement('.dual-play-icon', HTMLElement);
      this.bothLabel = this.requireElement('.dual-play-label', HTMLElement);
      this.timeElement = this.requireElement('.dual-time', HTMLElement);
      this.waveforms = Array.from(element.querySelectorAll('.dual-waveform'));
      this.muteControls = Array.from(element.querySelectorAll('[data-track-mute]')).map((button) => {
        const track = button.dataset.trackMute;
        if (track !== 'speaker0' && track !== 'speaker1') throw new Error(`Invalid mute button track: ${String(track)}`);
        return {
          button,
          track,
          icon: this.requireElement('.track-mute-icon', HTMLElement, button),
          label: this.requireElement('.track-mute-label', HTMLElement, button),
          row: this.requireElement(`[data-track="${track}"]`, HTMLElement),
        };
      });
      this.captionPanel = element.querySelector('.dual-caption');
      this.captionSource = element.dataset.captionSource || null;
      this.captionPrefix = element.dataset.captionPrefix || null;
      const reconstructionEnd = element.dataset.reconstructionEnd;
      /** @type {number | null} */
      this.reconstructionEndSeconds = reconstructionEnd === undefined ? null : Number(reconstructionEnd);
      if (this.captionPanel && this.captionSource !== 'separation' && this.captionSource !== 'regeneration') {
        throw new Error(`Invalid caption source: ${String(this.captionSource)}`);
      }
      if (this.captionSource === 'regeneration' && !this.captionPrefix) {
        throw new Error('Expansion captions require data-caption-prefix');
      }
      if (this.reconstructionEndSeconds !== null
        && (!Number.isFinite(this.reconstructionEndSeconds) || this.reconstructionEndSeconds <= 0)) {
        throw new Error('Reconstruction boundary must be a positive number of seconds');
      }
      if (this.waveforms.length !== 2 || this.muteControls.length !== 2) {
        throw new Error('Dual-track player requires exactly two waveforms and two mute controls');
      }
      this.waveformReady = this.initializeWaveforms();
      this.bindEvents();
      this.update();
    }

    /** @returns {boolean} */
    get isPlaying() {
      return !this.masterAudio.paused && !this.masterAudio.ended;
    }

    /**
     * Enforces the component's required DOM contract and provides a useful setup error.
     * @template {Element} T
     * @param {string} selector
     * @param {{new (...args: any[]): T}} elementType
     * @param {ParentNode} [root]
     * @returns {T}
     */
    requireElement(selector, elementType, root = this.element) {
      const element = root.querySelector(selector);
      if (!(element instanceof elementType)) throw new Error(`Dual-track player is missing ${selector}`);
      return element;
    }

    /** @returns {Promise<void>} */
    async initializeWaveforms() {
      const waveformLoads = this.waveforms.map((waveform) => {
        const trackMode = waveform.closest('.dual-track')?.dataset.track;
        if (trackMode !== 'speaker0' && trackMode !== 'speaker1') {
          throw new Error(`Invalid waveform track: ${String(trackMode)}`);
        }
        return loadWaveform(this.audios[trackMode], waveform, 52);
      });
      await Promise.all(waveformLoads);
    }

    bindEvents() {
      this.bothButton.addEventListener('click', () => { void this.togglePlayback(); });
      this.muteControls.forEach(({ button, track }) => {
        button.addEventListener('click', () => this.toggleMute(track));
      });
      this.waveforms.forEach((waveform) => {
        waveform.addEventListener('pointerdown', (event) => this.seekFromPointer(waveform, event));
        waveform.addEventListener('keydown', (event) => this.seekFromKeyboard(event));
      });
      this.audioList.forEach((audio) => {
        ['loadedmetadata', 'timeupdate', 'seeked', 'play', 'pause', 'ended'].forEach((eventName) => {
          audio.addEventListener(eventName, () => this.handleAudioEvent(audio, eventName));
        });
      });
    }

    /** Starts both speaker files from the same position or pauses them as one transport. @returns {Promise<void>} */
    async togglePlayback() {
      if (this.isPlaying) {
        this.audioList.forEach((audio) => audio.pause());
        return;
      }

      const duration = this.getDuration();
      if (this.masterAudio.ended || this.masterAudio.currentTime >= duration) this.seek(0);
      else this.seek(this.masterAudio.currentTime);
      pauseAllExcept(this.audioList);
      try {
        // Calling play() for both elements in the same user gesture preserves autoplay compatibility.
        await Promise.all(this.audioList.map((audio) => audio.play()));
        this.synchronizeTracks();
      } catch (error) {
        this.audioList.forEach((audio) => audio.pause());
        console.error('Unable to start synchronized speaker tracks.', error);
      }
      this.update();
    }

    /** Muting changes only audibility; both files continue on the shared timeline. @param {SpeakerTrack} track */
    toggleMute(track) {
      this.audios[track].muted = !this.audios[track].muted;
      this.update();
    }

    /** @returns {number} */
    getDuration() {
      return Number.isFinite(this.masterAudio.duration) ? this.masterAudio.duration : DEFAULT_AUDIO_DURATION_SECONDS;
    }

    /** Corrects meaningful native-media drift without continuously disturbing playback. */
    synchronizeTracks() {
      const followerAudio = this.audios.speaker1;
      if (Math.abs(followerAudio.currentTime - this.masterAudio.currentTime) > TRACK_SYNC_TOLERANCE_SECONDS) {
        followerAudio.currentTime = this.masterAudio.currentTime;
      }
    }

    /** Keeps both files aligned when seeking from either waveform. @param {number} seconds */
    seek(seconds) {
      this.audioList.forEach((audio) => {
        const duration = Number.isFinite(audio.duration) ? audio.duration : seconds;
        audio.currentTime = Math.min(duration, Math.max(0, seconds));
      });
    }

    /**
     * Seeks on pointerdown because synthetic click events may omit usable coordinates.
     * @param {HTMLElement} waveform
     * @param {PointerEvent} event
     */
    seekFromPointer(waveform, event) {
      const ratio = getWaveformPointerRatio(waveform, event.clientX);
      this.seek(ratio * this.getDuration());
      this.update();
    }

    /** @param {KeyboardEvent} event */
    seekFromKeyboard(event) {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      this.seek(this.masterAudio.currentTime + (event.key === 'ArrowRight' ? KEYBOARD_SEEK_STEP_SECONDS : -KEYBOARD_SEEK_STEP_SECONDS));
      this.update();
    }

    /** @param {HTMLAudioElement} audio @param {string} eventName */
    handleAudioEvent(audio, eventName) {
      if (audio === this.masterAudio && eventName === 'timeupdate' && this.isPlaying) this.synchronizeTracks();
      if (audio === this.masterAudio && eventName === 'ended') {
        this.audioList.forEach((trackAudio) => trackAudio.pause());
        this.seek(0);
      }
      this.update();
    }

    /** @param {'speaker0' | 'speaker1'} speakerMode @returns {{speakerId: string, transcripts: Utterance[], alignments: AlignedWord[]}} */
    getCaptionData(speakerMode) {
      const speakerId = speakerMode === 'speaker0' ? '0' : '1';
      const isRegeneration = this.captionSource === 'regeneration';
      const dataKey = isRegeneration ? `${this.captionPrefix}-s${speakerId}` : speakerId;
      return {
        speakerId,
        transcripts: (isRegeneration ? regenerationTranscriptData[dataKey] : transcriptData[dataKey]) || [],
        alignments: (isRegeneration ? regenerationAlignmentData[dataKey] : alignmentData[dataKey]) || [],
      };
    }

    renderCaption() {
      if (!this.captionPanel) return;
      if (this.element.dataset.captionLoadError) {
        this.captionPanel.textContent = this.element.dataset.captionLoadError;
        return;
      }
      /** @type {SpeakerTrack[]} */
      const requestedSpeakers = ['speaker0', 'speaker1'];
      const captionData = requestedSpeakers.map((speakerMode) => this.getCaptionData(speakerMode));
      const firstCaptionStart = Math.min(...captionData
        .map(({ transcripts }) => transcripts[0]?.s)
        .filter(Number.isFinite));
      const captionTime = !this.isPlaying && this.masterAudio.currentTime === 0 && Number.isFinite(firstCaptionStart)
        ? firstCaptionStart
        : this.masterAudio.currentTime;
      const fragment = document.createDocumentFragment();
      const isChinese = document.documentElement.lang === 'zh-CN';

      captionData.forEach(({ speakerId, transcripts, alignments }) => {
        const activeUtterance = transcripts.find(({ s, e }) => captionTime >= s && captionTime <= e);
        if (!activeUtterance) return;
        const speaker = isChinese ? `说话人 ${speakerId}` : `Speaker ${speakerId}`;
        const row = document.createElement('div');
        row.className = 'caption-row';
        row.setAttribute('aria-label', `${speaker}: ${activeUtterance.text}`);
        const label = document.createElement('span');
        label.className = 'caption-speaker';
        label.textContent = speaker;
        row.appendChild(label);
        alignments
          .filter(({ s, e }) => e >= activeUtterance.s && s <= activeUtterance.e)
          .forEach((word) => {
            const token = document.createElement('span');
            token.className = 'caption-word';
            if (this.isPlaying && captionTime >= word.s && captionTime <= word.e) token.classList.add('active');
            token.textContent = word.w;
            row.appendChild(token);
          });
        fragment.appendChild(row);
      });
      this.captionPanel.replaceChildren(fragment);
    }

    update() {
      const duration = this.getDuration();
      const ratio = duration > 0 ? this.masterAudio.currentTime / duration : 0;
      const isChinese = document.documentElement.lang === 'zh-CN';
      const isPlaying = this.isPlaying;
      const phaseLabel = this.reconstructionEndSeconds === null
        ? null
        : (this.masterAudio.currentTime < this.reconstructionEndSeconds
          ? (isChinese ? '当前阶段 · 原始声音重建' : 'Current phase · Original voice reconstruction')
          : (isChinese ? '当前阶段 · 对话扩展' : 'Current phase · Dialogue expansion'));
      this.timeElement.textContent = `${formatTime(this.masterAudio.currentTime)} / ${formatTime(duration)}`;

      this.waveforms.forEach((waveform) => {
        const bars = waveform.querySelectorAll('span');
        bars.forEach((bar, index) => bar.classList.toggle('played', (index + 0.5) / bars.length <= ratio));
        waveform.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
        waveform.setAttribute('aria-valuetext', `${formatTime(this.masterAudio.currentTime)} / ${formatTime(duration)}`);
        const seekLabel = isChinese ? '音频波形，可点击跳转' : 'Audio waveform; click to seek';
        waveform.setAttribute('aria-label', phaseLabel ? `${seekLabel}${isChinese ? '。' : '. '}${phaseLabel}` : seekLabel);
        if (phaseLabel) waveform.dataset.phaseLabel = phaseLabel;
        else delete waveform.dataset.phaseLabel;
        updateWaveformPlayhead(waveform, ratio);
        waveform.classList.toggle('is-playing', isPlaying);
      });

      const bothLabel = isPlaying
        ? (isChinese ? '暂停双轨' : 'Pause both')
        : (isChinese ? '播放双轨' : 'Play both');
      this.bothButton.classList.toggle('is-active', isPlaying);
      this.bothIcon.textContent = isPlaying ? 'Ⅱ' : '▶';
      this.bothLabel.textContent = bothLabel;
      this.bothButton.setAttribute('aria-label', bothLabel);

      this.muteControls.forEach(({ button, track, label, row }) => {
        const speakerNumber = track === 'speaker0' ? '0' : '1';
        const isMuted = this.audios[track].muted;
        const actionLabel = isMuted
          ? (isChinese ? `取消静音说话人 ${speakerNumber}` : `Unmute Speaker ${speakerNumber}`)
          : (isChinese ? `静音说话人 ${speakerNumber}` : `Mute Speaker ${speakerNumber}`);
        button.classList.toggle('is-active', isMuted);
        button.setAttribute('aria-label', actionLabel);
        button.setAttribute('aria-pressed', String(isMuted));
        button.title = actionLabel;
        label.textContent = isMuted
          ? (isChinese ? '已静音' : 'Muted')
          : (isChinese ? '静音' : 'Mute');
        row.classList.toggle('is-muted', isMuted);
      });
      this.renderCaption();
    }
  }

  // Single-track players and responsive layout

  function updatePlayerLabels() {
    const isChinese = document.documentElement.lang === 'zh-CN';
    players.forEach((player) => {
      const audio = player.querySelector('audio');
      const button = player.querySelector('.wave-play');
      const action = audio.paused ? (isChinese ? '播放' : 'Play') : (isChinese ? '暂停' : 'Pause');
      button.setAttribute('aria-label', `${action}: ${isChinese ? audio.dataset.zhLabel : audio.dataset.enLabel}`);
      player.querySelector('.waveform').setAttribute('aria-label', isChinese ? '音频波形，可点击跳转' : 'Audio waveform; click to seek');
    });
    dualPlayerControllers.forEach((controller) => controller.update());
  }

  /** Keeps matching examples in two-column comparisons on the same horizontal baseline. */
  function alignComparisonRows() {
    document.querySelectorAll('#separation .comparison-grid, #regeneration .comparison-grid').forEach((comparison) => {
      const panels = comparison.querySelectorAll('.method-panel');
      const headers = Array.from(panels, (panel) => requireElement(panel, '.method-header', HTMLElement));
      const rows = Array.from(panels, (panel) => Array.from(panel.querySelectorAll('.audio-case')));

      headers.forEach((header) => { header.style.minHeight = ''; });
      rows.flat().forEach((row) => { row.style.minHeight = ''; });
      if (window.matchMedia('(max-width: 760px)').matches) return;
      if (rows.length !== 2) throw new Error('Comparison grids require exactly two method panels.');

      const headerHeight = Math.max(...headers.map((header) => header.getBoundingClientRect().height));
      headers.forEach((header) => { header.style.minHeight = `${Math.ceil(headerHeight)}px`; });
      for (let index = 0; index < Math.min(rows[0].length, rows[1].length); index += 1) {
        const rowHeight = Math.max(rows[0][index].getBoundingClientRect().height, rows[1][index].getBoundingClientRect().height);
        rows[0][index].style.minHeight = `${Math.ceil(rowHeight)}px`;
        rows[1][index].style.minHeight = `${Math.ceil(rowHeight)}px`;
      }
    });
  }

  /** @param {HTMLElement} player @returns {Promise<void>} */
  async function initializeSingleTrackPlayer(player) {
    const audio = requireElement(player, 'audio', HTMLAudioElement);
    const button = requireElement(player, '.wave-play', HTMLButtonElement);
    const waveform = requireElement(player, '.waveform', HTMLElement);
    button.addEventListener('click', () => {
      if (audio.paused) {
        audioElements.forEach((candidate) => { if (candidate !== audio) candidate.pause(); });
        audio.play().catch(() => {});
      } else audio.pause();
    });
    waveform.addEventListener('click', (event) => {
      const ratio = getWaveformPointerRatio(waveform, event.clientX);
      if (Number.isFinite(audio.duration)) audio.currentTime = ratio * audio.duration;
    });
    waveform.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const direction = event.key === 'ArrowRight' ? 1 : -1;
      const duration = audio.duration || DEFAULT_AUDIO_DURATION_SECONDS;
      audio.currentTime = Math.min(duration, Math.max(0, audio.currentTime + direction * KEYBOARD_SEEK_STEP_SECONDS));
    });
    ['loadedmetadata', 'timeupdate', 'seeked', 'play', 'pause', 'ended'].forEach((eventName) => {
      audio.addEventListener(eventName, () => {
        if (eventName === 'ended') audio.currentTime = 0;
        button.textContent = audio.paused ? '▶' : 'Ⅱ';
        updatePlayer(player);
        updatePlayerLabels();
      });
    });
    updatePlayer(player);
    await loadWaveform(audio, waveform, 48);
  }

  // Page initialization

  /** Loads file-backed captions before player construction so the first frame is correct. */
  async function initializePage() {
    await Promise.all(Array.from(dualPlayerElements, async (element) => {
      try {
        await loadExternalCaptionData(element);
      } catch (error) {
        console.error(error);
        element.dataset.captionLoadError = 'Caption JSON failed to load. Serve this page over HTTP and check the JSON paths.';
      }
    }));
    const waveformLoads = Array.from(players, initializeSingleTrackPlayer);
    dualPlayerElements.forEach((element) => dualPlayerControllers.push(new DualTrackPlayer(element)));
    waveformLoads.push(...dualPlayerControllers.map((controller) => controller.waveformReady));
    languageToggle.addEventListener('click', () => applyLanguage(document.documentElement.lang === 'en' ? 'zh-CN' : 'en'));
    window.addEventListener('resize', () => {
      if (waveformResizeTimer !== null) window.clearTimeout(waveformResizeTimer);
      waveformResizeTimer = window.setTimeout(() => requestAnimationFrame(() => {
        alignComparisonRows();
        refreshResponsiveWaveforms();
      }), RESIZE_DEBOUNCE_MS);
    });
    applyLanguage(readLanguage());
    await Promise.all(waveformLoads);
  }

  void initializePage().catch((error) => {
    console.error('Page initialization failed.', error);
  });
})();
