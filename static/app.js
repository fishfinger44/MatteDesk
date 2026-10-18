// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
/**
 * MatteDesk - Interactive Frame Player & Roto Canvas
 * Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 6, 12: Stage E1).
 */

class MatteDeskViewer {
  constructor() {
    this.canvas = document.getElementById('mainCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.canvasWrapper = document.getElementById('canvasWrapper');

    // Project state
    this.projectLoaded = false;
    this.activePlate = 'beauty';
    this.firstFrame = 1;
    this.lastFrame = 100;
    this.currentFrame = 1;
    this.fps = 24;
    this.isPlaying = false;
    this.playTimer = null;
    this.lastFrameTimestamp = 0;

    // Viewport transform (Pan & Zoom)
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isPanning = false;
    this.panStartX = 0;
    this.panStartY = 0;
    this.hasInitialFit = false;

    // Comparison of an auxiliary plate with beauty (stage 0): none | ab | wipe | diff | edges
    // ref = right-hand side of the header pair: a plate, or a comp background ('__color' = white / own colour / checker,
    // 'bg' = the target background plate) – then the right side is the cut-out: beauty RGB x object alpha over it
    this.compare = { mode: 'none', wipe: 0.5, holdC: false, gain: 4, ref: 'beauty', bg: { color: '#ffffff', checker: false } };
    try { Object.assign(this.compare.bg, JSON.parse(localStorage.getItem('md_cmp_bg') || '{}') || {}); } catch (e) { /* no storage */ }
    this.alignedPreview = false;   // auxiliary plates: show the proxy warped by the alignment report
    this.edgeCache = new Map();    // beauty edge maps per frame (compare 'edges')

    // Frame cache: Map<string, Image> (keyed by plate[_al]_frameNum)
    this.frameCache = new Map();
    this.loadingFrames = new Set();
    this.cacheLimit = 250;

    // Z8: full-resolution tiles of the plate when the proxy is magnified (zoom > 100 % of the proxy) –
    // GET /api/frame/{f}/tile, the same display conversion as the proxy; the proxy stays underneath until a tile arrives.
    this.tiles = { cache: new Map(), limit: 16, timer: null, region: null };
    this.TILE_MAX = 2048;

    // Active tool

    // Version / stale-server check first: it must work even when a later init step fails
    this.checkServerVersion();

    // Bind DOM elements (each step guarded: one broken step must not leave every button dead without a word)
    for (const [name, fn] of [['bindElements', () => this.bindElements()], ['initEvents', () => this.initEvents()],
                              ['initWebSocket', () => this.initWebSocket()], ['resizeCanvas', () => this.resizeCanvas()]]) {
      try { fn(); } catch (e) { console.error(`MatteDesk init ${name}:`, e); this.showInitError(name, e); }
    }

    // Check environment & current project
    this.checkDoctorStatus();
    this.loadCurrentProject();
    setTimeout(() => this.checkUpdate(), 4000);
  }

  // Newer MatteDesk in the public GitHub releases (at most once a day; engine/updates.py): short list of changes,
  // link to the full changelog; installer copy -> download + start the new setup, git copy -> "git pull"
  async checkUpdate(force = false, pre = null) {
    let u = pre;
    if (!u) {
      try { u = await (await fetch('/api/update' + (force ? '?force=1' : ''))).json(); } catch (e) { return null; }
    }
    if (!u || !u.available || (u.skipped && !force)) return u;
    const notes = this.esc(u.notes || '').replace(/\n/g, '<br>');
    const inst = u.can_install;
    const html = `<div>Masz wersję <b>${this.esc(u.current)}</b>.</div>
      ${notes ? `<div style="margin:10px 0;padding:8px 10px;background:#141416;border:1px solid #2a2a2e;border-radius:6px;max-height:40vh;overflow-y:auto;">${notes}</div>` : ''}
      <div><a href="${this.esc(u.changelog)}" target="_blank" rel="noopener" style="color:#e8a33d;">Pełna lista zmian (CHANGELOG)</a> ·
        <a href="${this.esc(u.url)}" target="_blank" rel="noopener" style="color:#e8a33d;">strona wydania</a></div>
      <div style="margin-top:8px;color:#9a9aa0;">${inst ? 'Zaktualizuj = pobranie instalatora i jego uruchomienie; MatteDesk się zamknie, ustawienia zostają.'
        : 'Kopia z gita: zaktualizuj przez <code>git pull</code> w folderze MatteDesk i uruchom MatteDesk ponownie.'}</div>
      <label style="display:block;margin-top:8px;"><input type="checkbox" id="updSkip"> Nie przypominaj o tej wersji</label>`;
    const yes = await this.uiDialog({ title: `Nowa wersja MatteDesk ${u.version}`, html, ok: inst ? 'Zaktualizuj' : 'OK', cancel: 'Później', tone: 'info' });
    const skip = document.getElementById('updSkip');
    if (!yes && skip && skip.checked) {
      fetch('/api/update/skip', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ version: u.version }) });
    }
    if (yes && inst) {
      const r = await fetch('/api/update/install', { method: 'POST' });
      if (!r.ok) { await this.uiDialog({ title: 'Aktualizacja niemożliwa', tone: 'error', html: this.esc((await r.json()).detail || '') }); return u; }
      for (let i = 0; i < 1800; i++) {      // up to ~30 min
        await new Promise((res) => setTimeout(res, 1000));
        let st;
        try { st = await (await fetch('/api/update/progress')).json(); } catch (e) { break; }   // server stopped: setup runs
        if (st.state === 'failed') { await this.uiDialog({ title: 'Aktualizacja nie powiodła się', tone: 'error', html: this.esc(st.error || '') }); return u; }
        if (st.state === 'downloading' && st.total) this.showToast(`Pobieranie aktualizacji: ${Math.round(100 * st.done / st.total)} %`);
        if (st.state === 'started') break;
      }
      document.body.innerHTML = '<div style="padding:40px;font:16px sans-serif;color:#dcdcdf;background:#19191b;height:100vh;">Instalator aktualizacji działa – to okno można zamknąć.</div>';
      window.close();
    }
    return u;
  }

  bindElements() {
    this.plateSelect = document.getElementById('plateSelect');
    this.shotNameDisplay = document.getElementById('shotNameDisplay');
    this.shotRangeDisplay = document.getElementById('shotRangeDisplay');
    this.shotResDisplay = document.getElementById('shotResDisplay');

    this.frameInput = document.getElementById('frameInput');
    this.frameTotalDisplay = document.getElementById('frameTotalDisplay');
    this.frameIndexDisplay = document.getElementById('frameIndexDisplay');
    this.fpsSelect = document.getElementById('fpsSelect');

    this.btnPlayPause = document.getElementById('btnPlayPause');
    this.btnPrevFrame = document.getElementById('btnPrevFrame');
    this.btnNextFrame = document.getElementById('btnNextFrame');
    this.btnFirstFrame = document.getElementById('btnFirstFrame');
    this.btnLastFrame = document.getElementById('btnLastFrame');

    this.timelineTrack = document.getElementById('timelineTrack');
    this.timelinePlayhead = document.getElementById('timelinePlayhead');
    this.timelineBufferedBar = document.getElementById('timelineBufferedBar');
    this.timelineRuler = document.getElementById('timelineRuler');

    this.hudZoom = document.getElementById('hudZoom');
    this.hudCoords = document.getElementById('hudCoords');
    this.hudPlateRes = document.getElementById('hudPlateRes');
    this.loadingOverlay = document.getElementById('loadingOverlay');
    this.loadingMessage = document.getElementById('loadingMessage');

    // Modals
    this.openProjectModal = document.getElementById('openProjectModal');
    this.doctorModal = document.getElementById('doctorModal');
    this.shortcutsModal = document.getElementById('shortcutsModal');
    this.workDirInput = document.getElementById('workDirInput');
    this.btnBrowseWorkDir = document.getElementById('btnBrowseWorkDir');
    this.workDirStatusHint = document.getElementById('workDirStatusHint');
    this.inputBeautyPlate = document.getElementById('inputBeautyPlate');
    this.inputCleanPlate = document.getElementById('inputCleanPlate');
    this.inputBacklitPlate = document.getElementById('inputBacklitPlate');
    this.inputDepthPlate = document.getElementById('inputDepthPlate');
    this.doctorReportContent = document.getElementById('doctorReportContent');
    this.comfyStatusText = document.getElementById('comfyStatusText');
    this.comfyStatusDot = document.querySelector('.status-dot');

    // Clean Modal Elements (Stage E1c: Deflicker & Denoise)
    this.cleanModal = document.getElementById('cleanModal');
    this.closeCleanModalBtn = document.getElementById('closeCleanModalBtn');
    this.closeCleanBtn = document.getElementById('closeCleanBtn');
    this.cleanPlateSelect = document.getElementById('cleanPlateSelect');
    this.cleanBadgeStatus = document.getElementById('cleanBadgeStatus');
    this.cleanSummaryText = document.getElementById('cleanSummaryText');
    this.cleanMetricsGrid = document.getElementById('cleanMetricsGrid');
    this.metricCleanInit = document.getElementById('metricCleanInit');
    this.metricCleanFinal = document.getElementById('metricCleanFinal');
    this.metricCleanRedPct = document.getElementById('metricCleanRedPct');
    this.metricCleanNoiseRed = document.getElementById('metricCleanNoiseRed');
    this.cleanStage1Checkbox = document.getElementById('cleanStage1Checkbox');
    this.cleanStage2Checkbox = document.getElementById('cleanStage2Checkbox');
    this.cleanStage4Checkbox = document.getElementById('cleanStage4Checkbox');
    this.cleanStage3Checkbox = document.getElementById('cleanStage3Checkbox');
    this.cleanDenoiseCheckbox = document.getElementById('cleanDenoiseCheckbox');
    this.cleanExportModeSelect = document.getElementById('cleanExportModeSelect');
    this.cleanProgressBar = document.getElementById('cleanProgressBar');
    this.cleanProgressFill = document.getElementById('cleanProgressFill');
    this.cleanProgressText = document.getElementById('cleanProgressText');
    this.btnMeasureClean = document.getElementById('btnMeasureClean');
    this.btnRunClean = document.getElementById('btnRunClean');
    this.btnExportCleanNk = document.getElementById('btnExportCleanNk');

    // Camera 2D Modal Elements (Stage E1d)
    this.cameraModal = document.getElementById('cameraModal');
    this.closeCameraModalBtn = document.getElementById('closeCameraModalBtn');
    this.cameraBadgeStatus = document.getElementById('cameraBadgeStatus');
    this.cameraMotionTypeBadge = document.getElementById('cameraMotionTypeBadge');
    this.cameraSummaryText = document.getElementById('cameraSummaryText');
    this.cameraMetricsGrid = document.getElementById('cameraMetricsGrid');
    this.metricCamMotionType = document.getElementById('metricCamMotionType');
    this.metricCamResMedian = document.getElementById('metricCamResMedian');
    this.metricCamResP95 = document.getElementById('metricCamResP95');
    this.metricCamDrift = document.getElementById('metricCamDrift');
    this.cameraRefFrameInput = document.getElementById('cameraRefFrameInput');
    this.cameraAnchorIntervalInput = document.getElementById('cameraAnchorIntervalInput');
    this.cameraEvalWSelect = document.getElementById('cameraEvalWSelect');
    this.cameraProgressBar = document.getElementById('cameraProgressBar');
    this.cameraProgressFill = document.getElementById('cameraProgressFill');
    this.cameraProgressText = document.getElementById('cameraProgressText');
    this.cameraFileInfo = document.getElementById('cameraFileInfo');
    this.cameraJsonPathDisplay = document.getElementById('cameraJsonPathDisplay');
    this.cameraNkPathDisplay = document.getElementById('cameraNkPathDisplay');
    this.btnRunCameraTrack = document.getElementById('btnRunCameraTrack');
    this.btnExportCameraNk = document.getElementById('btnExportCameraNk');

    // Plates Management Modal
    this.openPlatesModalBtn = document.getElementById('openPlatesModalBtn');
    this.platesModal = document.getElementById('platesModal');
    this.closePlatesModalBtn = document.getElementById('closePlatesModalBtn');
    this.closePlatesBtn = document.getElementById('closePlatesBtn');
    this.platesModalShotName = document.getElementById('platesModalShotName');
    this.platesListContainer = document.getElementById('platesListContainer');
    this.plateNameSelect = document.getElementById('plateNameSelect');
    this.customPlateNameGroup = document.getElementById('customPlateNameGroup');
    this.customPlateNameInput = document.getElementById('customPlateNameInput');
    this.platePatternInput = document.getElementById('platePatternInput');
    this.btnInspectPlatePath = document.getElementById('btnInspectPlatePath');
    this.plateInspectResult = document.getElementById('plateInspectResult');
    this.plateGenProxiesCheckbox = document.getElementById('plateGenProxiesCheckbox');
    this.btnSavePlate = document.getElementById('btnSavePlate');
  }

  initEvents() {
    window.addEventListener('resize', () => {
      this.resizeCanvas();
      if (this.canvas.width > 0 && this.canvas.height > 0) {
        this.fitToScreen();
      }
      this.render();
    });

    if (window.ResizeObserver && this.canvasWrapper) {
      const ro = new ResizeObserver(() => {
        const prevW = this.canvas.width;
        const prevH = this.canvas.height;
        this.resizeCanvas();
        if (this.canvas.width > 0 && this.canvas.height > 0) {
          if (!this.hasInitialFit && this.hasFrame(this.currentFrame)) {
            this.fitToScreen();
          } else if (prevW !== this.canvas.width || prevH !== this.canvas.height) {
            this.fitToScreen();
          } else {
            this.render();
          }
        }
      });
      ro.observe(this.canvasWrapper);
    }

    // Plate selector
    if (this.plateSelect) {
      this.plateSelect.addEventListener('change', (e) => this.setActivePlate(e.target.value));
    }
    this.plateRefSelect = document.getElementById('plateRefSelect');
    if (this.plateRefSelect) this.plateRefSelect.addEventListener('change', (e) => this.setCompareRef(e.target.value));
    // comparison buttons next to the plate pair (moved from the separate bar under the header, 2026-10-15)
    const cmpSeg = document.getElementById('p0Cmp');
    if (cmpSeg) cmpSeg.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { if (!b.disabled) this.setCompare(b.dataset.m); }));
    const wipeIn = document.getElementById('p0Wipe');
    if (wipeIn) {
      wipeIn.addEventListener('input', (e) => { this.compare.wipe = +e.target.value; this.render(); });
      window.addEventListener('md-wipe', (e) => { wipeIn.value = e.detail; });
    }
    const gainIn = document.getElementById('p0Gain');
    if (gainIn) gainIn.addEventListener('change', (e) => { this.compare.gain = +e.target.value; this.updateHudMode(); this.render(); });
    const saveBg = () => { try { localStorage.setItem('md_cmp_bg', JSON.stringify(this.compare.bg)); } catch (e) { /* no storage */ } };
    const bgColor = document.getElementById('cmpBgColor');
    if (bgColor) {
      bgColor.value = this.compare.bg.color;
      bgColor.addEventListener('input', (e) => { this.compare.bg.color = e.target.value; this.compare.bg.checker = false; saveBg(); this.syncCompareUi(); this.render(); });
    }
    // the comparison row under the header starts where the "Plate:" pill starts (user request 2026-10-15);
    // the pill folded into the "⋯" menu (narrow window) -> the row starts at the left edge
    const cmpBar = document.getElementById('compareBar'), cmpInner = document.getElementById('compareInner');
    const pill = document.getElementById('plateSelectorPill');
    const alignCompareBar = () => {
      if (!cmpBar || !cmpInner || !pill) return;
      const folded = pill.classList.contains('in-more-menu') || !pill.offsetParent;
      const x = folded ? 16 : Math.round(pill.getBoundingClientRect().left - cmpBar.getBoundingClientRect().left);
      const max = Math.max(16, cmpBar.clientWidth - cmpInner.offsetWidth);   // never pushed out of the window
      cmpInner.style.marginLeft = `${Math.max(0, Math.min(x, max))}px`;
    };
    this.alignCompareBar = alignCompareBar;
    window.addEventListener('resize', alignCompareBar);
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => alignCompareBar());
      const hdr = document.querySelector('.app-header');
      if (hdr) for (const g of hdr.children) ro.observe(g);
      if (pill) ro.observe(pill);
      if (cmpInner) ro.observe(cmpInner);
    }
    if (window.MutationObserver && pill) new MutationObserver(alignCompareBar).observe(pill, { attributes: true, attributeFilter: ['class'] });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(alignCompareBar);
    setTimeout(alignCompareBar, 0);
    const bgChk = document.getElementById('cmpBgChecker');
    if (bgChk) bgChk.addEventListener('click', () => { this.compare.bg.checker = !this.compare.bg.checker; saveBg(); this.syncCompareUi(); this.render(); });

    // Transport buttons
    this.btnPlayPause.addEventListener('click', () => this.togglePlay());
    this.btnPrevFrame.addEventListener('click', () => this.stepFrame(-1));
    this.btnNextFrame.addEventListener('click', () => this.stepFrame(1));
    this.btnFirstFrame.addEventListener('click', () => this.goToFrame(this.firstFrame));
    this.btnLastFrame.addEventListener('click', () => this.goToFrame(this.lastFrame));

    this.frameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); this.goToFrame(parseInt(this.frameInput.value, 10) || this.firstFrame); this.frameInput.blur(); }
      else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') { e.preventDefault(); this.stepFrame(e.key === 'ArrowUp' ? 1 : -1); }
    });
    this.frameInput.addEventListener('change', (e) => {
      const val = parseInt(e.target.value, 10);
      if (!isNaN(val)) this.goToFrame(val);
    });

    this.fpsSelect.addEventListener('change', (e) => {
      this.fps = parseInt(e.target.value, 10);
      if (this.isPlaying) {
        this.stopPlayback();
        this.startPlayback();
      }
    });

    // Timeline Scrubbing
    let isScrubbing = false;
    const handleScrub = (e) => {
      const rect = this.timelineTrack.getBoundingClientRect();
      const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const frame = Math.round(this.firstFrame + pos * (this.lastFrame - this.firstFrame));
      this.goToFrame(frame);
    };

    this.timelineTrack.addEventListener('mousedown', (e) => {
      isScrubbing = true;
      handleScrub(e);
    });
    window.addEventListener('mousemove', (e) => {
      if (isScrubbing) handleScrub(e);
    });
    window.addEventListener('mouseup', () => {
      isScrubbing = false;
    });

    // Canvas Pan & Zoom
    this.canvas.addEventListener('mousedown', (e) => {
      if (e.mdHandled) return;
      if (this.compare.mode === 'wipe' && e.button === 0 && this.nearWipe(e)) { this.wipeDrag = true; return; }
      if (e.button === 0 || e.button === 1) { // Left or Middle click
        this.isPanning = true;
        this.panStartX = e.clientX - this.panX;
        this.panStartY = e.clientY - this.panY;
        this.canvas.classList.add('panning');
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (this.wipeDrag) { this.setWipeFromEvent(e); return; }
      if (this.isPanning) {
        this.panX = e.clientX - this.panStartX;
        this.panY = e.clientY - this.panStartY;
        this.render();
      }
      if (this.compare.mode === 'wipe') this.canvas.style.cursor = this.nearWipe(e) ? 'col-resize' : '';

      // Update HUD cursor position on plate
      const rect = this.canvas.getBoundingClientRect();
      const mouseCanvasX = e.clientX - rect.left;
      const mouseCanvasY = e.clientY - rect.top;
      const img = this.getCurrentImage();
      if (img) {
        const plateX = Math.round((mouseCanvasX - this.panX) / this.zoom);
        const plateY = Math.round((mouseCanvasY - this.panY) / this.zoom);
        this.hudCoords.textContent = `X: ${plateX}, Y: ${plateY}`;
      }
    });

    window.addEventListener('mouseup', () => {
      this.wipeDrag = false;
      if (this.isPanning) {
        this.isPanning = false;
        this.canvas.classList.remove('panning');
      }
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const newZoom = Math.max(0.1, Math.min(10.0, this.zoom * zoomFactor));

      // Zoom towards mouse pointer
      this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
      this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
      this.zoom = newZoom;

      this.hudZoom.textContent = `Zoom: ${Math.round(this.zoom * 100)}%`;
      this.render();
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.code === 'KeyC' && !e.ctrlKey && !e.metaKey && this.activePlate !== this.compare.ref) {
        if (!this.compare.holdC) { this.compare.holdC = true; this.render(); }
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          this.togglePlay();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          if (e.shiftKey) window.dispatchEvent(new CustomEvent('md-key-step', { detail: -1 })); else this.stepFrame(-1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          if (e.shiftKey) window.dispatchEvent(new CustomEvent('md-key-step', { detail: 1 })); else this.stepFrame(1);
          break;
        case 'Home':
          e.preventDefault();
          this.goToFrame(this.firstFrame);
          break;
        case 'End':
          e.preventDefault();
          this.goToFrame(this.lastFrame);
          break;
        case 'Digit1':
          this.setZoom100();
          break;
        case 'Digit2':
          this.fitToScreen();
          break;
      }
    });

    window.addEventListener('keyup', (e) => {
      if (e.code === 'KeyC' && this.compare.holdC) { this.compare.holdC = false; this.render(); }
    });

    // Toolbar buttons
    document.getElementById('toolZoomFitBtn').addEventListener('click', () => this.fitToScreen());
    document.getElementById('toolZoom100Btn').addEventListener('click', () => this.setZoom100());

    // Project Modal
    document.getElementById('openProjectModalBtn').addEventListener('click', () => {
      this.openProjectModal.style.display = 'flex';
      this.loadRecentProjects();
      if (!this.workDirInput.value && this.projectLoaded && this.currentWorkDir) {
        this.workDirInput.value = this.currentWorkDir;
        this.inspectWorkingDir(this.currentWorkDir);
      }
    });
    document.getElementById('closeProjectModalBtn').addEventListener('click', () => {
      this.openProjectModal.style.display = 'none';
    });
    document.getElementById('cancelOpenProjectBtn').addEventListener('click', () => {
      this.openProjectModal.style.display = 'none';
    });
    document.getElementById('confirmOpenProjectBtn').addEventListener('click', () => {
      this.openOrCreateProject();
    });

    if (this.btnBrowseWorkDir) {
      this.btnBrowseWorkDir.addEventListener('click', () => this.browseWorkingDir());
    }
    if (this.workDirInput) {
      this.workDirInput.addEventListener('change', () => this.inspectWorkingDir(this.workDirInput.value.trim()));
    }
    document.querySelectorAll('.btn-browse-plate').forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-target');
        this.browsePlateFolder(targetId, btn.hasAttribute('data-file'));
      });
    });

    // Doctor Modal
    document.getElementById('openDoctorBtn').addEventListener('click', () => {
      this.doctorModal.style.display = 'flex';
      this.fetchDoctorReport();
    });
    document.getElementById('closeDoctorModalBtn').addEventListener('click', () => {
      this.doctorModal.style.display = 'none';
    });
    document.getElementById('closeDoctorBtn').addEventListener('click', () => {
      this.doctorModal.style.display = 'none';
    });
    document.getElementById('refreshDoctorBtn').addEventListener('click', () => {
      this.fetchDoctorReport();
    });

    // Shortcuts Modal
    document.getElementById('btnHelpShortcuts').addEventListener('click', () => {
      this.shortcutsModal.style.display = 'flex';
    });
    document.getElementById('closeShortcutsBtn').addEventListener('click', () => {
      this.shortcutsModal.style.display = 'none';
    });
    document.getElementById('confirmShortcutsBtn').addEventListener('click', () => {
      this.shortcutsModal.style.display = 'none';
    });
    // also closed by a click outside the card and by Esc (capture: before the stage tools see the Esc)
    this.shortcutsModal.addEventListener('mousedown', (e) => { if (e.target === this.shortcutsModal) this.shortcutsModal.style.display = 'none'; });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.shortcutsModal.style.display !== 'none') { e.stopPropagation(); e.preventDefault(); this.shortcutsModal.style.display = 'none'; }
    }, true);

    // Clean Modal Events (Stage E1c: Deflicker & Denoise)
    if (this.closeCleanModalBtn) {
      this.closeCleanModalBtn.addEventListener('click', () => {
        this.cleanModal.style.display = 'none';
      });
    }
    if (this.closeCleanBtn) {
      this.closeCleanBtn.addEventListener('click', () => {
        this.cleanModal.style.display = 'none';
      });
    }
    if (this.cleanPlateSelect) {
      this.cleanPlateSelect.addEventListener('change', () => this.checkExistingCleanReport());
    }
    if (this.btnMeasureClean) {
      this.btnMeasureClean.addEventListener('click', () => this.measurePlateClean());
    }
    if (this.btnRunClean) {
      this.btnRunClean.addEventListener('click', () => this.runPlateClean());
    }
    if (this.btnExportCleanNk) {
      this.btnExportCleanNk.addEventListener('click', () => this.exportCleanNk());
    }

    // Camera 2D Events (Stage E1d)
    if (this.closeCameraModalBtn) {
      this.closeCameraModalBtn.addEventListener('click', () => {
        this.cameraModal.style.display = 'none';
      });
    }
    if (this.btnRunCameraTrack) {
      this.btnRunCameraTrack.addEventListener('click', () => this.runCameraTrack());
    }
    if (this.btnExportCameraNk) {
      this.btnExportCameraNk.addEventListener('click', () => this.exportCameraNk());
    }

    // Plates Management Modal Events
    if (this.openPlatesModalBtn) {
      this.openPlatesModalBtn.addEventListener('click', () => this.openPlatesModal());
    }
    if (this.closePlatesModalBtn) {
      this.closePlatesModalBtn.addEventListener('click', () => {
        this.platesModal.style.display = 'none';
      });
    }
    if (this.closePlatesBtn) {
      this.closePlatesBtn.addEventListener('click', () => {
        this.platesModal.style.display = 'none';
      });
    }
    if (this.plateNameSelect) {
      this.plateNameSelect.addEventListener('change', (e) => {
        if (this.customPlateNameGroup) {
          this.customPlateNameGroup.style.display = e.target.value === 'custom' ? 'block' : 'none';
        }
      });
    }
    const btnBrowsePlate = document.getElementById('btnBrowsePlatePath');
    if (btnBrowsePlate) btnBrowsePlate.addEventListener('click', () => this.browsePlatePath());
    const btnBrowseFile = document.getElementById('btnBrowsePlateFile');
    if (btnBrowseFile) btnBrowseFile.addEventListener('click', () => this.browsePlatePath(true));
    if (this.btnInspectPlatePath) {
      this.btnInspectPlatePath.addEventListener('click', () => this.inspectPlatePath());
    }
    if (this.btnSavePlate) {
      this.btnSavePlate.addEventListener('click', () => this.savePlate());
    }
  }

  initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/events`;
    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          this.handleWsMessage(msg);
        } catch (e) {}
      };
      this.ws.onclose = () => {
        setTimeout(() => this.initWebSocket(), 5000);
      };
    } catch (e) {}
  }

  handleWsMessage(msg) {
    if (msg.type === 'proxy_progress') {
      const pct = Math.round((msg.done / msg.total) * 100);
      this.timelineBufferedBar.style.width = `${pct}%`;
      this.showToast(`Generowanie proxy: ${msg.done} / ${msg.total} (${pct}%)`);
    } else if (msg.type === 'project_opened') {
      this.loadCurrentProject();
    } else if (msg.type === 'align_progress' || msg.type === 'align_apply_progress') {
      window.dispatchEvent(new CustomEvent('md-align-progress', { detail: msg }));
    } else if (msg.type === 'clean_progress') {
      if (this.cleanProgressBar && this.cleanProgressFill) {
        const pct = Math.round((msg.done / msg.total) * 100);
        this.cleanProgressBar.style.display = 'flex';
        this.cleanProgressFill.style.width = `${pct}%`;
        this.cleanProgressText.textContent = `${msg.step} (${pct}%)`;
      }
    } else if (msg.type === 'camera_progress') {
      if (this.cameraProgressBar && this.cameraProgressFill) {
        const pct = Math.round((msg.done / msg.total) * 100);
        this.cameraProgressBar.style.display = 'flex';
        this.cameraProgressFill.style.width = `${pct}%`;
        this.cameraProgressText.textContent = `${msg.step} (${pct}%)`;
      }
    } else if (msg.type === 'plates_updated') {
      this.updatePlateSelector(msg.plates);
      window.dispatchEvent(new CustomEvent('md-plates-updated', { detail: msg }));
      if (this.platesModal && this.platesModal.style.display !== 'none') {
        this.fetchAndRenderPlates();
      }
    }
  }

  resizeCanvas() {
    if (!this.canvasWrapper) return;
    const rect = this.canvasWrapper.getBoundingClientRect();
    const w = Math.round(rect.width || this.canvasWrapper.clientWidth || 0);
    const h = Math.round(rect.height || this.canvasWrapper.clientHeight || 0);
    if (w > 50 && h > 50 && (this.canvas.width !== w || this.canvas.height !== h)) {
      this.canvas.width = w;    // assigning width/height clears the bitmap – only when the size really changed
      this.canvas.height = h;
    }
  }

  // --------------------------------------------------------------------------
  // Project Loading & State
  // --------------------------------------------------------------------------
  async loadCurrentProject() {
    try {
      const res = await fetch('/api/project/current');
      const data = await res.json();
      if (data.loaded) {
        this.applyProject(data);
      }
    } catch (e) {
      console.warn('Could not load current project', e);
    }
  }

  async loadRecentProjects() {
    const group = document.getElementById('recentProjectsGroup');
    const list = document.getElementById('recentProjectsList');
    if (!group || !list) return;
    try {
      const res = await fetch('/api/project/recent');
      const data = await res.json();
      const items = data.recent || [];
      list.innerHTML = '';
      items.forEach((it) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn-secondary btn-sm recent-project-btn';
        b.textContent = (it.exists ? '' : '⚠ ') + it.work_dir;
        b.title = it.exists ? 'Wstaw do pola katalogu' : 'Katalog nie istnieje na dysku';
        b.addEventListener('click', () => {
          this.workDirInput.value = it.work_dir;
          this.inspectWorkingDir(it.work_dir);
        });
        list.appendChild(b);
      });
      group.style.display = items.length ? '' : 'none';
    } catch (e) {
      console.warn('Could not load recent projects', e);
      group.style.display = 'none';
    }
  }

  async browseWorkingDir() {
    try {
      const res = await fetch('/api/dialog/pick_folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Wybierz katalog roboczy projektu' })
      });
      const data = await res.json();
      if (data.folder) {
        this.workDirInput.value = data.folder;
        await this.inspectWorkingDir(data.folder);
      }
    } catch (e) {
      console.warn('Native picker failed:', e);
    }
  }

  // project wizard: 📁 = sequence folder, 🎞️ = a file (one frame of a sequence, or a video – decoded to a sequence on open)
  async browsePlateFolder(targetInputId, file = false) {
    try {
      const res = await fetch(file ? '/api/dialog/pick_file' : '/api/dialog/pick_folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: window.mdT(file ? 'Wybierz plik (klatka albo wideo)' : "Wybierz folder sekwencji plate'u") })
      });
      const data = await res.json();
      const picked = file ? data.path : data.folder;
      if (picked) {
        const input = document.getElementById(targetInputId);
        if (input) {
          input.value = picked;
        }
      }
    } catch (e) {
      console.warn('Native picker failed:', e);
    }
  }

  async inspectWorkingDir(dirPath) {
    if (!dirPath) return;
    if (this.workDirStatusHint) {
      this.workDirStatusHint.textContent = 'Sprawdzanie katalogu roboczego...';
      this.workDirStatusHint.className = 'field-hint';
    }

    try {
      const res = await fetch('/api/project/inspect_working_dir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dir_path: dirPath })
      });
      const resp = await res.json();
      if (resp.success && resp.info) {
        const info = resp.info;
        if (!info.exists) {
          this.workDirStatusHint.textContent = 'Katalog nie istnieje na dysku. Zostanie utworzony po zatwierdzeniu.';
          this.workDirStatusHint.className = 'field-hint warning';
          return;
        }

        if (info.manifest_found) {
          const shot = info.shot || 'nieznane';
          this.workDirStatusHint.textContent = `✓ Znaleziono manifest: shot_manifest_${shot}.json. Pola płyt uzupełnione automatycznie.`;
          this.workDirStatusHint.className = 'field-hint success';

          const plates = info.plates || {};
          if (plates.beauty && this.inputBeautyPlate) this.inputBeautyPlate.value = plates.beauty;
          if ((plates.cleanplate || plates.clean_plate) && this.inputCleanPlate) {
            this.inputCleanPlate.value = plates.cleanplate || plates.clean_plate;
          }
          if (plates.backlit && this.inputBacklitPlate) this.inputBacklitPlate.value = plates.backlit;
          if (plates.depth && this.inputDepthPlate) this.inputDepthPlate.value = plates.depth;
        } else {
          this.workDirStatusHint.textContent = 'Katalog gotowy. Wprowadź ścieżki płyt, aby zainicjalizować ujęcie.';
          this.workDirStatusHint.className = 'field-hint';

          const det = info.detected_plates || {};
          if (det.beauty && this.inputBeautyPlate && !this.inputBeautyPlate.value) this.inputBeautyPlate.value = det.beauty;
          if (det.cleanplate && this.inputCleanPlate && !this.inputCleanPlate.value) this.inputCleanPlate.value = det.cleanplate;
          if (det.backlit && this.inputBacklitPlate && !this.inputBacklitPlate.value) this.inputBacklitPlate.value = det.backlit;
          if (det.depth && this.inputDepthPlate && !this.inputDepthPlate.value) this.inputDepthPlate.value = det.depth;
        }
      }
    } catch (e) {
      if (this.workDirStatusHint) {
        this.workDirStatusHint.textContent = `Błąd sprawdzania katalogu: ${e}`;
        this.workDirStatusHint.className = 'field-hint error';
      }
    }
  }

  async openOrCreateProject() {
    const workDir = this.workDirInput.value.trim();
    if (!workDir) {
      alert('Podaj lub wybierz katalog roboczy ujęcia.');
      return;
    }

    const plates = {};
    if (this.inputBeautyPlate && this.inputBeautyPlate.value.trim()) {
      plates.beauty = this.inputBeautyPlate.value.trim();
    }
    if (this.inputCleanPlate && this.inputCleanPlate.value.trim()) {
      plates.cleanplate = this.inputCleanPlate.value.trim();
    }
    if (this.inputBacklitPlate && this.inputBacklitPlate.value.trim()) {
      plates.backlit = this.inputBacklitPlate.value.trim();
    }
    if (this.inputDepthPlate && this.inputDepthPlate.value.trim()) {
      plates.depth = this.inputDepthPlate.value.trim();
    }

    this.showLoading(true, 'Wczytywanie / konfigurowanie projektu...');
    try {
      const res = await fetch('/api/project/open_or_create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          work_dir: workDir,
          plates: plates
        })
      });
      const data = await res.json();
      if (data.success) {
        this.openProjectModal.style.display = 'none';
        this.currentWorkDir = workDir;
        this.applyProject(data);
        this.showToast(`Projekt ${data.manifest?.shot || ''} otwarty pomyślnie`);
      } else {
        alert(`Błąd otwierania projektu: ${data.detail || data.error}`);
      }
    } catch (e) {
      alert(`Błąd połączenia z serwerem: ${e}`);
    } finally {
      this.showLoading(false);
    }
  }

  applyProject(data) {
    this.projectLoaded = true;
    const m = data.manifest;
    this.manifest = m;
    this.projectData = data.project || m;
    this.currentWorkDir = m.work || data.work_dir || data.working_dir || '';

    this.firstFrame = m.first || 1;
    this.lastFrame = m.last || 100;
    this.fps = m.fps || 24;
    this.fpsSelect.value = String(this.fps);
    this.currentFrame = this.firstFrame;

    this.shotNameDisplay.textContent = m.shot || 'Ujęcie';
    this.shotRangeDisplay.textContent = `F${this.firstFrame} - F${this.lastFrame} (${this.lastFrame - this.firstFrame + 1} kl.)`;
    this.shotResDisplay.textContent = `${m.plate_w}x${m.plate_h} (${m.colorspace || 'ACES'})`;
    this.hudPlateRes.textContent = `${m.plate_w}x${m.plate_h}`;

    // Update plate selector
    this.updatePlateSelector(m.plates);

    this.frameInput.min = this.firstFrame;
    this.frameInput.max = this.lastFrame;
    this.frameInput.value = this.currentFrame;
    this.frameTotalDisplay.textContent = `/ ${this.lastFrame}`;

    // Update timeline ruler
    this.updateTimelineRuler();

    // Clear frame cache & reset initial fit flag. Project generation (bug 2026-10-15: after opening another project the
    // preview kept showing the previous one): frames / tiles still loading for the old project are dropped when they
    // arrive, the "loading" set and the tile cache are cleared, and the URLs carry the generation (no shared HTTP cache).
    this.projectGen = (this.projectGen || 0) + 1;
    this.frameCache.clear();
    this.loadingFrames.clear();
    if (this.frameErrors) this.frameErrors.clear();
    if (this.tiles) { this.tiles.cache.clear(); this.tiles.region = null; }
    this.hasInitialFit = false;

    // Go to first frame
    this.goToFrame(this.firstFrame);
    window.dispatchEvent(new CustomEvent('md-project', { detail: data }));
  }

  setActivePlate(name) {
    if (!name) return;
    const prev = this.activePlate;
    this.activePlate = name;
    if (this.plateSelect && this.plateSelect.value !== name) this.plateSelect.value = name;
    if (name === 'beauty') this.alignedPreview = false;
    if (name === this.compare.ref) this.compare.mode = 'none';   // nothing to compare with itself
    this.loadingFrames.clear();
    this.goToFrame(this.currentFrame);
    this.updateHudMode();
    this.syncCompareUi();
    if (prev !== name) window.dispatchEvent(new CustomEvent('md-plate', { detail: { plate: name, prev } }));
  }

  // --- comparison of the left plate with the right side of the header pair (a plate or a comp background)
  isBgRef(ref) { const r = ref === undefined ? this.compare.ref : ref; return r === '__color' || r === 'bg'; }
  // the comp background covers the view (objects.js then skips the mask overlays, they would tint the comp)
  compareCoversView() {
    return this.isBgRef() && (this.compare.mode === 'comp' || this.compare.mode === 'wipe' || this.compare.holdC);
  }
  loadCompareRef(f) {   // images the right side needs in frame f
    const ref = this.compare.ref;
    if (this.isBgRef()) {
      this.loadFrame(f, null, 'beauty', false);   // RGB always from beauty
      if (ref === 'bg') this.loadFrame(f, null, 'bg', false);
    } else if ((this.activePlate || 'beauty') !== ref) this.loadFrame(f, null, ref, false);
  }
  setCompare(mode) {
    this.compare.mode = mode || 'none';
    this.updateHudMode();
    this.syncCompareUi();
    this.loadCompareRef(this.currentFrame);
    this.render();
    this.notifyCompare();
  }
  // objects.js / zones.js: the cut-out view switches the band / alpha previews off (they looked like an error over it,
  // user report 2026-10-17) and back on when leaving it
  notifyCompare() {
    const covers = !!this.compareCoversView();
    if (covers === this._coversWas) return;
    this._coversWas = covers;
    window.dispatchEvent(new CustomEvent('md-compare', { detail: { covers } }));
  }
  setCompareRef(name) {
    if (!name) return;
    if (name === 'bg' && !this._bgPattern) {   // no target background yet: open the plates dialog on the BG form
      if (this.plateRefSelect) this.plateRefSelect.value = this.compare.ref;
      this.openPlatesModal();
      setTimeout(() => this.preparePlateForm('bg'), 300);
      return;
    }
    const wasBg = this.isBgRef();
    this.compare.ref = name;
    if (this.plateRefSelect && this.plateRefSelect.value !== name) this.plateRefSelect.value = name;
    if (this.isBgRef()) { if (!wasBg && this.compare.mode !== 'wipe' && this.compare.mode !== 'ab') this.compare.mode = 'comp'; }   // a background = see the cut-out at once
    else if (name === this.activePlate || this.compare.mode === 'comp') this.compare.mode = 'none';
    this.edgeCache.clear();
    this.updateHudMode();
    this.syncCompareUi();
    this.loadCompareRef(this.currentFrame);
    this.render();
    this.notifyCompare();
    window.dispatchEvent(new CustomEvent('md-plate', { detail: { plate: this.activePlate, prev: this.activePlate } }));
  }
  // buttons / extra controls of the comparison next to the plate pair
  syncCompareUi() {
    const seg = document.getElementById('p0Cmp');
    if (!seg) return;
    const bgRef = this.isBgRef();
    const cur = this.activePlate || 'beauty';
    const canCmp = bgRef || cur !== this.compare.ref;
    const m = this.compare.mode || 'none';
    seg.querySelectorAll('button').forEach((b) => {
      const k = b.dataset.m;
      const off = !canCmp || (bgRef ? (k === 'diff' || k === 'edges') : k === 'comp');
      b.disabled = off && k !== 'none';
      b.hidden = bgRef ? (k === 'diff' || k === 'edges') : k === 'comp';
      b.classList.toggle('active', m === k);
    });
    seg.title = canCmp ? '' : 'Wybierz dwa różne plate\'y (lewy i prawy select) albo tło po prawej, żeby porównać';
    const show = (id, on) => { const e = document.getElementById(id); if (e) e.hidden = !on; };
    show('p0WipeRow', m === 'wipe' && canCmp);
    show('p0GainRow', m === 'diff' && canCmp && !bgRef);
    show('cmpBgCtl', this.compare.ref === '__color');
    const chk = document.getElementById('cmpBgChecker');
    if (chk) chk.classList.toggle('active', !!this.compare.bg.checker);
    const col = document.getElementById('cmpBgColor');
    if (col) { col.value = this.compare.bg.color; col.classList.toggle('dim', !!this.compare.bg.checker); }
  }
  setAlignedPreview(on) {
    this.alignedPreview = !!on;
    this.loadingFrames.clear();
    this.goToFrame(this.currentFrame);
    this.updateHudMode();
  }
  updateHudMode() {
    const el = document.getElementById('hudMode');
    if (!el) return;
    const parts = [];
    const ref = this.compare.ref;
    if (this.activePlate !== 'beauty' || (this.activePlate !== ref && this.compare.mode !== 'none')) {
      parts.push(`${this.activePlate}${this.alignedPreview && this.activePlate !== 'beauty' ? ' · po dopasowaniu' : ''}`);
      const refName = this.isBgRef() ? this.bgRefLabel() : ref;
      const names = { comp: `wycięcie na tle: ${refName}`, ab: `A/B (C = ${refName})`, wipe: `wipe ${refName} – przeciągnij linię`, diff: 'różnica ×' + this.compare.gain, edges: `kontury ${ref}` };
      if (names[this.compare.mode]) parts.push(names[this.compare.mode]);
    }
    el.style.display = parts.length ? '' : 'none';
    el.textContent = parts.join(' · ');    this.notifyCompare && this.notifyCompare();
  }
  nearWipe(e) {
    const img = this.getCurrentImage(); if (!img) return false;
    const rect = this.canvas.getBoundingClientRect();
    const x = this.panX + img.width * this.compare.wipe * this.zoom;
    return Math.abs(e.clientX - rect.left - x) < 7;
  }
  setWipeFromEvent(e) {
    const img = this.getCurrentImage(); if (!img) return;
    const rect = this.canvas.getBoundingClientRect();
    this.compare.wipe = Math.max(0, Math.min(1, (e.clientX - rect.left - this.panX) / this.zoom / img.width));
    this.render();
    window.dispatchEvent(new CustomEvent('md-wipe', { detail: this.compare.wipe }));
  }

  updateTimelineRuler() {
    const ruler = this.timelineRuler;
    ruler.innerHTML = '';
    const span1 = document.createElement('span');
    span1.textContent = `F${this.firstFrame}`;
    const midFrame = Math.round((this.firstFrame + this.lastFrame) / 2);
    const span2 = document.createElement('span');
    span2.textContent = `F${midFrame}`;
    const span3 = document.createElement('span');
    span3.textContent = `F${this.lastFrame}`;
    ruler.appendChild(span1);
    ruler.appendChild(span2);
    ruler.appendChild(span3);
  }

  async triggerProxyGeneration() {
    if (!this.projectLoaded) {
      alert('Najpierw otwórz projekt.');
      return;
    }
    const plate = encodeURIComponent(this.activePlate || 'beauty');
    this.showToast(`Rozpoczęto generowanie proxy (${this.activePlate || 'beauty'})...`);
    try {
      const res = await fetch(`/api/project/generate_proxies?plate=${plate}`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        this.showToast(`Wygenerowano ${data.rendered_count} klatek proxy!`);
        this.frameCache.clear();
        this.goToFrame(this.currentFrame);
      }
    } catch (e) {
      alert(`Błąd generowania proxy: ${e}`);
    }
  }

  // --------------------------------------------------------------------------
  // Frame Management & Preloading Cache
  // --------------------------------------------------------------------------
  goToFrame(frameNum, dir) {
    if (frameNum < this.firstFrame) frameNum = this.firstFrame;
    if (frameNum > this.lastFrame) frameNum = this.lastFrame;

    const changed = this.currentFrame !== frameNum;
    this.currentFrame = frameNum;
    this.frameInput.value = frameNum;
    this.frameIndexDisplay.textContent = `(idx: ${frameNum - this.firstFrame})`;

    // Update playhead
    const range = Math.max(1, this.lastFrame - this.firstFrame);
    const pct = ((this.currentFrame - this.firstFrame) / range) * 100;
    this.timelinePlayhead.style.left = `${pct}%`;

    // Ensure frame is cached or load it (placeholder meanwhile)
    this.loadFrame(frameNum, (img) => {
      this.render();
    });
    this.render();

    // Preload surrounding frames
    this.preloadNearby(frameNum);
    if (this.compare.mode !== 'none' || this.isBgRef()) this.loadCompareRef(frameNum);
    window.dispatchEvent(new CustomEvent('md-frame', { detail: { frame: frameNum, changed, dir: dir || 0 } }));
  }

  stepFrame(delta) {
    let next = this.currentFrame + delta;
    if (next > this.lastFrame) next = this.firstFrame;
    if (next < this.firstFrame) next = this.lastFrame;
    this.goToFrame(next);
  }

  loadFrame(frameNum, callback, plateName, alignedFlag) {
    const plate = plateName || this.activePlate || 'beauty';
    const aligned = alignedFlag === undefined ? (plate !== 'beauty' && this.alignedPreview) : alignedFlag;
    const cacheKey = this.frameKey(plate, frameNum, aligned);

    if (this.frameCache.has(cacheKey)) {
      if (callback) callback(this.frameCache.get(cacheKey));
      return;
    }

    if (this.loadingFrames.has(cacheKey)) return;
    this.loadingFrames.add(cacheKey);

    const img = new Image();
    const gen = this.projectGen || 0;
    img.crossOrigin = 'anonymous';
    img.src = `/api/frame/${frameNum}?plate=${encodeURIComponent(plate)}${aligned ? '&aligned=1' : ''}&g=${gen}`;
    img.onload = () => {
      if (gen !== (this.projectGen || 0)) return;   // a frame of the previous project – never cache / show it
      this.frameCache.set(cacheKey, img);
      this.loadingFrames.delete(cacheKey);
      window.dispatchEvent(new CustomEvent('md-frame-loaded', { detail: { frame: frameNum, plate } }));   // Z20 load bar
      if (this.frameCache.size > this.cacheLimit) {
        const oldestKey = this.frameCache.keys().next().value;
        this.frameCache.delete(oldestKey);
      }
      if (frameNum === this.currentFrame) {
        if (cacheKey === this.frameKey(this.activePlate || 'beauty', frameNum)) {
          this.resizeCanvas();
          if (!this.hasInitialFit && this.canvas.width > 0 && this.canvas.height > 0) this.fitToScreen();
          else this.render();
        } else if (plate === this.compare.ref || (this.isBgRef() && (plate === 'beauty' || plate === 'bg'))) {
          this.render();   // the comparison partner arrived
        }
      }
      if (callback) callback(img);
    };
    img.onerror = () => {
      if (gen !== (this.projectGen || 0)) return;
      this.loadingFrames.delete(cacheKey);
      if (aligned) {
        // "po dopasowaniu" outside the measured range (or no report): show the plate without alignment + a note (Z17)
        this.loadFrame(frameNum, (raw) => {
          this.alignedFallback = this.alignedFallback || new Set();
          this.alignedFallback.add(cacheKey);
          this.frameCache.set(cacheKey, raw);
          if (frameNum === this.currentFrame) this.render();
          if (callback) callback(raw);
        }, plate, false);
        return;
      }
      this.frameErrors = this.frameErrors || new Set();
      this.frameErrors.add(cacheKey);
      if (frameNum === this.currentFrame) this.render();
    };
  }

  getCurrentImage() {
    return this.frameCache.get(this.frameKey(this.activePlate || 'beauty', this.currentFrame));
  }
  getImage(plate, frameNum, aligned) {
    return this.frameCache.get(this.frameKey(plate, frameNum, aligned));
  }

  frameKey(plate, frameNum, aligned) {
    const al = aligned === undefined ? (plate !== 'beauty' && this.alignedPreview) : aligned;
    return `${plate}${al ? '_al' : ''}_${frameNum}`;
  }

  hasFrame(frameNum) {
    return this.frameCache.has(this.frameKey(this.activePlate || 'beauty', frameNum));
  }

  updatePlateSelector(plates) {
    if (!this.plateSelect) return;
    this.plateSelect.innerHTML = '';
    if (this.plateRefSelect) this.plateRefSelect.innerHTML = '';
    if (!plates || Object.keys(plates).length === 0) {
      const opt = document.createElement('option');
      opt.value = 'beauty';
      opt.textContent = 'Beauty Plate (X1)';
      this.plateSelect.appendChild(opt);
      if (this.plateRefSelect) this.plateRefSelect.appendChild(opt.cloneNode(true));
      this.activePlate = 'beauty';
      this.compare.ref = 'beauty';
      return;
    }

    const labels = {
      beauty: 'Beauty Plate (X1)',
      cleanplate: 'Clean Plate (03 / Background)',
      clean_plate: 'Clean Plate (03 / Background)',
      backlit: 'Backlit Plate (X2 / Silhouette)',
      depth: 'Depth Plate'
    };

    // the target background "bg" is not a plate of the shot: never on the left, only as a comp background on the right
    const bgPattern = typeof plates.bg === 'string' ? plates.bg : '';
    if (bgPattern !== (this._bgPattern || '')) {   // new / changed / removed source: drop its cached frames
      for (const k of [...this.frameCache.keys()]) if (k.startsWith('bg_')) this.frameCache.delete(k);
      this._bgPattern = bgPattern;
    }
    let foundActive = false;
    for (const [key, pattern] of Object.entries(plates)) {
      if (!pattern || typeof pattern !== 'string' || key === 'bg') continue;
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = labels[key] || key;
      if (key === this.activePlate) {
        opt.selected = true;
        foundActive = true;
      }
      this.plateSelect.appendChild(opt);
      if (this.plateRefSelect) {
        const o2 = opt.cloneNode(true);
        o2.selected = key === this.compare.ref;
        this.plateRefSelect.appendChild(o2);
      }
    }

    if (!foundActive) {
      this.activePlate = this.plateSelect.value || 'beauty';
    }
    if (this.plateRefSelect) {
      const grp = document.createElement('optgroup');
      grp.label = 'Tło compu (wycięcie z beauty)';
      const oc = document.createElement('option');
      oc.value = '__color'; oc.textContent = 'Tło: białe / kolor / szachownica';
      grp.appendChild(oc);
      const ob = document.createElement('option');
      ob.value = 'bg';
      ob.textContent = bgPattern ? 'BG – tło docelowe' : 'BG – tło docelowe (ustaw…)';
      grp.appendChild(ob);
      this.plateRefSelect.appendChild(grp);
      if (![...this.plateRefSelect.options].some((o) => o.value === this.compare.ref && !o.disabled)) {
        this.compare.ref = 'beauty';
        if (this.compare.mode === 'comp') this.compare.mode = 'none';
      }
      this.plateRefSelect.value = this.compare.ref;
    }
    this.syncCompareUi();
  }
  bgRefLabel() {
    if (this.compare.ref === 'bg') return 'BG';
    return this.compare.bg.checker ? 'szachownica' : (this.compare.bg.color.toLowerCase() === '#ffffff' ? 'białe' : this.compare.bg.color);
  }

  preloadNearby(centerFrame) {
    const range = this.isPlaying ? 15 : 6;
    for (let offset = 1; offset <= range; offset++) {
      const fForward = centerFrame + offset;
      if (fForward <= this.lastFrame) this.loadFrame(fForward);
      const fBack = centerFrame - offset;
      if (fBack >= this.firstFrame) this.loadFrame(fBack);
    }
  }

  // --------------------------------------------------------------------------
  // Playback Control (High-Performance 24 fps)
  // --------------------------------------------------------------------------
  togglePlay() {
    if (this.isPlaying) {
      this.stopPlayback();
    } else {
      this.startPlayback();
    }
  }

  playRange() {
    const wa = window.MD_WA && window.MD_WA.range;
    const lo = wa ? Math.max(this.firstFrame, wa[0]) : this.firstFrame;
    const hi = wa ? Math.min(this.lastFrame, wa[1]) : this.lastFrame;
    return lo <= hi ? [lo, hi] : [this.firstFrame, this.lastFrame];
  }

  startPlayback() {
    this.isPlaying = true;
    const [lo, hi] = this.playRange();
    if (this.currentFrame < lo || this.currentFrame > hi) this.goToFrame(lo, 1);
    this.btnPlayPause.textContent = '⏸';
    this.btnPlayPause.classList.add('playing');
    this.lastFrameTimestamp = performance.now();

    const frameDuration = 1000.0 / this.fps;

    const playLoop = (now) => {
      if (!this.isPlaying) return;

      const elapsed = now - this.lastFrameTimestamp;
      if (elapsed >= frameDuration) {
        // play loops inside the work area [ ] only (user request 2026-10-16); no work area = the whole shot
        const [lo, hi] = this.playRange();
        let next = this.currentFrame + 1;
        if (next > hi || next < lo) next = lo;
        // Z20: like Nuke, wait for the next frame (plate + overlay of the view) instead of showing it bare; at most 2 s
        if (this.frameReady && !this.frameReady(next)) {
          this.loadFrame(next); this.preloadNearby(next);
          if (!this.playWaitSince) this.playWaitSince = now;
          if (now - this.playWaitSince < 2000) { this.playAnimationId = requestAnimationFrame(playLoop); return; }
        }
        this.playWaitSince = 0;
        this.lastFrameTimestamp = now - (elapsed % frameDuration);
        this.goToFrame(next, 1);
      }

      this.playAnimationId = requestAnimationFrame(playLoop);
    };

    this.playAnimationId = requestAnimationFrame(playLoop);
  }

  stopPlayback() {
    this.isPlaying = false;
    this.btnPlayPause.textContent = '▶';
    this.btnPlayPause.classList.remove('playing');
    if (this.playAnimationId) {
      cancelAnimationFrame(this.playAnimationId);
    }
  }

  // --------------------------------------------------------------------------
  // Viewport Render
  // --------------------------------------------------------------------------
  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Clear background
    ctx.fillStyle = '#121214';
    ctx.fillRect(0, 0, w, h);

    const img = this.getCurrentImage();
    if (!img) {
      // Draw placeholder
      const failed = this.frameErrors && this.frameErrors.has(this.frameKey(this.activePlate || 'beauty', this.currentFrame));
      ctx.fillStyle = '#1f1f22';
      ctx.fillRect(w / 4, h / 4, w / 2, h / 2);
      ctx.fillStyle = failed ? '#fbbf24' : '#9a9aa0';
      ctx.font = '14px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(failed ? `Brak klatki ${this.currentFrame} plate'u „${this.activePlate}”${this.alignedPreview ? ' (po dopasowaniu – poza zakresem pomiaru?)' : ''}`
                          : `Wczytywanie klatki ${this.currentFrame}...`, w / 2, h / 2);
      return;
    }

    if (this.zoom <= 0 || isNaN(this.zoom)) {
      this.zoom = 1.0;
    }

    ctx.save();
    // Apply pan & zoom
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // Disable smoothing when zoomed in past 150% for crisp pixel review
    ctx.imageSmoothingEnabled = this.zoom < 1.5;

    ctx.drawImage(img, 0, 0);
    this.drawTiles(ctx, img);
    this.drawCompare(ctx, img);

    ctx.restore();
    this.updateHudZoom();
    if (this._cmpNote && (this.compareCoversView() || this.compare.mode === 'comp')) {
      ctx.font = '13px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'left';
      const tw = ctx.measureText(this._cmpNote).width;
      ctx.fillStyle = 'rgba(20,16,4,0.75)'; ctx.fillRect(8, h - 34, tw + 16, 26);
      ctx.fillStyle = '#fbbf24'; ctx.fillText(this._cmpNote, 16, h - 16);
    }
    if (this.alignedPreview && this.alignedFallback && this.alignedFallback.has(this.frameKey(this.activePlate || 'beauty', this.currentFrame))) {
      ctx.fillStyle = 'rgba(20,16,4,0.75)'; ctx.fillRect(8, 8, Math.min(w - 16, 560), 26);
      ctx.fillStyle = '#fbbf24'; ctx.font = '13px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(`F${this.currentFrame} poza zakresem pomiaru – plate „${this.activePlate}” bez dopasowania`, 16, 26);
    }
  }

  // ---- Z8: full-resolution tiles -------------------------------------------------------------------------------
  plateScale(img) {   // plate px per proxy px (kp); 1 when the proxy IS the plate
    const pw = this.projectData && this.projectData.plate_w;
    return img && pw && img.width ? pw / img.width : 1;
  }
  tilesWanted(img) {
    img = img || this.getCurrentImage();
    if (!img || !this.projectLoaded || !this.projectData) return false;
    if (this.compare.mode !== 'none' || ((this.activePlate || 'beauty') !== 'beauty' && this.alignedPreview)) return false;   // comparisons / "po dopasowaniu": proxy only
    return this.zoom > 1.0 && this.plateScale(img) > 1.0;
  }
  // visible area in plate px + 10 % margin, rounded to 64 px, clamped to the plate; split into chunks <= TILE_MAX
  tileRegionFor(img) {
    const kp = this.plateScale(img);
    const pw = this.projectData.plate_w, ph = this.projectData.plate_h;
    const vx0 = (-this.panX) / this.zoom, vy0 = (-this.panY) / this.zoom;
    const vx1 = (this.canvas.width - this.panX) / this.zoom, vy1 = (this.canvas.height - this.panY) / this.zoom;
    const mw = (vx1 - vx0) * 0.1, mh = (vy1 - vy0) * 0.1;
    const r64 = (v, up) => (up ? Math.ceil(v / 64) * 64 : Math.floor(v / 64) * 64);
    const x0 = Math.max(0, r64((vx0 - mw) * kp, false)), y0 = Math.max(0, r64((vy0 - mh) * kp, false));
    const x1 = Math.min(pw, r64((vx1 + mw) * kp, true)), y1 = Math.min(ph, r64((vy1 + mh) * kp, true));
    if (x1 <= x0 || y1 <= y0) return null;
    const chunks = [];
    for (let y = y0; y < y1; y += this.TILE_MAX) for (let x = x0; x < x1; x += this.TILE_MAX) {
      const w = Math.min(this.TILE_MAX, x1 - x), h = Math.min(this.TILE_MAX, y1 - y);
      chunks.push({ x, y, w, h });
    }
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, kp, chunks, plate: this.activePlate || 'beauty', f: this.currentFrame };
  }
  tileKey(c, plate, f) { return `${plate}_${f}_${c.x}_${c.y}_${c.w}_${c.h}`; }
  // the current region when its tiles are wanted – { x, y, w, h, kp, chunks: [{x,y,w,h,img|null}] } or null (objects.js / zones.js
  // draw their full-resolution overlays on the same chunks; `complete` = every chunk loaded)
  tileRegion() {
    const img = this.getCurrentImage();
    if (!this.tilesWanted(img)) return null;
    const reg = this.tileRegionFor(img);
    if (!reg) return null;
    for (const c of reg.chunks) { const t = this.tiles.cache.get(this.tileKey(c, reg.plate, reg.f)); c.img = t && t.complete && t.naturalWidth > 0 ? t : null; }
    reg.complete = reg.chunks.every((c) => c.img);
    return reg;
  }
  scheduleTiles() {
    clearTimeout(this.tiles.timer);
    this.tiles.timer = setTimeout(() => this.loadTiles(), 150);
  }
  loadTiles() {
    const img = this.getCurrentImage();
    if (!this.tilesWanted(img)) return;
    const reg = this.tileRegionFor(img);
    if (!reg) return;
    for (const c of reg.chunks) {
      const key = this.tileKey(c, reg.plate, reg.f);
      if (this.tiles.cache.has(key)) continue;
      const t = new Image();
      const gen = this.projectGen || 0;
      t.crossOrigin = 'anonymous';
      t.onload = () => { if (gen === (this.projectGen || 0) && reg.f === this.currentFrame) this.render(); };
      t.onerror = () => { if (gen === (this.projectGen || 0)) this.tiles.cache.delete(key); };
      t.src = `/api/frame/${reg.f}/tile?plate=${encodeURIComponent(reg.plate)}&x=${c.x}&y=${c.y}&w=${c.w}&h=${c.h}&g=${gen}`;
      this.tiles.cache.set(key, t);
      while (this.tiles.cache.size > this.tiles.limit) this.tiles.cache.delete(this.tiles.cache.keys().next().value);
    }
  }
  drawTiles(ctx, img) {
    this.tiles.region = null;
    if (!this.tilesWanted(img)) return;
    const reg = this.tileRegion();
    if (!reg) return;
    this.tiles.region = reg;
    const smooth = ctx.imageSmoothingEnabled;
    ctx.imageSmoothingEnabled = this.zoom < reg.kp * 1.5;   // a tile is at plate resolution: crisp pixels only past 150 % of the plate
    for (const c of reg.chunks) if (c.img) ctx.drawImage(c.img, c.x / reg.kp, c.y / reg.kp, c.w / reg.kp, c.h / reg.kp);
    ctx.imageSmoothingEnabled = smooth;
    if (!reg.complete) this.scheduleTiles();
  }
  updateHudZoom() {
    if (!this.hudZoom) return;
    const base = `Zoom: ${Math.round(this.zoom * 100)}%`;
    const reg = this.tiles.region;
    if (!reg) { this.hudZoom.textContent = base; return; }
    const pct = Math.round(100 * this.zoom / reg.kp);   // zoom in plate px
    this.hudZoom.textContent = `${base} · ${reg.complete ? `${pct} % 4K` : 'ładowanie 4K…'}`;
  }

  // Comparison with beauty (plate space, inside the pan/zoom transform). A/B = beauty while C is held;
  // wipe = beauty left of the line; diff = |beauty - aux| x gain; edges = beauty contours over the aux plate.
  drawCompare(ctx, img) {
    const ref = this.compare.ref;
    this._cmpNote = '';
    if (this.isBgRef()) { this.drawBgCompare(ctx, img); return; }
    if ((this.activePlate || 'beauty') === ref) return;
    const mode = this.compare.mode;
    if (mode === 'none' && !this.compare.holdC) return;
    const beauty = this.getImage(ref, this.currentFrame, false);   // the right-hand plate of the pair (default beauty)
    if (!beauty) { this.loadFrame(this.currentFrame, null, ref, false); return; }
    const W = img.width, H = img.height;
    // A/B: the aux plate stays on screen, beauty only while C is held (was: beauty always in A/B mode)
    if (this.compare.holdC) { ctx.drawImage(beauty, 0, 0, W, H); return; }
    if (mode === 'ab') return;
    if (mode === 'wipe') {
      const x = W * this.compare.wipe;
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, x, H); ctx.clip(); ctx.drawImage(beauty, 0, 0, W, H); ctx.restore();
      ctx.strokeStyle = '#ffcc33'; ctx.lineWidth = 2 / this.zoom; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.fillStyle = '#ffcc33'; ctx.font = `${12 / this.zoom}px "IBM Plex Sans", sans-serif`; ctx.textAlign = 'left';
      ctx.fillText(ref, 6 / this.zoom, 16 / this.zoom); ctx.textAlign = 'right'; ctx.fillText(this.activePlate, W - 6 / this.zoom, 16 / this.zoom);
      return;
    }
    if (mode === 'diff') {
      const c = this.scratch(W, H); const g = c.getContext('2d');
      g.globalCompositeOperation = 'source-over'; g.drawImage(beauty, 0, 0, W, H);
      g.globalCompositeOperation = 'difference'; g.drawImage(img, 0, 0, W, H);
      g.globalCompositeOperation = 'lighter';
      for (let i = 1; i < this.compare.gain; i++) g.drawImage(c, 0, 0);
      g.globalCompositeOperation = 'source-over';
      ctx.drawImage(c, 0, 0);
      return;
    }
    if (mode === 'edges') {
      const key = `${ref}_${this.currentFrame}_${W}`;
      let e = this.edgeCache.get(key);
      if (!e) { e = this.edgeMap(beauty, W, H); this.edgeCache.set(key, e); if (this.edgeCache.size > 40) this.edgeCache.delete(this.edgeCache.keys().next().value); }
      ctx.drawImage(e, 0, 0);
    }
  }
  // Right side = a comp background: the cut-out (beauty RGB x alpha of the selected object – final alpha, else the
  // core / key) over white / own colour / checker or over the target background plate "bg". Plate space, W x H.
  drawBgCompare(ctx, img) {
    const mode = this.compare.mode;
    if (mode === 'none' && !this.compare.holdC) return;
    if (mode === 'ab' && !this.compare.holdC) return;
    const W = img.width, H = img.height;
    const comp = this.bgComp(W, H);
    if (!comp) return;
    if (this.compare.holdC || mode === 'comp') { ctx.drawImage(comp, 0, 0, W, H); return; }
    if (mode === 'wipe') {
      const x = W * this.compare.wipe;
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, x, H); ctx.clip(); ctx.drawImage(comp, 0, 0, W, H); ctx.restore();
      ctx.strokeStyle = '#ffcc33'; ctx.lineWidth = 2 / this.zoom; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.fillStyle = '#ffcc33'; ctx.font = `${12 / this.zoom}px "IBM Plex Sans", sans-serif`; ctx.textAlign = 'left';
      ctx.fillText(`na tle: ${this.bgRefLabel()}`, 6 / this.zoom, 16 / this.zoom); ctx.textAlign = 'right'; ctx.fillText(this.activePlate, W - 6 / this.zoom, 16 / this.zoom);
    }
  }
  bgComp(W, H) {
    const f = this.currentFrame;
    const beauty = this.getImage('beauty', f, false);
    if (!beauty) { this.loadFrame(f, null, 'beauty', false); this._cmpNote = 'Wczytywanie beauty…'; return null; }
    const bgSpec = this.compare.ref === 'bg' ? 'bg' : (this.compare.bg.checker ? 'checker' : this.compare.bg.color);
    let bgImg = null;
    if (bgSpec === 'bg') {
      bgImg = this.getImage('bg', f, false);
      if (!bgImg) {
        const failed = this.frameErrors && this.frameErrors.has(this.frameKey('bg', f, false));
        if (!failed) { this.loadFrame(f, null, 'bg', false); this._cmpNote = 'Wczytywanie tła BG…'; return null; }
        this._cmpNote = `Brak klatki tła BG dla F${f} – sprawdź ścieżkę w ⚙️ Plate'y`;
        return null;
      }
    }
    const obj = window.MD_OBJ;
    const m = obj && obj.matteFor ? obj.matteFor(f) : null;
    if (m === 'loading') { this._cmpNote = 'Wczytywanie maski…'; return null; }
    if (!m) this._cmpNote = obj && obj.name ? (obj.matteNote || `„${obj.name}”: brak alpha i rdzenia w F${f} – samo tło`) : 'Wybierz obiekt (lista Obiekt) – bez maski widać samo tło';
    else {   // which mask the cut-out uses (it follows the selected stage, objects.js matteFor)
      const src = { core: 'rdzeń SAM3', key: 'klucz SAM3', core_fix: 'maska 1+2' }[m.kind] || (m.run ? `wersja ${m.run}` : 'alpha');
      this._cmpNote = `„${m.group || obj.name}”${m.group ? ' (grupa ze szwem)' : ''} F${f}: wycięcie – ${src}`;
    }
    // cached per frame / matte / background; the image objects are compared too (a recomputed mask = a new Image)
    const key = `${f}|${m ? m.key : '-'}|${bgSpec}|${W}x${H}`;
    this._bgCompCache = this._bgCompCache || new Map();
    const hit = this._bgCompCache.get(key);
    if (hit && hit.beauty === beauty && hit.matte === (m && m.img) && hit.bgImg === bgImg) return hit.c;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d');
    if (bgSpec === 'checker') {
      for (let y = 0; y < H; y += 16) for (let x = 0; x < W; x += 16) { g.fillStyle = ((x + y) / 16) % 2 ? '#999' : '#666'; g.fillRect(x, y, 16, 16); }
    } else if (bgSpec === 'bg') g.drawImage(bgImg, 0, 0, W, H);
    else { g.fillStyle = bgSpec; g.fillRect(0, 0, W, H); }
    if (m) {
      const b = g.getImageData(0, 0, W, H);
      const t = document.createElement('canvas'); t.width = W; t.height = H;
      const tg = t.getContext('2d');
      tg.drawImage(beauty, 0, 0, W, H); const p = tg.getImageData(0, 0, W, H);
      tg.clearRect(0, 0, W, H); tg.drawImage(m.img, 0, 0, W, H); const a = tg.getImageData(0, 0, W, H);
      const bd = b.data, pd = p.data, ad = a.data;
      for (let i = 0; i < bd.length; i += 4) {
        const al = ad[i] / 255;
        if (al <= 0) continue;
        bd[i] = pd[i] * al + bd[i] * (1 - al); bd[i + 1] = pd[i + 1] * al + bd[i + 1] * (1 - al); bd[i + 2] = pd[i + 2] * al + bd[i + 2] * (1 - al);
      }
      g.putImageData(b, 0, 0);
    }
    this._bgCompCache.delete(key);
    this._bgCompCache.set(key, { c, beauty, matte: m && m.img, bgImg });
    while (this._bgCompCache.size > 16) this._bgCompCache.delete(this._bgCompCache.keys().next().value);
    return c;
  }
  scratch(W, H) {
    if (!this._scratch) this._scratch = document.createElement('canvas');
    if (this._scratch.width !== W || this._scratch.height !== H) { this._scratch.width = W; this._scratch.height = H; }
    else this._scratch.getContext('2d').clearRect(0, 0, W, H);
    return this._scratch;
  }
  edgeMap(src, W, H) {
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d'); g.drawImage(src, 0, 0, W, H);
    const d = g.getImageData(0, 0, W, H), p = d.data, out = g.createImageData(W, H), o = out.data;
    const lum = new Float32Array(W * H);
    for (let i = 0, j = 0; i < p.length; i += 4, j++) lum[j] = 0.299 * p[i] + 0.587 * p[i + 1] + 0.114 * p[i + 2];
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx = -lum[i - W - 1] - 2 * lum[i - 1] - lum[i + W - 1] + lum[i - W + 1] + 2 * lum[i + 1] + lum[i + W + 1];
      const gy = -lum[i - W - 1] - 2 * lum[i - W] - lum[i - W + 1] + lum[i + W - 1] + 2 * lum[i + W] + lum[i + W + 1];
      const m = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 0.6);
      if (m > 40) { const k = i * 4; o[k] = 80; o[k + 1] = 255; o[k + 2] = 120; o[k + 3] = m; }
    }
    g.putImageData(out, 0, 0);
    return c;
  }

  fitToScreen() {
    this.resizeCanvas();
    const canvasW = this.canvas.width;
    const canvasH = this.canvas.height;
    if (canvasW <= 50 || canvasH <= 50) return;

    const img = this.getCurrentImage();
    if (!img || !img.width || !img.height) return;

    const imgW = img.width;
    const imgH = img.height;

    const availableW = Math.max(10, canvasW - 40);
    const availableH = Math.max(10, canvasH - 40);
    const scale = Math.max(0.01, Math.min(availableW / imgW, availableH / imgH, 1.0));
    this.zoom = scale;
    this.panX = Math.round((canvasW - imgW * scale) / 2);
    this.panY = Math.round((canvasH - imgH * scale) / 2);

    this.hasInitialFit = true;
    this.hudZoom.textContent = `Zoom: ${Math.round(this.zoom * 100)}%`;
    this.render();
  }

  setZoom100() {
    const img = this.getCurrentImage();
    if (!img) return;

    const canvasW = this.canvas.width;
    const canvasH = this.canvas.height;
    const imgW = img.width || 1920;
    const imgH = img.height || 1080;

    this.zoom = 1.0;
    this.panX = Math.round((canvasW - imgW) / 2);
    this.panY = Math.round((canvasH - imgH) / 2);

    this.hudZoom.textContent = 'Zoom: 100%';
    this.render();
  }

  // --------------------------------------------------------------------------
  // Doctor & Diagnostics
  // --------------------------------------------------------------------------
  // Server running older code than on disk (started before a git pull / update): the API lacks new endpoints and the
  // Doctor shows wrong data -> a red banner asking to restart the MatteDesk console window.
  async checkServerVersion() {
    let msg = '';
    try {
      const r = await fetch('/api/version');
      if (!r.ok) {
        msg = 'Serwer MatteDesk działa na starym kodzie (sprzed aktualizacji).';
      } else {
        const d = await r.json();
        const el = document.getElementById('appVersion');
        if (el) el.textContent = `v${d.version}`;
        document.title = `${document.title.replace(/ ·? ?v[\d.]+$/, '')} · v${d.version}`;
        if (d.stale || (d.on_disk && d.on_disk !== d.version)) msg = `Serwer MatteDesk działa na starszym kodzie niż na dysku (v${d.version}${d.on_disk && d.on_disk !== d.version ? ' → v' + d.on_disk : ''}).`;
      }
    } catch (e) { return; }
    if (!msg) return;
    let b = document.getElementById('staleServerBanner');
    if (!b) {
      b = document.createElement('div');
      b.id = 'staleServerBanner';
      // bottom of the window: never covers the header buttons (Doktor / Projekt)
      b.style.cssText = 'position:fixed;bottom:0;left:0;right:0;z-index:20000;background:#7f1d1d;color:#fff;padding:10px 44px 10px 16px;font-size:14px;font-weight:600;text-align:center;box-shadow:0 -2px 12px rgba(0,0,0,.5);';
      document.body.appendChild(b);
    }
    b.innerHTML = `⚠ ${msg} Zamknij MatteDesk i uruchom go ponownie: skrót z menu Start / pulpitu sam zaproponuje zatrzymanie starego serwera (kopia z gita: zamknij okno konsoli i uruchom <code>start_v….bat</code>).`
      + '<button type="button" title="Ukryj" style="position:absolute;right:10px;top:50%;transform:translateY(-50%);background:none;border:0;color:#fff;font-size:18px;cursor:pointer;">✕</button>';
    b.querySelector('button').addEventListener('click', () => b.remove());
    this.serverStale = msg;
  }

  async checkDoctorStatus() {
    // ComfyUI being started by MatteDesk: show it and check again, do not judge the environment yet
    try {
      const ls = await (await fetch('/api/comfy/launch')).json();
      if (ls.state === 'starting') {
        this.comfyStatusText.textContent = `ComfyUI: uruchamiam (${ls.how || ''})…`;
        this.comfyStatusDot.classList.remove('online');
        clearTimeout(this._comfyWaitT);
        this._comfyWaitT = setTimeout(() => this.checkDoctorStatus(), 3000);
        return;
      }
      if (ls.state === 'failed' && !this._launchFailShown) {
        this._launchFailShown = true;
        this.uiDialog({ title: 'Nie udało się uruchomić ComfyUI', tone: 'error', html: this.esc(ls.error || '') });
      }
    } catch (e) { /* older server */ }
    try {
      const res = await fetch('/api/doctor');
      const data = await res.json();
      const comfyItem = data.items ? data.items.find((it) => it.name === 'ComfyUI Service') : null;

      // Startup: something required is missing -> open the Doctor once (with the "Zainstaluj" buttons)
      if (!this._doctorAutoShown && data.summary && data.summary.error > 0) {
        this._doctorAutoShown = true;
        this.doctorModal.style.display = 'flex';
        this.fetchDoctorReport();
      }

      let portStr = '';
      if (comfyItem && comfyItem.details && comfyItem.details.url) {
        try {
          const u = new URL(comfyItem.details.url);
          if (u.port) portStr = `:${u.port}`;
        } catch (e) {}
      } else if (data.config && data.config.comfy_port) {
        portStr = `:${data.config.comfy_port}`;
      }

      if (comfyItem && comfyItem.status === 'ok') {
        this.comfyStatusText.textContent = `ComfyUI ${portStr} [Online]`;
        this.comfyStatusDot.classList.add('online');
      } else {
        const portOnly = portStr ? portStr.replace(':', '') : '';
        this.comfyStatusText.textContent = `ComfyUI: Offline ${portOnly ? '(' + portOnly + ')' : ''}`;
        this.comfyStatusDot.classList.remove('online');
      }
    } catch (e) {
      this.comfyStatusText.textContent = 'Backend: Rozłączony';
    }
  }

  showInitError(step, err) {
    const b = document.createElement('div');
    b.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:20001;background:#7f1d1d;color:#fff;padding:10px 16px;font-size:13px;font-weight:600;';
    b.textContent = `⚠ Błąd uruchamiania strony (${step}): ${err && err.message ? err.message : err}. Odśwież stronę przez Ctrl+F5; jeśli zostaje – skopiuj ten tekst do Claude.`;
    document.body.appendChild(b);
  }

  // ---- in-app dialogs (no window.confirm / alert: Chrome lets the user block those, then buttons silently do nothing)
  esc(t) { return String(t == null ? '' : t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  restartNote() {
    return '<div style="margin-top:10px;padding:10px 12px;border:2px solid #f0703a;border-radius:6px;background:rgba(240,112,58,.14);">'
      + '<b style="color:#f0703a;">🔁 Po instalacji węzłów trzeba zrestartować ComfyUI</b><br>'
      + '<span style="font-size:12px;">Bez restartu ComfyUI nie załaduje nowych węzłów. Doktor pokaże wtedy przycisk „Zrestartuj ComfyUI”.</span></div>';
  }

  uiDialog({ title, html = '', ok = 'OK', cancel = null, tone = 'info' }) {
    return new Promise((resolve) => {
      const col = { info: '#e8a33d', ok: '#10b981', warn: '#f0703a', error: '#ef4444' }[tone] || '#e8a33d';
      const bg = document.createElement('div');
      bg.className = 'md-dialog';
      bg.style.cssText = 'position:fixed;inset:0;z-index:25000;background:rgba(10,10,12,.7);display:flex;align-items:center;justify-content:center;';
      bg.innerHTML = `<div role="dialog" style="width:min(560px,92vw);max-height:80vh;display:flex;flex-direction:column;background:#19191b;color:#dcdcdf;border:1px solid #3c3c42;border-top:4px solid ${col};border-radius:10px;box-shadow:0 20px 50px rgba(0,0,0,.6);">
          <div style="padding:14px 18px 6px;font-size:16px;font-weight:700;">${this.esc(title)}</div>
          <div style="padding:6px 18px 12px;font-size:13px;line-height:1.5;overflow:auto;">${html}</div>
          <div style="padding:12px 18px;border-top:1px solid #2a2a2e;display:flex;gap:8px;justify-content:flex-end;">
            ${cancel ? `<button type="button" class="btn btn-secondary" data-r="0">${this.esc(cancel)}</button>` : ''}
            <button type="button" class="btn btn-primary" data-r="1" style="background:${col};border-color:${col};">${this.esc(ok)}</button>
          </div></div>`;
      const onKey = (e) => {
        if (e.key === 'Escape') { e.stopPropagation(); done(false); }
        if (e.key === 'Enter') { e.stopPropagation(); done(true); }
      };
      const done = (v) => { document.removeEventListener('keydown', onKey, true); bg.remove(); resolve(v); };
      bg.querySelectorAll('button[data-r]').forEach((b) => b.addEventListener('click', () => done(b.dataset.r === '1')));
      bg.addEventListener('click', (e) => { if (e.target === bg && cancel) done(false); });
      document.addEventListener('keydown', onKey, true);
      document.body.appendChild(bg);
      bg.querySelector('button[data-r="1"]').focus();
    });
  }

  // In-app replacement for window.confirm() (Chrome may block native dialogs): plain text, first line = title when no title given
  uiConfirm(msg, { title = null, ok = 'OK', cancel = 'Anuluj', tone = 'warn' } = {}) {
    let body = String(msg);
    if (!title) { const i = body.indexOf('\n'); title = i < 0 ? body : body.slice(0, i); body = i < 0 ? '' : body.slice(i + 1).replace(/^\n+/, ''); }
    return this.uiDialog({ title, ok, cancel, tone, html: this.esc(body).replace(/\n/g, '<br>') });
  }

  // Result of one or more installs: ✅ / ❌ per step, short message, raw log collapsed, restart reminder when needed
  showInstallResults(results) {
    const names = { nodes: 'Węzły MatteDesk (PlateMatte, MatAnyone2-Keyframes)', matanyone2_node: 'Węzeł ComfyUI-MatAnyone2',
      inpaint_nodes: 'Węzły inpaint (LaMa)', depthcrafter_nodes: 'Węzły DepthCrafter', minimax_nodes: 'Węzły MiniMax-Remover', vhs_nodes: 'Węzły VideoHelperSuite', minimax_weights: 'Wagi MiniMax-Remover', sam3: 'Wagi SAM3.1', matanyone2_weights: 'Wagi MatAnyone2', lama: 'Wagi LaMa',
      cotracker: 'CoTracker3', vitmatte: 'ViTMatte', ltx: 'Modele LTX-2.5 Alpha-Gen', videomama: 'VideoMaMa (węzeł + wagi)', restart_comfy: 'Restart ComfyUI' };
    const allOk = results.every((r) => r.ok);
    const needsRestart = results.some((r) => r.ok && ['nodes', 'matanyone2_node', 'inpaint_nodes', 'depthcrafter_nodes', 'minimax_nodes', 'vhs_nodes', 'videomama'].includes(r.what));
    const rows = results.map((r) => {
      const main = (r.lines || []).filter((l) => l && !/^Zrestartuj ComfyUI/.test(l));
      const pick = (re) => main.filter((l) => re.test(l)).join(' · ');
      const short = r.ok ? (pick(/-> OK|pobrane|skopiowany|zrestartowany|już jest/i) || pick(/aktualne/i) || 'Gotowe') : (main.find((l) => /BŁĄD|błąd|kod [1-9]|nie /i.test(l)) || main[0] || 'Błąd');
      const det = (main.length > 1 || !r.ok)
        ? `<details style="margin-top:4px;font-size:11px;color:#9a9aa0;"><summary style="cursor:pointer;">szczegóły</summary><pre style="white-space:pre-wrap;margin:4px 0 0;">${this.esc(main.join('\n'))}</pre></details>`
        : '';
      return `<div style="padding:8px 0;border-bottom:1px solid #2a2a2e;">
        <div style="font-weight:600;">${r.ok ? '✅' : '❌'} ${this.esc(names[r.what] || r.what)}</div>
        <div style="font-size:12px;color:${r.ok ? '#9a9aa0' : '#fca5a5'};margin-top:2px;">${this.esc(String(short).slice(0, 300))}</div>${det}
      </div>`;
    }).join('');
    return this.uiDialog({ title: allOk ? 'Gotowe' : 'Instalacja z błędami', tone: allOk ? (needsRestart ? 'warn' : 'ok') : 'error',
      html: rows + (needsRestart ? this.restartNote() : '') });
  }

  // Runs POST /api/install/<what> and shows a progress bar (step / file, MB, speed, ETA) polled from /api/install/progress
  async installWithProgress(what) {
    let bar = document.getElementById('installBar');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'installBar';
      bar.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:10000;width:min(560px,92vw);background:#19191b;border:1px solid #e8a33d;border-radius:8px;padding:12px 14px;font-size:12px;color:#dcdcdf;box-shadow:0 8px 24px rgba(0,0,0,.5);';
      document.body.appendChild(bar);
    }
    bar.style.display = 'block';
    let last = { file: '', t: 0, done: 0 }, speed = 0, stop = false;
    const fmt = (b) => (b >= 1073741824 ? (b / 1073741824).toFixed(2) + ' GB' : (b / 1048576).toFixed(0) + ' MB');
    const draw = (p) => {
      const pct = p.total ? Math.min(100, (100 * p.done) / p.total) : 0;
      let line = `<b>${what}</b> · ${p.step || '…'}${p.file ? ' · ' + p.file : ''}`;
      if (p.total) {
        const eta = speed > 0 ? Math.max(0, (p.total - p.done) / speed) : 0;
        line += `<br>${fmt(p.done)} / ${fmt(p.total)} (${pct.toFixed(0)}%) · ${(speed / 1048576).toFixed(1)} MB/s${eta ? ' · ok. ' + (eta >= 90 ? Math.ceil(eta / 60) + ' min' : Math.ceil(eta) + ' s') : ''}`;
      }
      const fill = p.total ? `width:${pct}%` : 'width:100%;opacity:.45;animation:pulse 1.2s infinite';
      bar.innerHTML = `${line}<div style="margin-top:6px;height:8px;background:#2a2a2e;border-radius:4px;overflow:hidden;"><div style="height:100%;background:#e8a33d;${fill}"></div></div>`;
    };
    const poll = async () => {
      while (!stop) {
        try {
          const p = await (await fetch('/api/install/progress')).json();
          const now = performance.now();
          if (p.file !== last.file) last = { file: p.file, t: now, done: p.done || 0 };
          else if (now - last.t > 1500) { speed = ((p.done || 0) - last.done) / ((now - last.t) / 1000); last = { file: p.file, t: now, done: p.done || 0 }; }
          if (p.active) draw(p);
        } catch (e) { /* server busy */ }
        await new Promise((r) => setTimeout(r, 500));
      }
    };
    const polling = poll();
    try {
      const r = await fetch(`/api/install/${what}`, { method: 'POST' });
      const d = await r.json();
      return { ok: r.ok, data: d };
    } finally {
      stop = true; await polling; bar.style.display = 'none';
    }
  }

  async fetchDoctorReport() {
    this.doctorReportContent.innerHTML = '<div class="spinner"></div> Sprawdzanie środowiska...';
    try {
      const res = await fetch('/api/doctor');
      const data = await res.json();
      let html = `<div style="margin-bottom:12px; font-weight:600;">Wariant ComfyUI: ${data.comfy_flavor} | Status ogólny: ${data.all_ok ? '<span style="color:#10b981">OK</span>' : '<span style="color:#f0703a">Wymaga uwagi</span>'}</div>`;
      html += '<table style="width:100%; border-collapse:collapse; font-size:12px;">';
      const ACTIONS = {
        install_ffmpeg: ['ffmpeg', 'Wymagane teraz: TAK – wczytywanie ujęć wideo (MP4 / MOV) i podglądy. Instaluje ffmpeg przez winget (Gyan.FFmpeg, ok. 100 MB, akceptuje jego licencję w Twoim imieniu); gdy winget nie działa – pobiera paczkę portable z gyan.dev do bin/ffmpeg MatteDesk. Ścieżka trafia do config.local.json.'],
        install_git: ['git', 'Wymagane teraz: TAK – Doktor klonuje nim węzły ComfyUI z GitHuba (MatAnyone2, VideoHelperSuite, DepthCrafter) i instaluje CoTracker3. Instaluje Git for Windows przez winget (Git.Git, ok. 60 MB, akceptuje jego licencję w Twoim imieniu); Windows może zapytać o zgodę administratora.'],
        install_nodes: ['nodes', 'Kopiuje węzły PlateMatte i MatAnyone2-Keyframes do custom_nodes ComfyUI (stara wersja → backups/).'],
        install_cotracker: ['cotracker', 'Instaluje pakiet CoTracker3 w Pythonie ComfyUI (< 1 MB, PyTorch ComfyUI zostaje bez zmian) i pobiera wagi scaled_offline.pth (97 MB). CoTracker3 liczy się potem na GPU przez ComfyUI.'],
        install_vitmatte: ['vitmatte', 'Wymagane teraz: TAK – strefy z presetem „włosy” i silnik „vitmatte”. Pobiera wagi ViTMatte (hustvl/vitmatte-base-composition-1k, ~390 MB) do cache Hugging Face; transformers doinstaluje w Pythonie ComfyUI tylko, jeśli ich brak. ViTMatte liczy się potem na GPU w Pythonie ComfyUI.'],
        install_sam3: ['sam3', 'Pobiera wagi SAM3 (sam3.1_multiplex_fp16.safetensors, ~1,7 GB) z huggingface.co/Comfy-Org/sam3.1 do models/checkpoints.'],
        install_matanyone2_weights: ['matanyone2_weights', 'Pobiera wagi MatAnyone2 (matanyone2.pth, ~135 MB) z github.com/pq-yang/MatAnyone2 do models/matanyone2.'],
        install_matanyone2_node: ['matanyone2_node', 'Klonuje węzeł ComfyUI-MatAnyone2 (github.com/spiritform/comfy-matanyone2) do custom_nodes ComfyUI (jeśli go nie ma), instaluje jego wymagania i uruchamia jego install.py w Pythonie ComfyUI (pakiet matanyone2 z GitHub pq-yang/MatAnyone2, hydra-core, thinplate; kilka MB, PyTorch bez zmian).'],
        install_lama: ['lama', 'Pobiera big-lama.pt (~196 MB) do models/inpaint (opcjonalne).'],
        restart_comfy: ['restart_comfy', 'Restartuje ComfyUI (przez ComfyUI-Manager) i czeka, aż wstanie (zwykle 30–90 s). Niezapisane zmiany w oknie ComfyUI przepadną.'],
        install_depthcrafter_nodes: ['depthcrafter_nodes', 'Klonuje węzły DepthCrafter (github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes) do custom_nodes ComfyUI i instaluje ich wymagania. Wymagane teraz: nie (tylko funkcja „Głębia”). Do czego: mapa głębi ujęcia (etap 0). Model ok. 4–6 GB pobierze się sam przy 1. użyciu.'],
        install_minimax_nodes: ['minimax_nodes', 'Klonuje węzły MiniMax-Remover (github.com/1038lab/ComfyUI-MiniMax-Remover) do custom_nodes ComfyUI i instaluje ich wymagania. Wymagane teraz: nie – moduł Inpainting jest dopiero w planach. Do czego: usuwanie obiektów z wideo. Wagi 2,6 GB – osobny przycisk.'],
        install_minimax_weights: ['minimax_weights', 'Wymagane teraz: nie – moduł Inpainting jest dopiero w planach. Pobiera wagi MiniMax-Remover (2,6 GB: transformer 2,15 GB + vae 0,48 GB) z huggingface.co/zibojia/minimax-remover do models/MiniMax-Remover.'],
        install_vhs_nodes: ['vhs_nodes', 'Wymagane teraz: TAK – ujęcia 8-bitowe (PNG/JPG) w SAM3, MatAnyone2 i strefach (węzeł VHS_LoadImagesPath). Klonuje ComfyUI-VideoHelperSuite (github.com/Kosinkadink/ComfyUI-VideoHelperSuite, ok. 1 MB) do custom_nodes ComfyUI i instaluje jego wymagania w Pythonie ComfyUI (imageio-ffmpeg ok. 30 MB; opencv zwykle już jest).'],
        install_inpaint_nodes: ['inpaint_nodes', 'Klonuje comfyui-inpaint-nodes (github.com/Acly) do custom_nodes ComfyUI (opcjonalne).'],
        install_videomama: ['videomama', 'Wymagane teraz: nie – tylko etap 4 „VideoMaMa” (opcjonalny). Do czego: kosmyki włosów i detal krawędzi z maski 1+2. Klonuje węzeł ComfyUI-VideoMaMa (github.com/okdalto/ComfyUI-VideoMaMa) do custom_nodes ComfyUI i pobiera 7,1 GB wag (UNet VideoMaMa 6,1 GB z huggingface.co/SammyLim/VideoMaMa + image encoder i VAE SVD-xt 1,5 GB) do models/videomama. Po instalacji węzła – restart ComfyUI.'],
        install_ltx: ['ltx', 'Wymagane teraz: nie – tylko etap 7 „LTX Alpha-Gen” (opcjonalny). Do czego: generatywna alfa włosów, dymu, rozmycia ruchu. Pobiera ok. 40 GB (transformer int8 21,5 GB, Gemma int8 15,4 GB, VAE 1,8 GB, LoRA Alpha-Gen 1,3 GB) z huggingface.co/Lightricks do models/ (przerwane pobieranie wznawia się). Wymaga logowania do Hugging Face („hf auth login”) i „Agree and Access” na stronach Lightricks/LTX-2.5 i Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen.'],
      };
      const required = [];   // items with an install action -> "Zainstaluj wszystkie brakujące"
      let nRequired = 0;
      for (const it of data.items) {
        const color = it.status === 'ok' ? '#10b981' : it.status === 'warning' ? '#f0703a' : '#ef4444';
        const tag = it.status === 'ok' ? 'OK' : it.status === 'warning' ? 'UWAGA' : 'BŁĄD';
        const act = ACTIONS[it.action];
        const isRestart = act && act[0] === 'restart_comfy';
        // "Zainstaluj wszystkie": required (errors) AND optional (warnings) that have an install button,
        // not items only waiting for a ComfyUI restart
        // MiniMax-Remover: future Inpainting module, not used by the program yet -> only its own buttons
        // LTX-2.5: optional stage 6, ~40 GB behind a Hugging Face login -> only its own button
        const notInBatch = ['minimax_nodes', 'minimax_weights', 'ltx', 'videomama'];
        if (act && !isRestart && !notInBatch.includes(act[0]) && (it.status === 'error' || it.status === 'warning')
            && !required.some((r) => r[0] === act[0])) { required.push(act); if (it.status === 'error') nRequired++; }
        // third-party licence: the user downloads the component and accepts its licence (non-commercial ones highlighted)
        const licEsc = (it.license || '').replace(/"/g, '&quot;');
        const licAsk = it.license ? ` Licencja: ${licEsc}${it.license_nc ? ' (NIE do pracy komercyjnej)' : ''} – pobierając, akceptujesz ją.` : '';
        const licLine = !it.license ? '' : it.license_nc
          ? `<br><small style="color:#f87171;">⚠ Licencja: ${it.license} – nie do pracy komercyjnej. Pobierasz i akceptujesz ją sam (THIRD_PARTY_LICENSES.md).</small>`
          : `<br><small style="color:#9a9aa0;">Licencja: ${it.license}</small>`;
        const btn = it.action === 'start_comfy'
          ? `<br><button class="btn doctor-start-comfy" style="margin-top:4px;padding:2px 10px;font-size:12px;background:#10b981;color:#06281d;font-weight:700;border:0;">▶ Uruchom ComfyUI</button>`
          : !act ? '' : isRestart
          ? `<br><button class="btn doctor-install" data-what="restart_comfy" data-ask="${act[1]} Kontynuować?" style="margin-top:4px;padding:2px 10px;font-size:12px;background:#f0703a;color:#111;font-weight:700;border:0;">🔁 Zrestartuj ComfyUI</button>`
          : `<br><button class="btn btn-primary doctor-install" data-what="${act[0]}" data-ask="${act[1]}${licAsk} Potem ewentualnie restart ComfyUI. Kontynuować?" style="margin-top:4px;padding:2px 10px;font-size:12px;">Zainstaluj</button>`;
        html += `<tr style="border-bottom:1px solid #303035;">
          <td style="padding:6px; font-weight:600; color:${color}; width:60px;">[${tag}]</td>
          <td style="padding:6px; color:#9a9aa0; width:140px;">${it.name}</td>
          <td style="padding:6px;">${it.message}${licLine}${it.fix_suggestion ? `<br><small style="color:#f0703a;">Sugestia: ${it.fix_suggestion}</small>` : ''}${(it.links || []).map(l => { const m = l.match(/^(https?:\/\/\S+)(.*)$/); return `<br><small>Pobierz: ${m ? `<a href="${m[1]}" target="_blank" rel="noopener" style="color:#f2bb66;">${m[1]}</a>${m[2]}` : l}</small>`; }).join('')}${btn}</td>
        </tr>`;
      }
      html += '</table>';
      if (required.length) {
        const nOpt = required.length - nRequired;
        const col = nRequired ? '#ef4444' : '#e8a33d';
        const label = [nRequired ? `wymagane: ${nRequired}` : '', nOpt ? `opcjonalne: ${nOpt}` : ''].filter(Boolean).join(', ');
        html = `<div style="margin-bottom:12px;padding:10px;border:1px solid ${col};border-radius:6px;">
          <b style="color:${col};">Do zainstalowania – ${label}.</b>
          <button class="btn btn-primary doctor-install-all" style="margin-left:10px;padding:2px 12px;font-size:12px;">Zainstaluj wszystkie brakujące (${required.length})</button>
        </div>` + html;
      }
      // ComfyUI switch: Desktop <-> ComfyUI from git; installs/downloads go to the active one
      try {
        const cc = await (await fetch('/api/config/comfy')).json();
        const pf = cc.git || {};
        const isP = cc.flavor === 'git';
        const inp = 'style="background:#19191b;color:#dcdcdf;border:1px solid #3c3c42;border-radius:4px;padding:2px 6px;font-size:12px;"';
        html = `<div style="margin-bottom:12px;padding:10px;border:1px solid #3c3c42;border-radius:6px;font-size:12px;">
          <b>Używany ComfyUI:</b>
          <select id="comfyFlavorSel" ${inp}><option value="desktop"${isP ? '' : ' selected'}>ComfyUI Desktop</option><option value="git"${isP ? ' selected' : ''}>ComfyUI z gita / portable</option></select>
          <span id="comfyGitBox" style="display:${isP ? 'inline' : 'none'};margin-left:8px;">
            folder (z main.py): <input id="comfyPortDir" size="38" value="${(pf.comfy_dir || '').replace(/"/g, '&quot;')}" ${inp}>
            port: <input id="comfyPortPort" size="5" value="${pf.comfy_port || 8188}" ${inp}>
            <label><input type="checkbox" id="comfyPortShared"${pf.shared_models ? ' checked' : ''}> modele wspólne z Desktopem</label>
          </span>
          <label style="margin-left:10px;"><input type="checkbox" id="comfyAutostart"${cc.autostart === false ? '' : ' checked'}> uruchamiaj ComfyUI razem z MatteDesk</label>
          <label style="margin-left:10px;">okno MatteDesk: <select id="appBrowserSel" ${inp}><option value="default"${cc.app_browser === 'edge' ? '' : ' selected'}>domyślna przeglądarka</option><option value="edge"${cc.app_browser === 'edge' ? ' selected' : ''}>Edge</option></select></label>
          <label style="margin-left:10px;"><input type="checkbox" id="updCheckBox"${cc.update_check === false ? '' : ' checked'}> sprawdzaj aktualizacje MatteDesk przy starcie</label>
          <button class="btn btn-secondary" id="comfySaveBtn" style="margin-left:8px;padding:2px 10px;font-size:12px;">Zapisz</button>
          <div style="margin-top:6px;color:#9a9aa0;">Aktywny${cc.kind === 'portable' ? ' (ComfyUI portable)' : ''}: ${cc.comfy_dir || '(nie ustawiony)'} · modele: ${cc.models_dir || '-'} · port ${cc.comfy_port}. „Zainstaluj” działa na aktywnym.</div>
        </div>` + html;
      } catch (e) { /* config endpoint unavailable: report only */ }
      // ComfyUI restart needed: nodes installed but not loaded yet (or an install asked for it in this session)
      const nodeNames = ['ComfyUI-PlateMatte', 'ComfyUI-MatAnyone2-Keyframes', 'ComfyUI-MatAnyone2', 'Inpaint LaMa (comfyui-inpaint-nodes)', 'ComfyUI-DepthCrafter-Nodes', 'MiniMax-Remover (wideo)'];
      const waiting = data.items.filter((it) => it.details && it.details.needs_restart).map((it) => it.name);
      if (this.comfyRestartPending && data.items.filter((it) => nodeNames.includes(it.name)).every((it) => it.status === 'ok')) this.comfyRestartPending = false;
      if (waiting.length || this.comfyRestartPending) {
        html = `<div style="margin-bottom:12px;padding:12px;border:2px solid #f0703a;border-radius:6px;background:rgba(240,112,58,.14);">
          <div style="font-size:15px;font-weight:700;color:#f0703a;">🔁 Zrestartuj ComfyUI</div>
          <div style="margin-top:4px;">Nowe węzły i ich zależności działają dopiero po restarcie ComfyUI (Desktop: zamknij i otwórz aplikację).
          ${waiting.length ? 'Czekają na restart: <b>' + waiting.join(', ') + '</b>.' : ''} Po restarcie kliknij „Odśwież test”.</div>
          <button class="btn doctor-install" data-what="restart_comfy" data-ask="Restartuje ComfyUI (przez ComfyUI-Manager) i czeka, aż wstanie (zwykle 30–90 s). Niezapisane zmiany w oknie ComfyUI przepadną. Kontynuować?" style="margin-top:8px;padding:4px 14px;font-size:13px;background:#f0703a;color:#111;font-weight:700;border:0;">🔁 Zrestartuj ComfyUI teraz</button></div>` + html;
      }
      if (this.serverStale) {
        html = `<div style="margin-bottom:12px;padding:10px;border:2px solid #ef4444;border-radius:6px;background:rgba(239,68,68,.12);font-weight:600;">
          ⚠ ${this.serverStale} Raport może być niepełny (brak przycisków, „port undefined”). Zamknij okno konsoli MatteDesk i uruchom ponownie <code>start_v….bat</code>.</div>` + html;
      }
      this.doctorReportContent.innerHTML = html;
      const flSel = document.getElementById('comfyFlavorSel');
      if (flSel) {
        flSel.addEventListener('change', () => { document.getElementById('comfyGitBox').style.display = flSel.value === 'git' ? 'inline' : 'none'; });
        document.getElementById('comfySaveBtn').addEventListener('click', async () => {
          const body = { flavor: flSel.value, autostart: document.getElementById('comfyAutostart').checked };
          if (flSel.value === 'git') {
            body.comfy_dir = document.getElementById('comfyPortDir').value.trim();
            body.comfy_port = parseInt(document.getElementById('comfyPortPort').value, 10) || 8188;
            body.shared_models = document.getElementById('comfyPortShared').checked;
          }
          fetch('/api/config/app_browser', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ app_browser: document.getElementById('appBrowserSel').value }) });
          fetch('/api/update/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: document.getElementById('updCheckBox').checked }) });
          const r = await fetch('/api/config/comfy', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
          if (!r.ok) { await this.uiDialog({ title: 'Nie zapisano', tone: 'error', html: this.esc((await r.json()).detail) }); return; }
          this.fetchDoctorReport(); this.checkDoctorStatus();
        });
      }
      this.doctorReportContent.querySelectorAll('.doctor-start-comfy').forEach((b) => b.addEventListener('click', async () => {
        b.disabled = true; b.textContent = 'Uruchamiam ComfyUI…';
        const st = await (await fetch('/api/comfy/start', { method: 'POST' })).json();
        if (st.state === 'failed') { await this.uiDialog({ title: 'Nie udało się uruchomić ComfyUI', tone: 'error', html: this.esc(st.error || '') }); this.fetchDoctorReport(); return; }
        for (let i = 0; i < 120; i++) {   // up to ~4 min
          await new Promise((r) => setTimeout(r, 2000));
          const ls = await (await fetch('/api/comfy/launch')).json();
          b.textContent = `Uruchamiam ComfyUI… ${i * 2 + 2} s`;
          if (ls.state !== 'starting') {
            if (ls.state === 'failed') await this.uiDialog({ title: 'Nie udało się uruchomić ComfyUI', tone: 'error', html: this.esc(ls.error || '') });
            break;
          }
        }
        this.checkDoctorStatus(); this.fetchDoctorReport();
      }));
      const allBtn = this.doctorReportContent.querySelector('.doctor-install-all');
      if (allBtn) allBtn.addEventListener('click', async () => {
        const ok = await this.uiDialog({
          title: 'Zainstalować brakujące składniki?', ok: 'Zainstaluj', cancel: 'Anuluj',
          html: `<ol style="margin:0 0 10px 18px;padding:0;">${required.map((r) => `<li style="margin-bottom:6px;">${this.esc(r[1])}</li>`).join('')}</ol>`
            + this.restartNote() });
        if (!ok) return;
        allBtn.disabled = true;
        const results = [];
        for (const r of required) {
          allBtn.textContent = `Instaluję: ${r[0]}…`;
          try {
            const resp = await this.installWithProgress(r[0]);
            if (resp.ok && resp.data.restart_comfy) this.comfyRestartPending = true;
            results.push({ what: r[0], ok: resp.ok && resp.data.success !== false, lines: resp.ok ? (resp.data.log || []) : [resp.data.detail] });
          } catch (e) { results.push({ what: r[0], ok: false, lines: [String(e)] }); }
        }
        await this.showInstallResults(results);
        this.fetchDoctorReport();
      });
      this.doctorReportContent.querySelectorAll('.doctor-install').forEach((b) => b.addEventListener('click', async () => {
        const isRestart = b.dataset.what === 'restart_comfy';
        const go = await this.uiDialog({
          title: isRestart ? 'Zrestartować ComfyUI?' : 'Zainstalować?', ok: isRestart ? '🔁 Zrestartuj' : 'Zainstaluj', cancel: 'Anuluj',
          tone: isRestart ? 'warn' : 'info', html: this.esc(b.dataset.ask.replace(/ (Kontynuować|Continue)\?$/, '')) });
        if (!go) return;
        b.disabled = true; b.textContent = b.dataset.what === 'restart_comfy' ? 'Restartuję ComfyUI…' : 'Instaluję…';
        try {
          const r = await this.installWithProgress(b.dataset.what);
          if (r.ok && r.data.restart_comfy) this.comfyRestartPending = true;
          if (r.ok && r.data.restarted) this.comfyRestartPending = false;
          await this.showInstallResults([{ what: b.dataset.what, ok: r.ok && r.data.success !== false, lines: r.ok ? (r.data.log || []) : [r.data.detail] }]);
        } catch (e) { await this.showInstallResults([{ what: b.dataset.what, ok: false, lines: [String(e)] }]); }
        this.fetchDoctorReport();
      }));
    } catch (e) {
      this.doctorReportContent.innerHTML = `<span style="color:#ef4444;">Błąd pobierania raportu doktora: ${e}</span>`;
    }
  }

  // --------------------------------------------------------------------------
  // Plate Cleaning & Deflicker (Stage E1c, SPEC Section 4c)
  // --------------------------------------------------------------------------
  openCleanModal() {
    if (!this.projectLoaded) {
      alert('Najpierw otwórz projekt ujęcia.');
      return;
    }
    this.cleanModal.style.display = 'flex';
    this.populateCleanPlateSelect();
    this.checkExistingCleanReport();
  }

  populateCleanPlateSelect() {
    if (!this.cleanPlateSelect) return;
    this.cleanPlateSelect.innerHTML = '';
    const plates = (this.manifest && this.manifest.plates) || { beauty: 'beauty' };
    const labels = {
      beauty: 'Beauty Plate (X1)',
      cleanplate: 'Clean Plate (03 / Background)',
      clean_plate: 'Clean Plate (03 / Background)',
      backlit: 'Backlit Plate (X2 / Silhouette)',
      depth: 'Depth Plate'
    };
    for (const p of Object.keys(plates)) {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = labels[p] || p;
      this.cleanPlateSelect.appendChild(opt);
    }
    if (this.activePlate && plates[this.activePlate]) {
      this.cleanPlateSelect.value = this.activePlate;
    }
  }

  async checkExistingCleanReport() {
    const plate = this.cleanPlateSelect.value;
    try {
      const res = await fetch(`/api/plate/clean/report?plate=${encodeURIComponent(plate)}`);
      const data = await res.json();
      if (data.exists && data.report) {
        this.displayCleanReport(data.report);
      } else {
        this.resetCleanModalUI();
      }
    } catch (e) {
      console.error('Error fetching clean report:', e);
    }
  }

  resetCleanModalUI() {
    this.cleanBadgeStatus.className = 'badge badge-info';
    this.cleanBadgeStatus.textContent = 'Oczekuje na pomiar';
    this.cleanSummaryText.textContent = 'Kliknij „Zmierz migotanie”, aby sprawdzić stabilność ekspozycji i kolorów metodą przepływu optycznego (DIS).';
    this.cleanMetricsGrid.style.display = 'none';
    this.cleanProgressBar.style.display = 'none';
    this.btnExportCleanNk.disabled = true;
    this.btnRunClean.disabled = true;
  }

  displayCleanReport(rep) {
    if (rep.reduction_pct >= 50) {
      this.cleanBadgeStatus.className = 'badge badge-success';
      this.cleanBadgeStatus.textContent = `Wyczyszczony (-${rep.reduction_pct}%)`;
    } else {
      this.cleanBadgeStatus.className = 'badge badge-warning';
      this.cleanBadgeStatus.textContent = 'Wymagane czyszczenie';
    }

    const stagesStr = rep.stages_applied ? rep.stages_applied.join(' + ') : 'Deflicker v10';
    this.cleanSummaryText.textContent = `Raport dla ${rep.plate_name}: Zastosowano ${stagesStr}. Redukcja migotania: ${rep.reduction_pct}%. Zakres wzmocnienia: ${rep.gain_min} - ${rep.gain_max}.`;
    this.cleanMetricsGrid.style.display = 'grid';

    const b_init = rep.flicker_before;
    const b_final = rep.flicker_after;
    this.metricCleanInit.textContent = `R ${b_init.r}% / G ${b_init.g}% / B ${b_init.b}% (p95: ${b_init.p95}%)`;
    this.metricCleanFinal.textContent = `R ${b_final.r}% / G ${b_final.g}% / B ${b_final.b}% (p95: ${b_final.p95}%)`;
    this.metricCleanRedPct.textContent = `${rep.reduction_pct}%`;
    this.metricCleanNoiseRed.textContent = rep.noise_reduction_pct > 0 ? `${rep.noise_reduction_pct}%` : 'Wyłączone';

    this.btnExportCleanNk.disabled = !rep.nk_group_path;
    this.btnRunClean.disabled = false;
  }

  async measurePlateClean() {
    const plate = this.cleanPlateSelect.value;
    this.btnMeasureClean.disabled = true;
    this.cleanBadgeStatus.className = 'badge badge-info';
    this.cleanBadgeStatus.textContent = 'Mierzenie migotania...';
    this.cleanSummaryText.textContent = 'Obliczanie przepływu DIS Optical Flow i wariancji czasowej dla klatek...';

    try {
      const res = await fetch('/api/plate/clean/measure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plate: plate, max_frames: 16 })
      });
      const data = await res.json();
      if (data.success) {
        const fl = data.flicker;
        this.cleanMetricsGrid.style.display = 'grid';
        this.metricCleanInit.textContent = `R ${fl.r}% / G ${fl.g}% / B ${fl.b}% (p95: ${fl.p95}%)`;
        this.metricCleanFinal.textContent = '-';
        this.metricCleanRedPct.textContent = '-';
        this.metricCleanNoiseRed.textContent = '-';

        if (data.needs_deflicker) {
          this.cleanBadgeStatus.className = 'badge badge-warning';
          this.cleanBadgeStatus.textContent = 'Wykryto migotanie';
          this.cleanSummaryText.textContent = `Średnie migotanie wynosi ${fl.mean}% (p95: ${fl.p95}%). Zalecane uruchomienie łańcucha deflickera (Etap 1 + 2 + 4).`;
        } else {
          this.cleanBadgeStatus.className = 'badge badge-success';
          this.cleanBadgeStatus.textContent = 'Stabilny plate';
          this.cleanSummaryText.textContent = `Plate ma niskie migotanie (${fl.mean}%). Czyszczenie opcjonalne.`;
        }
        this.btnRunClean.disabled = false;
      } else {
        alert(`Błąd pomiaru: ${data.detail || 'Nieznany błąd'}`);
      }
    } catch (e) {
      alert(`Błąd połączenia z serwerem: ${e}`);
    } finally {
      this.btnMeasureClean.disabled = false;
    }
  }

  async runPlateClean() {
    const plate = this.cleanPlateSelect.value;
    const mode = this.cleanExportModeSelect.value;
    let confirmed = false;

    if (mode !== 'nuke_group') {
      confirmed = await this.uiConfirm(`Wybrano tryb zapisu pełnych plików EXR (${mode}). Zapisać wygenerowane klatki w folderze projektu?`, { title: 'Zapisać pełne klatki EXR?', ok: 'Zapisz', tone: 'info' });
      if (!confirmed) return;
    }

    this.btnRunClean.disabled = true;
    this.cleanProgressBar.style.display = 'flex';
    this.cleanProgressFill.style.width = '10%';
    this.cleanProgressText.textContent = 'Start przetwarzania deflickera...';

    try {
      const res = await fetch('/api/plate/clean/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate: plate,
          target_reduction: 0.70,
          enable_stage1: this.cleanStage1Checkbox.checked,
          enable_stage2: this.cleanStage2Checkbox.checked,
          enable_stage3: this.cleanStage3Checkbox.checked,
          enable_stage4: this.cleanStage4Checkbox.checked,
          enable_denoise: this.cleanDenoiseCheckbox.checked,
          export_mode: mode,
          confirmed: confirmed
        })
      });
      const data = await res.json();
      if (data.success && data.report) {
        this.displayCleanReport(data.report);
        this.showToast(`Czyszczenie plate'u ${plate} zakończone sukcesem (-${data.report.reduction_pct}%)!`);
        if (data.nk_script_path) {
          alert(`Wygenerowano węzeł Nuke:\n${data.nk_script_path}\n\nGroup Deflicker_${plate} jest gotowy do podpięcia pod Read w Nuke.`);
        }
      } else {
        alert(`Błąd: ${data.detail || 'Nie udało się zakończyć czyszczenia'}`);
      }
    } catch (e) {
      alert(`Błąd połączenia z serwerem: ${e}`);
    } finally {
      this.btnRunClean.disabled = false;
      this.cleanProgressBar.style.display = 'none';
    }
  }

  async exportCleanNk() {
    const plate = this.cleanPlateSelect.value;
    try {
      const res = await fetch(`/api/plate/clean/report?plate=${encodeURIComponent(plate)}`);
      const data = await res.json();
      if (data.exists && data.report && data.report.nk_group_path) {
        alert(`Ścieżka do skryptu Nuke:\n${data.report.nk_group_path}`);
      } else {
        alert('Najpierw wykonaj czyszczenie plate\'u.');
      }
    } catch (e) {
      alert(`Błąd: ${e}`);
    }
  }

  // --------------------------------------------------------------------------
  // Plate Management Modal (Add / Update / Inspect / Delete Plates)
  // --------------------------------------------------------------------------
  openPlatesModal() {
    if (!this.projectLoaded) {
      alert('Najpierw otwórz projekt ujęcia.');
      return;
    }
    if (this.platesModalShotName) {
      this.platesModalShotName.textContent = `Ujęcie: ${this.shotNameDisplay ? this.shotNameDisplay.textContent : '-'}`;
    }
    this.platesModal.style.display = 'flex';
    this.fetchAndRenderPlates();
  }

  async fetchAndRenderPlates() {
    if (!this.platesListContainer) return;
    this.platesListContainer.innerHTML = '<div class="spinner"></div> Wczytywanie płyt...';
    try {
      const res = await fetch('/api/project/plates');
      const data = await res.json();
      if (data.success) {
        this.renderPlatesList(data.plates);
      } else {
        this.platesListContainer.innerHTML = `<div class="field-hint error">Błąd: ${data.detail || 'Nie udało się wczytać listy płyt'}</div>`;
      }
    } catch (e) {
      this.platesListContainer.innerHTML = `<div class="field-hint error">Błąd połączenia: ${e}</div>`;
    }
  }

  // fill the "add / update" form of the plates dialog for one role (BG row, "change path", BG chosen in the header)
  preparePlateForm(name, pattern = '') {
    if (!this.plateNameSelect) return;
    const known = [...this.plateNameSelect.options].some((o) => o.value === name);
    this.plateNameSelect.value = known ? name : 'custom';
    if (!known && this.customPlateNameInput) this.customPlateNameInput.value = name;
    if (this.customPlateNameGroup) this.customPlateNameGroup.style.display = known ? 'none' : 'block';
    this.platePatternInput.value = pattern || '';
    if (this.plateInspectResult) {
      this.plateInspectResult.textContent = name === 'bg'
        ? 'Tło docelowe BG: pojedyncza klatka, sekwencja albo wideo – „📄 Plik…” albo „📁 Przeglądaj…”, potem „💾 Zapisz plate w projekcie”.'
        : 'Możesz wkleić folder sekwencji albo ścieżkę z wzorcem ####.';
      this.plateInspectResult.className = 'field-hint';
    }
    this.platePatternInput.scrollIntoView({ block: 'center' });
    this.platePatternInput.focus();
  }

  renderPlatesList(plates) {
    if (!this.platesListContainer) return;
    this.platesListContainer.innerHTML = '';
    if (!plates || plates.length === 0) {
      this.platesListContainer.innerHTML = '<div class="field-hint">Brak zdefiniowanych płyt w projekcie.</div>';
      return;
    }
    // the target background is not a plate of the shot, but it is set here: a row even when it is missing
    if (!plates.some((p) => p.name === 'bg')) {
      const card = document.createElement('div');
      card.className = 'plate-card plate-card-empty';
      card.innerHTML = `<div class="plate-card-left"><div class="plate-card-header"><span class="plate-badge plate-badge-custom">bg</span>
        <span class="plate-title">Tło docelowe (BG) – podgląd compu</span></div>
        <div class="plate-pattern-text">Nie ustawione – pojedyncza klatka, sekwencja albo wideo, na którym oglądasz wycięcie (prawa strona porównania „BG”).</div></div>`;
      const acts = document.createElement('div'); acts.className = 'plate-card-actions';
      const set = document.createElement('button'); set.className = 'btn btn-primary btn-xs'; set.textContent = '➕ Ustaw BG…';
      set.onclick = () => this.preparePlateForm('bg');
      acts.appendChild(set); card.appendChild(acts);
      this.platesListContainer.appendChild(card);
    }

    plates.forEach((p) => {
      const card = document.createElement('div');
      card.className = 'plate-card';

      let badgeClass = 'plate-badge-custom';
      if (p.name === 'beauty') badgeClass = 'plate-badge-beauty';
      else if (p.name === 'cleanplate' || p.name === 'clean_plate') badgeClass = 'plate-badge-cleanplate';
      else if (p.name === 'backlit') badgeClass = 'plate-badge-backlit';
      else if (p.name === 'depth') badgeClass = 'plate-badge-depth';

      const left = document.createElement('div');
      left.className = 'plate-card-left';

      const header = document.createElement('div');
      header.className = 'plate-card-header';
      header.innerHTML = `
        <span class="plate-badge ${badgeClass}">${p.name}</span>
        <span class="plate-title">${p.label || p.name}</span>
      `;

      const patternEl = document.createElement('div');
      patternEl.className = 'plate-pattern-text';
      patternEl.textContent = p.pattern || '(Brak przypisanej ścieżki)';
      patternEl.title = p.pattern || '';

      const stats = document.createElement('div');
      stats.className = 'plate-card-stats';

      const fileStat = document.createElement('span');
      fileStat.className = `plate-stat-pill ${p.exists ? 'success' : 'warning'}`;
      fileStat.textContent = p.exists ? `✓ ${p.frame_count} kl.` : '⚠️ Brak plików';
      stats.appendChild(fileStat);
      if (p.mismatch) {   // backlit / clear plate of another resolution: not used by any stage
        const mm = document.createElement('span');
        mm.className = 'plate-stat-pill warning';
        mm.textContent = `⚠️ ${p.mismatch}`;
        mm.title = 'Inna rozdzielczość niż beauty – ten plate nie jest używany w żadnym etapie (pewnie z innego ujęcia). Podmień go albo usuń.';
        stats.appendChild(mm);
      }

      const proxyStat = document.createElement('span');
      proxyStat.className = `plate-stat-pill ${p.proxy_count > 0 ? 'success' : ''}`;
      proxyStat.textContent = p.proxy_count > 0 ? `Proxy: ${p.proxy_count} kl.` : 'Brak proxy';
      stats.appendChild(proxyStat);

      left.appendChild(header);
      left.appendChild(patternEl);
      left.appendChild(stats);

      const actions = document.createElement('div');
      actions.className = 'plate-card-actions';

      // Button: Select & View
      const btnView = document.createElement('button');
      btnView.className = 'btn btn-secondary btn-xs';
      btnView.textContent = '👁️ Podgląd';
      btnView.title = 'Przełącz widok na tę warstwę';
      btnView.onclick = () => {
        this.setActivePlate(p.name);
        this.platesModal.style.display = 'none';
        this.showToast(`Przełączono podgląd na warstwę: ${p.label || p.name}`);
      };
      actions.appendChild(btnView);

      // Button: change the path (fills the form below for this role)
      const btnEdit = document.createElement('button');
      btnEdit.className = 'btn btn-ghost btn-xs';
      btnEdit.textContent = '✎ Zmień';
      btnEdit.title = 'Zmień ścieżkę tej warstwy (formularz poniżej)';
      btnEdit.onclick = () => this.preparePlateForm(p.name, p.pattern || '');
      actions.appendChild(btnEdit);

      // Button: Generate Proxy (if available and needed)
      if (p.exists && p.proxy_count === 0) {
        const btnGen = document.createElement('button');
        btnGen.className = 'btn btn-ghost btn-xs';
        btnGen.textContent = '⚡ Proxy';
        btnGen.title = 'Wygeneruj proxy 1920px dla tej płyty';
        btnGen.onclick = async () => {
          this.showToast(`Generowanie proxy dla ${p.name}...`);
          try {
            const res = await fetch(`/api/project/generate_proxies?plate=${encodeURIComponent(p.name)}`, { method: 'POST' });
            const d = await res.json();
            if (d.success) {
              this.showToast(`Wygenerowano ${d.rendered_count} klatek proxy dla ${p.name}!`);
              this.fetchAndRenderPlates();
            }
          } catch (e) {
            alert(`Błąd generowania proxy: ${e}`);
          }
        };
        actions.appendChild(btnGen);
      }

      // Button: Delete (only for auxiliary plates, not beauty)
      if (!p.is_primary) {
        const btnDel = document.createElement('button');
        btnDel.className = 'btn btn-ghost btn-xs text-danger';
        btnDel.textContent = '🗑️';
        btnDel.title = 'Usuń tę warstwę z projektu';
        btnDel.onclick = () => this.deletePlate(p.name);
        actions.appendChild(btnDel);
      }

      card.appendChild(left);
      card.appendChild(actions);
      this.platesListContainer.appendChild(card);
    });
  }

  // "Browse…" in the plates dialog: native folder picker (starts in the folder of the current path), then checks the sequence
  async browsePlatePath(file = false) {
    const cur = this.platePatternInput.value.trim().replace(/\\/g, '/');
    const start = cur ? (/#|%0\d*d|\.\w{2,4}$/.test(cur.split('/').pop()) ? cur.split('/').slice(0, -1).join('/') : cur) : '';
    try {
      const res = await fetch(file ? '/api/dialog/pick_file' : '/api/dialog/pick_folder', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: window.mdT(file ? 'Wybierz plik (klatka albo wideo)' : "Wybierz folder sekwencji plate'u"), initial_dir: start || null })   // system dialog title in the UI language
      });
      const data = await res.json();
      const picked = file ? data.path : data.folder;
      if (!picked) return;
      this.platePatternInput.value = picked;
      await this.inspectPlatePath();
    } catch (e) {
      this.plateInspectResult.textContent = `Okno wyboru folderu: ${e}`;
      this.plateInspectResult.className = 'field-hint error';
    }
  }

  async inspectPlatePath() {
    const rawPath = this.platePatternInput.value.trim();
    if (!rawPath) {
      this.plateInspectResult.textContent = 'Wprowadź najpierw ścieżkę lub folder.';
      this.plateInspectResult.className = 'field-hint error';
      return;
    }

    this.plateInspectResult.textContent = 'Sprawdzanie ścieżki i wykrywanie klatek...';
    this.plateInspectResult.className = 'field-hint';

    try {
      const res = await fetch('/api/project/plate/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: rawPath })
      });
      const data = await res.json();
      if (data.success && data.info) {
        const info = data.info;
        if (info.exists && info.frame_count > 0) {
          const dims = (info.width && info.height) ? `${info.width}x${info.height}, ` : '';
          this.plateInspectResult.textContent = `✓ Wykryto sekwencję: ${info.frame_count} klatek (${dims}${info.colorspace || 'ACES'}, ${info.source_type || 'sekwencja'})`;
          this.plateInspectResult.className = 'field-hint success';
          if (info.pattern && info.pattern !== rawPath) {
            this.platePatternInput.value = info.pattern;
          }
        } else if (info.exists) {
          this.plateInspectResult.textContent = 'Folder istnieje, ale nie znaleziono w nim pasujących klatek obrazu.';
          this.plateInspectResult.className = 'field-hint error';
        } else {
          this.plateInspectResult.textContent = '⚠️ Ścieżka nie istnieje na dysku. Zostanie zapisana jako podana.';
          this.plateInspectResult.className = 'field-hint warning';
        }
      }
    } catch (e) {
      this.plateInspectResult.textContent = `Błąd sprawdzania: ${e}`;
      this.plateInspectResult.className = 'field-hint error';
    }
  }

  async savePlate() {
    const selName = this.plateNameSelect.value;
    const plateName = selName === 'custom' ? this.customPlateNameInput.value.trim() : selName;
    if (!plateName) {
      alert('Podaj identyfikator płyty.');
      return;
    }

    const pattern = this.platePatternInput.value.trim();
    if (!pattern) {
      alert('Podaj ścieżkę lub folder dla płyty.');
      return;
    }

    const genProxies = this.plateGenProxiesCheckbox.checked;

    this.btnSavePlate.disabled = true;
    this.btnSavePlate.textContent = 'Zapisywanie...';

    try {
      const res = await fetch('/api/project/plate/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate_name: plateName,
          pattern: pattern,
          generate_proxies: genProxies
        })
      });

      const data = await res.json();
      if (data.success) {
        this.showToast(`Płyta ${plateName} zapisana pomyślnie!`);
        this.platePatternInput.value = '';
        this.plateInspectResult.textContent = 'Możesz wkleić bezpośredni folder sekwencji lub ścieżkę z wzorcem ####.';
        this.plateInspectResult.className = 'field-hint';
        if (this.customPlateNameInput) this.customPlateNameInput.value = '';

        // Update selector and refresh list
        if (data.manifest && data.manifest.plates) {
          this.updatePlateSelector(data.manifest.plates);
        }
        this.fetchAndRenderPlates();
      } else {
        alert(`Błąd zapisu płyty: ${data.detail || data.error}`);
      }
    } catch (e) {
      alert(`Błąd połączenia: ${e}`);
    } finally {
      this.btnSavePlate.disabled = false;
      this.btnSavePlate.textContent = '💾 Zapisz płytę w projekcie';
    }
  }

  async deletePlate(plateName) {
    if (!(await this.uiConfirm(`Plate „${plateName}” zostanie usunięty z projektu.`, { title: 'Usunąć plate?', ok: 'Usuń', tone: 'error' }))) {
      return;
    }

    try {
      const res = await fetch(`/api/project/plate/${encodeURIComponent(plateName)}`, {
        method: 'DELETE'
      });
      const data = await res.json();
      if (data.success) {
        this.showToast(`Płyta ${plateName} usunięta`);
        if (data.manifest && data.manifest.plates) {
          this.updatePlateSelector(data.manifest.plates);
        }
        this.fetchAndRenderPlates();
      } else {
        alert(`Błąd: ${data.detail || data.error}`);
      }
    } catch (e) {
      alert(`Błąd połączenia: ${e}`);
    }
  }

  // -------------------------------------------------------------------------
  // Camera 2D Methods (Stage E1d)
  // -------------------------------------------------------------------------

  openCameraModal() {
    if (!this.projectLoaded) {
      alert('Najpierw otwórz projekt ujęcia');
      return;
    }
    this.cameraModal.style.display = 'flex';
    if (this.cameraRefFrameInput) {
      this.cameraRefFrameInput.value = this.firstFrame || 1;
    }
    this.checkExistingCameraReport();
  }

  async checkExistingCameraReport() {
    try {
      const res = await fetch('/api/camera/report');
      const data = await res.json();
      if (data.exists && data.camera2d) {
        this.displayCameraReport(data.camera2d, data.nk_path);
      } else {
        this.resetCameraModalUI();
      }
    } catch (e) {
      console.error('Error fetching camera report:', e);
    }
  }

  resetCameraModalUI() {
    if (this.cameraBadgeStatus) {
      this.cameraBadgeStatus.className = 'badge badge-info';
      this.cameraBadgeStatus.textContent = 'Oczekuje na analizę';
    }
    if (this.cameraMotionTypeBadge) this.cameraMotionTypeBadge.style.display = 'none';
    if (this.cameraSummaryText) {
      this.cameraSummaryText.textContent = 'Kliknij „Analizuj ruch kamery”, aby obliczyć trajektorię i sprawdzić stabilność ujęcia.';
    }
    if (this.cameraMetricsGrid) this.cameraMetricsGrid.style.display = 'none';
    if (this.cameraProgressBar) this.cameraProgressBar.style.display = 'none';
    if (this.cameraFileInfo) this.cameraFileInfo.style.display = 'none';
    if (this.btnExportCameraNk) this.btnExportCameraNk.disabled = true;
  }

  displayCameraReport(cam, nkPath) {
    if (this.cameraBadgeStatus) {
      const isGood = cam.residual_median_px < 0.5 && cam.anchor_drift_px < 1.0;
      this.cameraBadgeStatus.className = isGood ? 'badge badge-success' : 'badge badge-warning';
      this.cameraBadgeStatus.textContent = isGood ? 'Śledzenie stabilne' : 'Wymaga uwagi';
    }

    if (this.cameraMotionTypeBadge) {
      this.cameraMotionTypeBadge.style.display = 'inline-block';
      this.cameraMotionTypeBadge.textContent = `Typ: ${cam.motion_type.toUpperCase()}`;
    }

    if (this.cameraSummaryText) {
      this.cameraSummaryText.textContent = `Obliczono homografie dla ${cam.frames.length} klatek. Dryf na kotwicach skompensowany.`;
    }

    if (this.cameraMetricsGrid) {
      this.cameraMetricsGrid.style.display = 'grid';
      if (this.metricCamMotionType) this.metricCamMotionType.textContent = cam.motion_type;
      if (this.metricCamResMedian) this.metricCamResMedian.textContent = `${cam.residual_median_px} px`;
      if (this.metricCamResP95) this.metricCamResP95.textContent = `${cam.residual_p95_px} px`;
      if (this.metricCamDrift) this.metricCamDrift.textContent = `${cam.anchor_drift_px} px`;
    }

    if (this.cameraFileInfo) {
      this.cameraFileInfo.style.display = 'block';
      if (this.cameraJsonPathDisplay) this.cameraJsonPathDisplay.textContent = 'camera/camera2d.json';
      if (this.cameraNkPathDisplay) this.cameraNkPathDisplay.textContent = nkPath || `camera/Camera2D_${cam.shot_name}.nk`;
    }

    if (this.btnExportCameraNk) this.btnExportCameraNk.disabled = false;
  }

  async runCameraTrack() {
    if (!this.projectLoaded) return;
    this.btnRunCameraTrack.disabled = true;
    this.btnRunCameraTrack.textContent = '⏳ Analizuję...';
    if (this.cameraProgressBar) {
      this.cameraProgressBar.style.display = 'flex';
      this.cameraProgressFill.style.width = '0%';
      this.cameraProgressText.textContent = 'Start...';
    }

    try {
      const refFrame = parseInt(this.cameraRefFrameInput.value, 10) || this.firstFrame || 1;
      const interval = parseInt(this.cameraAnchorIntervalInput.value, 10) || 24;
      const evalW = parseInt(this.cameraEvalWSelect.value, 10) || 1024;

      const res = await fetch('/api/camera/track', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ref_frame: refFrame,
          anchor_interval: interval,
          eval_w: evalW,
          export_nk: true
        })
      });

      const data = await res.json();
      if (data.success && data.camera2d) {
        this.displayCameraReport(data.camera2d, data.nk_script_path);
        this.showToast('Analiza ruchu kamery 2D zakończona pomyślnie!');
      } else {
        alert(`Błąd analizy: ${data.detail || data.error}`);
      }
    } catch (e) {
      alert(`Błąd połączenia: ${e}`);
    } finally {
      this.btnRunCameraTrack.disabled = false;
      this.btnRunCameraTrack.textContent = '▶️ Analizuj ruch kamery';
      if (this.cameraProgressBar) this.cameraProgressBar.style.display = 'none';
    }
  }

  async exportCameraNk() {
    try {
      const res = await fetch('/api/camera/export_nk', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        alert(`Wygenerowano skrypt Nuke:\n${data.nk_path}`);
        if (this.cameraNkPathDisplay) this.cameraNkPathDisplay.textContent = data.nk_path;
      } else {
        alert(`Błąd eksportu: ${data.detail || data.error}`);
      }
    } catch (e) {
      alert(`Błąd połączenia: ${e}`);
    }
  }

  showLoading(visible, msg = '') {
    this.loadingOverlay.style.display = visible ? 'flex' : 'none';
    this.loadingMessage.textContent = msg;
  }

  showToast(msg, ms = 3500) {
    console.log('[MatteDesk]', msg);
    let t = document.getElementById('mdToast');
    if (!t) { t = document.createElement('div'); t.id = 'mdToast'; t.className = 'md-toast'; document.body.appendChild(t); }
    t.textContent = msg; t.style.display = 'block';
    clearTimeout(this._toastT); this._toastT = setTimeout(() => { t.style.display = 'none'; }, ms);
  }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  window.viewer = new MatteDeskViewer();
  // Every alert() of the app as an in-app window: Chrome lets the user block page dialogs, then messages vanish silently
  window.alert = (msg) => { window.viewer.uiDialog({ title: 'MatteDesk', html: window.viewer.esc(msg).replace(/\n/g, '<br>') }); };
});
