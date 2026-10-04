/* Suika Engine & AI Observatory - interactive dashboard (vanilla JS, file:// friendly).
 * Loads trace_*.js captured by viz/capture.py and replays them with full
 * engine + AI-internals overlays. No build step, no ES modules (CORS-safe). */
(function () {
  "use strict";

  // ------------------------------------------------------------------ utils
  var $ = function (s) { return document.querySelector(s); };
  var SVGNS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }
  function rgb(c) { return "rgb(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + ")"; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function lerpColor(a, b, t) { return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function qColor(q, lo, hi) {
    var t = clamp((q - lo) / ((hi - lo) || 1), 0, 1), c;
    if (t < 0.5) c = lerpColor([224, 82, 77], [240, 200, 80], t / 0.5);
    else c = lerpColor([240, 200, 80], [90, 200, 120], (t - 0.5) / 0.5);
    return rgb(c);
  }

  // tooltip
  var tip = $("#tooltip");
  function showTip(html, ev) {
    tip.innerHTML = html; tip.style.display = "block";
    var x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + 290 > window.innerWidth) x = ev.clientX - 290;
    if (y + 120 > window.innerHeight) y = ev.clientY - 120;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.style.display = "none"; }

  // ------------------------------------------------------------------ state
  var TRACE = null;             // current trace object
  var META = null;
  var decisionAtFrame = [];     // frameIdx -> decision array index (or -1)
  var state = {
    frameIdx: 0, playing: false, speed: 1, lastDec: -2,
    tab: "tree", overlays: { velocity: true, merges: true, aim: true, cand: true, labels: false, dead: true },
    qDomain: [0, 1],
    play: { active: false, animating: false, done: false, hoverX: null, current: null, rule: "loop" },
    autoplay: false, pinned: false
  };
  var T = null;                 // canvas transform
  var CW = 560, CH = 760;       // logical canvas size
  var WORLD_DEFAULT = { x0: 403, x1: 875, y0: 52, y1: 688 };
  var WORLD = WORLD_DEFAULT;    // per-trace view-box (adapts to engine screen size)
  var Q = new URLSearchParams(location.search);   // deep-link params

  function fitTransform() {
    var sx = CW / (WORLD.x1 - WORLD.x0), sy = CH / (WORLD.y1 - WORLD.y0);
    var s = Math.min(sx, sy);
    var ox = (CW - (WORLD.x1 - WORLD.x0) * s) / 2 - WORLD.x0 * s;
    var oy = (CH - (WORLD.y1 - WORLD.y0) * s) / 2 - WORLD.y0 * s;
    return { s: s, X: function (x) { return ox + x * s; }, Y: function (y) { return oy + y * s; }, R: function (r) { return r * s; } };
  }
  function colorOf(t) {
    var ft = META && META.fruit_table;
    if (ft && ft[t]) return rgb(ft[t].color);
    return "#888";
  }
  function nameOf(t) { return (META && META.fruit_table && META.fruit_table[t]) ? META.fruit_table[t].name : ("#" + t); }

  // ------------------------------------------------------------------ loading
  var PLAY_MODES = ["__play__", "__play", "play", "human", "live"];
  function isPlayMode(v) { return PLAY_MODES.indexOf(v) >= 0; }

  function gameLabel(t) {
    return t.agent + " · seed" + t.seed + " · " + t.final_score + "分" +
      (t.censored ? " · 评测上限时仍存活" : "") + " · " + t.max_fruit_name;
  }
  function fillGameOptions(sel, traces) {
    if (!sel) return;
    clear(sel);
    traces.forEach(function (t) {
      var o = document.createElement("option");
      o.value = t.name; o.textContent = gameLabel(t);
      sel.appendChild(o);
    });
    if (window.SUIKA_MANIFEST.live_play_enabled) {
      var po = document.createElement("option");
      po.value = "__play__"; po.textContent = "🎮 人类对局（实时引擎）";
      sel.appendChild(po);
    }
  }

  function ruleLabel(meta) {
    if (meta.two_watermelon_rule === "merge_disappear_score") {
      return "双西瓜消除 + " + meta.watermelon_merge_points + " 分";
    }
    return meta.rule_mode === "poof" ? "双西瓜消除" : "双西瓜保留（旧规则）";
  }

  function loadManifest() {
    if (!window.SUIKA_MANIFEST) {
      $("#loading").classList.add("show");
      $("#loading").textContent = "未找到 traces/manifest.js — 请先运行: python -m viz.make_demo";
      return;
    }
    var traces = window.SUIKA_MANIFEST.traces || [];
    var sel = $("#gameSelect"), psel = $("#playGameSelect");
    fillGameOptions(sel, traces);
    fillGameOptions(psel, traces);
    var pref = traces.filter(function (t) { return t.name === window.SUIKA_MANIFEST.featured; })[0] || traces[0];
    var gq = Q.get("game");
    var wantPlay = isPlayMode(gq);
    if (!wantPlay && gq && traces.some(function (t) { return t.name === gq; })) {
      pref = traces.filter(function (t) { return t.name === gq; })[0];
    }
    if (pref) sel.value = pref.name;
    sel.onchange = function () {
      if (sel.value === "__play__") enterPlay();
      else { exitPlay(); selectTrace(sel.value); }
    };
    if (psel) psel.onchange = function () {
      // picking a match from the play view always previews the WHOLE trajectory
      if (psel.value === "__play__") { enterPlay(); return; }
      exitPlay(); selectTrace(psel.value, true);
    };
    renderArena();
    if (wantPlay) { state.pinned = true; enterPlay(); return; }
    if (pref) loadTrace(pref.name);
    else { $("#loading").classList.add("show"); $("#loading").textContent = "manifest 中没有 trace"; }
  }

  function selectTrace(name, autoplay) {
    var url = new URL(location.href);
    url.searchParams.set("game", name);
    ["frame", "step", "tab"].forEach(function (key) { url.searchParams.delete(key); });
    history.replaceState(null, "", url);
    Q = new URLSearchParams(url.search);
    loadTrace(name, autoplay);
  }

  function loadTrace(name, autoplay) {
    state.autoplay = !!autoplay;
    if (window.SUIKA_TRACES && window.SUIKA_TRACES[name]) { setupTrace(window.SUIKA_TRACES[name], name); return; }
    $("#loading").classList.add("show"); $("#loading").textContent = "载入 " + name + " …";
    var entry = (window.SUIKA_MANIFEST.traces || []).filter(function (t) { return t.name === name; })[0];
    var file = entry ? entry.file : ("trace_" + name + ".js");
    var gz = entry ? entry.file_gz : null;
    if (gz && window.fetch && window.DecompressionStream) {
      fetch("../traces/" + gz).then(function (r) {
        if (!r.ok) throw new Error("http " + r.status);
        return r.arrayBuffer();
      }).then(function (buf) {
        return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
      }).then(function (txt) {
        $("#loading").classList.remove("show");
        var tr = JSON.parse(txt);
        window.SUIKA_TRACES = window.SUIKA_TRACES || {};
        window.SUIKA_TRACES[name] = tr;
        setupTrace(tr, name);
      }).catch(function (e) {
        $("#loading").classList.add("show"); $("#loading").textContent = "无法加载 " + gz + "（" + e.message + "）";
      });
      return;
    }
    var sc = document.createElement("script");
    sc.src = "../traces/" + file;
    sc.onload = function () {
      $("#loading").classList.remove("show");
      if (window.SUIKA_TRACES && window.SUIKA_TRACES[name]) setupTrace(window.SUIKA_TRACES[name], name);
      else { $("#loading").classList.add("show"); $("#loading").textContent = "载入失败: " + name; }
    };
    sc.onerror = function () { $("#loading").classList.add("show"); $("#loading").textContent = "无法加载 " + file; };
    document.body.appendChild(sc);
  }

  function setupTrace(trace, name) {
    TRACE = trace; META = trace.meta;
    // keep both match selectors pointing at what is on screen
    syncSelects(name);
    // frame -> decision map
    decisionAtFrame = new Array(trace.frames.length).fill(-1);
    trace.decisions.forEach(function (d, i) {
      for (var f = d.frame_start; f < d.frame_end && f < decisionAtFrame.length; f++) decisionAtFrame[f] = i;
    });
    state.frameIdx = 0; state.playing = false; state.lastDec = -2;
    $("#playBtn").textContent = "▶ 播放";
    var sl = $("#frameSlider"); sl.max = Math.max(0, trace.frames.length - 1); sl.value = 0;
    // rule badge
    $("#ruleBadge").textContent = ruleLabel(META);
    // per-engine world view-box + engine badge (supports different screen sizes)
    WORLD = META.world ? META.world : WORLD_DEFAULT;
    T = fitTransform();
    var eng = META.engine || {};
    $("#engineBadge").textContent = "引擎: " + (eng.id || eng.name || "?") +
      (eng.screen ? " " + eng.screen.width + "×" + eng.screen.height : "");
    // step select
    var ss = $("#stepSelect"); clear(ss);
    trace.decisions.forEach(function (d, i) {
      var o = document.createElement("option"); o.value = i;
      o.textContent = "步 " + d.step + " · " + d.current.name + " → col " + (d.col == null ? "?" : d.col) + " · " + d.score + "分";
      ss.appendChild(o);
    });
    ss.onchange = function () { jumpToDecision(parseInt(ss.value, 10)); };
    // q domain from all tree edges (for consistent colouring)
    computeQDomain();
    renderCurve();
    // open on an informative step (a few drops in) rather than the empty
    // first frame, so the board and AI panels have content immediately.
    // ?step= and ?tab= deep-link override.
    var initIdx = Math.min(6, Math.max(0, trace.decisions.length - 1));
    var sq = Q.get("step");
    if (sq != null && sq !== "") initIdx = clamp(parseInt(sq, 10) || 0, 0, trace.decisions.length - 1);
    jumpToDecision(initIdx);
    var tq = Q.get("tab");
    if (tq && isTabEnabled(tq)) switchTab(tq);
    var fq = Q.get("frame");
    var galleryEntry = window.SUIKA_MANIFEST.traces.filter(function (t) { return t.name === name; })[0];
    if ((fq == null || fq === "") && (sq == null || sq === "") && galleryEntry && galleryEntry.preview_frame != null) fq = galleryEntry.preview_frame;
    if (fq != null && fq !== "") {
      state.frameIdx = clamp(parseInt(fq, 10) || 0, 0, TRACE.frames.length - 1);
      state.lastDec = -2; renderAll();
    }
    // picked from the play view => preview the WHOLE game from frame 0
    if (state.autoplay) {
      state.autoplay = false;
      state.frameIdx = 0; state.lastDec = -2;
      renderAll(); play();
    }
    var status = $("#traceStatus");
    var area = META.play_area;
    status.textContent = META.final_score.toLocaleString() + " 分 · " + META.steps.toLocaleString() +
      " 次落子 · " + (area.right - area.left) + "×" + (area.bot - area.killy) + " · seed " + META.seed +
      (META.censored ? " · 达到评测上限，结束时仍存活" : (META.game_over ? " · 自然结束" : " · 录制片段")) +
      (META.capture_mode === "recorded_actions" ? " · 已核验的动作回放" : " · 包含决策数值");
    $("#downloadTrace").href = "../traces/" + (window.SUIKA_MANIFEST.traces.filter(function (t) { return t.name === name; })[0] || {}).file_gz;
    renderArena();
  }

  function computeQDomain() {
    var qs = [];
    (TRACE.decisions || []).forEach(function (d) {
      var t = d.internals && d.internals.tree;
      if (t) t.edges.forEach(function (e) { if (isFinite(e.Q)) qs.push(e.Q); });
    });
    if (qs.length) {
      qs.sort(function (a, b) { return a - b; });
      var lo = qs[Math.floor(qs.length * 0.05)], hi = qs[Math.floor(qs.length * 0.95)];
      if (hi - lo < 0.05) { hi = lo + 0.05; }
      state.qDomain = [lo, hi];
    } else state.qDomain = [0, 1];
  }

  // ------------------------------------------------------------------ board canvas
  function setupCanvas() {
    var cv = $("#board");
    var dpr = window.devicePixelRatio || 1;
    cv.width = CW * dpr; cv.height = CH * dpr;
    cv.style.width = CW + "px";
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    T = fitTransform();
  }

  function currentDecisionIdx() {
    var di = decisionAtFrame[state.frameIdx];
    return (di == null || di < 0) ? -1 : di;
  }

  function drawBoard() {
    if (!TRACE || !T) return;
    var cv = $("#board"), ctx = cv.getContext("2d");
    var dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, CW, CH);
    var pa = META.play_area;

    // container backdrop
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(T.X(pa.left), T.Y(pa.top), T.R(pa.right - pa.left), T.R(pa.bot - pa.top));

    // death line (killy)
    if (state.overlays.dead) {
      ctx.save();
      ctx.strokeStyle = "rgba(224,82,77,0.85)"; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
      ctx.beginPath(); ctx.moveTo(T.X(pa.left), T.Y(pa.killy)); ctx.lineTo(T.X(pa.right), T.Y(pa.killy)); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "rgba(224,82,77,0.8)"; ctx.font = "10px sans-serif";
      ctx.fillText("死亡线 killy=" + pa.killy, T.X(pa.left) + 4, T.Y(pa.killy) - 4);
    }

    // walls
    ctx.strokeStyle = "#4a5468"; ctx.lineWidth = 4; ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(T.X(pa.left), T.Y(pa.top - 8)); ctx.lineTo(T.X(pa.left), T.Y(pa.bot));
    ctx.lineTo(T.X(pa.right), T.Y(pa.bot)); ctx.lineTo(T.X(pa.right), T.Y(pa.top - 8));
    ctx.stroke();

    var frame = TRACE.frames[state.frameIdx];
    if (!frame) return;
    var di = currentDecisionIdx();
    var dec = di >= 0 ? TRACE.decisions[di] : null;

    // candidate score / visit bars along the bottom (before fruits)
    if (dec && state.overlays.cand) drawCandidateBars(ctx, dec, pa);

    // aim line for the chosen drop (suppressed while live-play is idle: the
    // hover aim below shows the ready fruit instead)
    var playIdle = state.play.active && !state.play.animating && !state.play.done;
    if (dec && state.overlays.aim && !playIdle) {
      ctx.save();
      ctx.strokeStyle = "rgba(90,220,240,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(T.X(dec.x), T.Y(pa.top - 26)); ctx.lineTo(T.X(dec.x), T.Y(pa.bot)); ctx.stroke();
      ctx.restore();
      // triangle pointer + preview fruit
      ctx.fillStyle = "rgba(90,220,240,0.95)";
      ctx.beginPath();
      ctx.moveTo(T.X(dec.x) - 7, T.Y(pa.top - 30)); ctx.lineTo(T.X(dec.x) + 7, T.Y(pa.top - 30)); ctx.lineTo(T.X(dec.x), T.Y(pa.top - 18)); ctx.closePath(); ctx.fill();
      // top "cloud" preview shows the NEXT waiting fruit. The current fruit is
      // already released and falling among the bodies, so drawing it here too
      // would show two copies.
      if (dec.next) {
        var nrad = (META.fruit_table && META.fruit_table[dec.next.type])
          ? META.fruit_table[dec.next.type].radius : 14;
        ctx.fillStyle = colorOf(dec.next.type);
        ctx.beginPath(); ctx.arc(T.X(dec.x), T.Y(pa.top - 40), T.R(nrad), 0, 7); ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,.4)"; ctx.lineWidth = 1; ctx.stroke();
      }
    }

    // human-play hover aim (live mode, waiting for click)
    if (playIdle && state.play.hoverX != null && state.play.current) {
      var hx = T.X(state.play.hoverX);
      ctx.save();
      ctx.strokeStyle = "rgba(90,220,240,0.55)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(hx, T.Y(pa.top - 26)); ctx.lineTo(hx, T.Y(pa.bot)); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = colorOf(state.play.current.type);
      ctx.beginPath(); ctx.arc(hx, T.Y(pa.top - 40), T.R(state.play.current.radius || 14), 0, 7); ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,.4)"; ctx.lineWidth = 1; ctx.stroke();
    }

    // fruits
    for (var i = 0; i < frame.b.length; i++) {
      var b = frame.b[i];
      var cx = T.X(b.x), cy = T.Y(b.y), r = T.R(b.r);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7);
      ctx.fillStyle = colorOf(b.t); ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.35)"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, r - 1.5, 0, 7); ctx.stroke();
      if (state.overlays.labels) {
        ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.font = "bold " + Math.max(9, r * 0.7) + "px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(b.t + 1), cx, cy);
        ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
      }
      if (state.overlays.velocity) {
        var sp = Math.hypot(b.vx, b.vy);
        if (sp > 6) {
          var sc = clamp(sp * 0.06, 4, 34);
          var ux = b.vx / sp, uy = b.vy / sp;
          ctx.strokeStyle = "rgba(250,220,120,0.85)"; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + ux * sc, cy + uy * sc); ctx.stroke();
        }
      }
    }

    // merge / poof events on this frame
    if (state.overlays.merges && frame.m && frame.m.length) {
      frame.m.forEach(function (m) {
        var cx = T.X(m.x), cy = T.Y(m.y);
        var poof = m.kind === "poof";
        ctx.beginPath(); ctx.arc(cx, cy, T.R(poof ? 40 : 26), 0, 7);
        ctx.strokeStyle = poof ? "rgba(250,220,60,0.95)" : "rgba(255,255,255,0.9)";
        ctx.lineWidth = 3; ctx.stroke();
        ctx.fillStyle = poof ? "#fadc3c" : "#fff"; ctx.font = "bold 13px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText((poof ? "消除! +" : "+") + m.points, cx, cy - T.R(poof ? 44 : 30));
        ctx.textAlign = "start";
      });
    }

    if (META.censored && state.frameIdx === TRACE.frames.length - 1) {
      ctx.fillStyle = "#fadc3c"; ctx.font = "bold 20px sans-serif"; ctx.textAlign = "center";
      ctx.fillText("评测上限 · 仍存活", CW / 2, 60); ctx.textAlign = "start";
    }
    // game over banner
    if (dec && dec.game_over) {
      ctx.fillStyle = "rgba(224,82,77,0.16)"; ctx.fillRect(0, 0, CW, CH);
      ctx.fillStyle = "#ff8b86"; ctx.font = "bold 30px sans-serif"; ctx.textAlign = "center";
      ctx.fillText("GAME OVER", CW / 2, 60); ctx.textAlign = "start";
    }
  }

  function drawCandidateBars(ctx, dec, pa) {
    var intern = dec.internals || {};
    var items = null, mode = null;
    if (intern.candidates && intern.candidates.length) {
      mode = "heur";
      items = intern.candidates.map(function (c) {
        var go = c.terms && c.terms.gameover < 0;
        return { x: c.x, v: go ? null : c.total, go: go, best: false };
      });
    } else if (intern.root && intern.root.N) {
      mode = "az";
      var K = intern.root.K, lo = pa.left, w = pa.right - pa.left;
      items = intern.root.N.map(function (n, c) {
        return { x: lo + w * (c + 0.5) / K, v: n, go: false, best: false };
      });
    } else if (intern.q && intern.q.length) {
      mode = "q";
      var Kq = intern.q.length, loq = pa.left, wq = pa.right - pa.left;
      items = intern.q.map(function (v, c) {
        return { x: loq + wq * (c + 0.5) / Kq, v: v, go: false, best: false };
      });
    }
    if (!items || !items.length) return;
    var bestI = -1, bestV = -Infinity;
    items.forEach(function (it, i) { if (!it.go && it.v != null && it.v > bestV) { bestV = it.v; bestI = i; } });
    if (bestI >= 0) items[bestI].best = true;
    var vmax = Math.max.apply(null, items.map(function (it) { return it.go || it.v == null ? 0 : Math.abs(it.v); })) || 1;
    var baseY = T.Y(pa.bot) + 16, maxH = 46;
    var qlo = Infinity, qhi = -Infinity, qspan = 1;
    if (mode === "q") {
      items.forEach(function (it) {
        if (!it.go && it.v != null) { if (it.v < qlo) qlo = it.v; if (it.v > qhi) qhi = it.v; }
      });
      if (!isFinite(qlo) || qhi <= qlo) { qlo = 0; qhi = 1; }
      qspan = qhi - qlo;
    }
    items.forEach(function (it) {
      var sx = T.X(it.x);
      if (it.go) {
        ctx.fillStyle = "rgba(224,82,77,0.85)";
        ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center";
        ctx.fillText("☠", sx, baseY + 14); ctx.textAlign = "start";
        return;
      }
      var h = mode === "q"
        ? 2 + (maxH - 2) * ((it.v - qlo) / qspan)
        : maxH * (Math.abs(it.v) / vmax);
      ctx.fillStyle = it.best ? "#fadc3c" : (mode === "az" ? "rgba(90,220,240,0.6)" : (mode === "q" ? "rgba(199,125,255,0.75)" : "rgba(150,160,180,0.6)"));
      ctx.fillRect(sx - 3, baseY - h, 6, h);
    });
  }

  // ------------------------------------------------------------------ HUD
  function renderHUD() {
    var frame = TRACE.frames[state.frameIdx];
    var di = currentDecisionIdx();
    var dec = di >= 0 ? TRACE.decisions[di] : null;
    $("#hudScore").textContent = frame ? frame.s : 0;
    $("#hudStep").textContent = dec ? dec.step : 0;
    $("#hudFruits").textContent = frame ? frame.b.length : 0;
    $("#hudHeight").textContent = dec ? dec.max_height : "—";
    var maxT = 0; if (frame) frame.b.forEach(function (b) { if (b.t > maxT) maxT = b.t; });
    $("#hudMax").textContent = frame && frame.b.length ? nameOf(maxT) : "—";
    if (dec) {
      $("#hudCur").style.background = colorOf(dec.current.type);
      $("#hudCurName").textContent = dec.current.name;
      $("#hudNext").style.background = colorOf(dec.next.type);
      $("#hudNextName").textContent = dec.next.name;
    }
    // live-play idle: the broadcast must show the fruit READY to drop now and
    // the one after it (not the already-fallen fruit from the last decision)
    if (state.play.active && !state.play.animating && state.play.current) {
      $("#hudCur").style.background = colorOf(state.play.current.type);
      $("#hudCurName").textContent = state.play.current.name;
      if (state.play.next) {
        $("#hudNext").style.background = colorOf(state.play.next.type);
        $("#hudNextName").textContent = state.play.next.name;
      }
    }
    $("#frameLabel").textContent = state.frameIdx + " / " + (TRACE.frames.length - 1);
    $("#frameSlider").value = state.frameIdx;
  }

  // ------------------------------------------------------------------ AI panels
  // ---- baseline leaderboard (viz/traces/benchmarks.js) ----
  function benchRows() { return (window.SUIKA_BENCH && window.SUIKA_BENCH.controlled) || []; }
  function fmt1(v) { return (v == null) ? "—" : (Math.round(v * 10) / 10).toFixed(1); }
  function pct(v) { return (v == null) ? "—" : Math.round(v * 100) + "%"; }
  function benchFor(kind) {
    if (!kind) return null;
    var rows = benchRows(), i;
    for (i = 0; i < rows.length; i++) if (rows[i].name === kind) return rows[i];
    function first(pfx) {
      for (var j = 0; j < rows.length; j++) if (rows[j].name.indexOf(pfx) === 0) return rows[j];
      return null;
    }
    if (kind.indexOf("w3b_mlp") === 0 || kind.indexOf("mlp_deep") >= 0) return first("w3b_mlp");
    if (kind.indexOf("w3b_tf") === 0 || kind.indexOf("tf_deep") >= 0) return first("w3b_tf");
    if (kind.indexOf("plaindqn") === 0) return first("plaindqn");
    if (kind.indexOf("heuristic") === 0) return first("heuristic");
    if (kind.indexOf("alphazero") === 0) return first("alphazero");
    if (kind.indexOf("random") === 0) return first("random");
    return null;
  }
  function benchCommunityFor(kind) {
    if (!kind) return null;
    var rows = (window.SUIKA_BENCH && window.SUIKA_BENCH.community) || [];
    for (var i = 0; i < rows.length; i++) {
      if (kind.indexOf(rows[i].name) === 0 || kind.indexOf(rows[i].name.split("_")[0]) === 0) return rows[i];
    }
    return null;
  }

  function renderAgentCard(dec) {
    if (!dec) { $("#agentTitle").textContent = "尚未落子"; $("#agentMeta").textContent = "播放或选择一步以查看 AI 决策。"; return; }
    var br = benchFor(dec.kind), cb = benchCommunityFor(dec.kind);
    var kindName = { random: "随机基线", heuristic: "启发式 + 1步物理前瞻", alphazero: "AlphaZero (PUCT MCTS + Object-Transformer)", human: "人类玩家（实时引擎）" }[dec.kind] || (br ? br.label : dec.kind);
    $("#agentTitle").textContent = "第 " + dec.step + " 步 · " + kindName;
    var m = [];
    m.push("当前 <b style='color:" + colorOf(dec.current.type) + "'>" + dec.current.name + "</b> → 落点 col " + (dec.col == null ? "?" : dec.col) + " (x=" + dec.x + ")");
    m.push("本步得分 +" + dec.reward + " · 累计 " + dec.score + " · 场上 " + dec.fruit_count + " 果");
    if (META.capture_mode === "recorded_actions") m.push("greedy DQN · 已记录动作回放 · 未保存 Q 值");
    if (META.source && META.source.grad_steps != null) m.push("评测 checkpoint：grad " + META.source.grad_steps);
    if (br) {
      m.push("<span class='benchline'>greedy " + br.seeds + " 种子：mean <b>" + fmt1(br.mean) +
             "</b> · max " + br.max + (br.min != null ? " · min " + br.min : "") +
             (br.p2000 != null ? " · ≥2000 局 " + pct(br.p2000) : "") + "</span>");
    }
    if (cb) {
      m.push("<span class='benchline'>社区基线：原生环境 mean " +
             (cb.native_mean == null ? "—" : fmt1(cb.native_mean)) +
             (cb.native_seeds ? "（" + cb.native_seeds + " 种子）" : "") +
             (cb.ours_mean != null ? " · 迁入本环境 " + fmt1(cb.ours_mean) : "") + "</span>");
    }
    if (dec.kind === "alphazero" && dec.internals && dec.internals.root) {
      m.push("MCTS 访问 " + dec.internals.root.total_visits + " 次 · 局面价值 v=" + dec.internals.root.value + " · ckpt step " + (META.ckpt_step || "?"));
    }
    $("#agentMeta").innerHTML = m.join("<br>");
  }

  function updateTabs(dec) {
    var hasTree = !!(dec && dec.internals && dec.internals.tree);
    var hasQ = !!(dec && dec.internals && dec.internals.q && dec.internals.q.length);
    var recorded = META && META.capture_mode === "recorded_actions";
    var hasPV = recorded || !!(dec && dec.internals && (dec.internals.root || hasQ));
    document.querySelector(".tab[data-tab='pv']").textContent = recorded ? "落子记录" : "策略 / 价值";
    var hasHeur = !!(dec && dec.internals && dec.internals.candidates && dec.internals.candidates.length);
    setTabEnabled("tree", hasTree); setTabEnabled("pv", hasPV); setTabEnabled("heur", hasHeur);
    // auto-pick a valid tab
    var order = ["tree", "pv", "heur"];
    if (!isTabEnabled(state.tab)) {
      for (var i = 0; i < order.length; i++) if (isTabEnabled(order[i])) { switchTab(order[i]); break; }
    }
  }
  function setTabEnabled(name, on) {
    var b = document.querySelector(".tab[data-tab='" + name + "']");
    if (!b) return; b.disabled = !on; b.style.opacity = on ? "1" : "0.35"; b.style.pointerEvents = on ? "auto" : "none"; b.dataset.enabled = on ? "1" : "0";
  }
  function isTabEnabled(name) { var b = document.querySelector(".tab[data-tab='" + name + "']"); return b && b.dataset.enabled === "1"; }
  function switchTab(name) {
    state.tab = name;
    document.querySelectorAll(".tab").forEach(function (t) { t.classList.toggle("active", t.dataset.tab === name); });
    document.querySelectorAll(".panel").forEach(function (p) { p.classList.toggle("active", p.id === "panel-" + name); });
  }

  function renderPanels(dec) {
    renderTree(dec); renderPV(dec); renderHeur(dec);
  }

  // ---- MCTS tree ----
  function renderTree(dec) {
    var root = $("#treeSvg"); clear(root);
    var stats = $("#treeStats");
    if (!dec || !dec.internals || !dec.internals.tree) { stats.innerHTML = "该决策无 MCTS 树（非 AlphaZero）。"; return; }
    var tree = dec.internals.tree;
    var W = 520, H = 440, mx = 26, myTop = 34, myBot = 26;
    root.setAttribute("viewBox", "0 0 " + W + " " + H);
    // layout
    var children = {};
    tree.edges.forEach(function (e) { (children[e.from] = children[e.from] || []).push(e); });
    for (var k in children) children[k].sort(function (a, b) { return a.action - b.action; });
    var leafX = [0], pos = {};
    (function dfs(id, depth) {
      var kids = children[id] || [];
      if (!kids.length) { pos[id] = { x: leafX[0]++, y: depth }; return; }
      var sum = 0; kids.forEach(function (e) { dfs(e.to, depth + 1); sum += pos[e.to].x; });
      pos[id] = { x: sum / kids.length, y: depth };
    })(tree.root_id, 0);
    var maxX = Math.max(1, leafX[0] - 1);
    var maxDepth = 0; for (var p in pos) maxDepth = Math.max(maxDepth, pos[p].y);
    function PX(id) { return mx + (pos[id].x / maxX) * (W - 2 * mx); }
    function PY(id) { var d = pos[id].y; return myTop + (maxDepth ? d / maxDepth : 0) * (H - myTop - myBot); }
    var lo = state.qDomain[0], hi = state.qDomain[1];

    // chosen path (follow argmax N from root)
    var chosenEdges = {}, node = tree.root_id, guard = 0;
    while (guard++ < 32) {
      var kids = children[node] || []; if (!kids.length) break;
      var be = kids[0]; kids.forEach(function (e) { if (e.N > be.N) be = e; });
      chosenEdges[be.from + "-" + be.to] = true; node = be.to;
    }

    // edges
    tree.edges.forEach(function (e) {
      var x1 = PX(e.from), y1 = PY(e.from), x2 = PX(e.to), y2 = PY(e.to);
      var w = clamp(0.8 + Math.sqrt(e.N) * 1.1, 0.8, 9);
      var chosen = chosenEdges[e.from + "-" + e.to];
      var midy = (y1 + y2) / 2;
      var path = svg("path", {
        d: "M" + x1 + "," + y1 + " C" + x1 + "," + midy + " " + x2 + "," + midy + " " + x2 + "," + y2,
        fill: "none", stroke: chosen ? "#fadc3c" : qColor(e.Q, lo, hi),
        "stroke-width": chosen ? w + 1 : w, "stroke-opacity": chosen ? 0.95 : 0.75
      }, root);
      path.style.cursor = "pointer";
      path.addEventListener("mousemove", function (ev) {
        showTip("<b>动作 col " + e.action + "</b><br><span class='tt-k'>访问 N:</span> " + e.N +
          "<br><span class='tt-k'>价值 Q:</span> " + e.Q.toFixed(3) +
          "<br><span class='tt-k'>先验 P:</span> " + e.P.toFixed(3) +
          "<br><span class='tt-k'>即时奖励:</span> " + e.reward, ev);
      });
      path.addEventListener("mouseleave", hideTip);
      // action label on root edges
      if (e.from === tree.root_id) {
        svg("text", { x: x2, y: y2 - 9, fill: chosen ? "#fadc3c" : "#8b93a3", "font-size": "9", "text-anchor": "middle" }, root).textContent = "c" + e.action;
      }
    });
    // nodes
    tree.nodes.forEach(function (n) {
      var cx = PX(n.id), cy = PY(n.id);
      var r = clamp(3 + Math.sqrt(n.visits) * 1.15, 3.5, 15);
      var fill = n.is_terminal ? "#e0524d" : qColor(n.value, lo, hi);
      var c = svg("circle", { cx: cx, cy: cy, r: r, fill: fill, stroke: n.id === tree.root_id ? "#5adcf0" : "rgba(0,0,0,.4)", "stroke-width": n.id === tree.root_id ? 2.5 : 1 }, root);
      c.style.cursor = "pointer";
      c.addEventListener("mousemove", function (ev) {
        showTip("<b>节点 (深度 " + n.depth + ")</b><br><span class='tt-k'>总访问:</span> " + n.visits +
          "<br><span class='tt-k'>价值 v:</span> " + n.value.toFixed(3) +
          "<br><span class='tt-k'>场上水果:</span> " + n.fruit_count +
          "<br><span class='tt-k'>堆顶 y:</span> " + (n.top_y == null ? "—" : n.top_y) +
          (n.is_terminal ? "<br><b style='color:#e0524d'>终局 (game over)</b>" : ""), ev);
      });
      c.addEventListener("mouseleave", hideTip);
    });
    stats.innerHTML = "树规模: <b>" + tree.nodes.length + "</b> 节点 / <b>" + tree.edges.length + "</b> 边（限深 " + tree.max_depth + "、每层保留 top-" + tree.top_n + "）· 根总访问 <b>" + tree.total_visits + "</b> · Q 配色域 [" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]";
  }

  // ---- policy / value ----
  function renderPV(dec) {
    var root = $("#pvSvg"); clear(root);
    var gauge = $("#valueGauge");
    var recorded = META && META.capture_mode === "recorded_actions";
    root.style.display = recorded ? "none" : "block";
    if (recorded) {
      gauge.textContent = dec ? "实际落点：列 " + dec.col + " / 128，x=" + dec.x +
        "；本步 +" + dec.reward + " 分。此回放来自已记录动作，未保存该 checkpoint 的 Q 值。" : "选择落子查看记录。";
      return;
    }
    if (!dec || !dec.internals) { gauge.innerHTML = "该决策无策略/价值数据。"; return; }
    if (!dec.internals.root) {
      if (dec.internals.q && dec.internals.q.length) return renderQPanel(dec, root, gauge);
      gauge.innerHTML = "该决策无策略/价值数据（非 AlphaZero / DQN）。"; return;
    }
    var r = dec.internals.root, K = r.K;
    var W = 520, H = 300, ml = 40, mr = 14, mt = 26, mb = 44;
    root.setAttribute("viewBox", "0 0 " + W + " " + H);
    var iw = W - ml - mr, ih = H - mt - mb;
    var slot = iw / K, bw = slot * 0.34;
    var pmax = Math.max.apply(null, r.P.concat(r.visit_dist)) || 1;
    var chosen = 0; r.N.forEach(function (n, i) { if (n > r.N[chosen]) chosen = i; });
    // axes
    svg("line", { x1: ml, y1: mt + ih, x2: ml + iw, y2: mt + ih, stroke: "#39414f" }, root);
    for (var g = 0; g <= 2; g++) {
      var yy = mt + ih - ih * (g / 2);
      svg("line", { x1: ml, y1: yy, x2: ml + iw, y2: yy, stroke: "#252b36" }, root);
      svg("text", { x: ml - 6, y: yy + 3, fill: "#7d8697", "font-size": "9", "text-anchor": "end" }, root).textContent = (pmax * g / 2).toFixed(2);
    }
    for (var c = 0; c < K; c++) {
      var cx = ml + slot * (c + 0.5);
      if (c === chosen) svg("rect", { x: ml + slot * c, y: mt, width: slot, height: ih, fill: "rgba(250,220,60,0.08)" }, root);
      // prior bar (blue)
      var hp = ih * (r.P[c] / pmax);
      svg("rect", { x: cx - bw - 1, y: mt + ih - hp, width: bw, height: hp, fill: "#4a90d9", rx: 1.5 }, root);
      // visit bar (gold) with Q-coloured cap
      var hv = ih * (r.visit_dist[c] / pmax);
      svg("rect", { x: cx + 1, y: mt + ih - hv, width: bw, height: hv, fill: "#d9b64a", rx: 1.5 }, root);
      var lo = state.qDomain[0], hi = state.qDomain[1];
      if (r.N[c] > 0) svg("rect", { x: cx + 1, y: mt + ih - hv - 3, width: bw, height: 3, fill: qColor(r.Q[c], lo, hi) }, root);
      svg("text", { x: cx, y: mt + ih + 14, fill: c === chosen ? "#fadc3c" : "#7d8697", "font-size": "9", "text-anchor": "middle" }, root).textContent = c;
      // hover target
      (function (cc) {
        var hit = svg("rect", { x: ml + slot * cc, y: mt, width: slot, height: ih, fill: "transparent" }, root);
        hit.style.cursor = "pointer";
        hit.addEventListener("mousemove", function (ev) {
          showTip("<b>列 col " + cc + "</b><br><span class='tt-k'>先验 P:</span> " + r.P[cc].toFixed(3) +
            "<br><span class='tt-k'>访问 N:</span> " + r.N[cc] + " (" + (r.visit_dist[cc] * 100).toFixed(1) + "%)" +
            "<br><span class='tt-k'>价值 Q:</span> " + r.Q[cc].toFixed(3) + (cc === chosen ? "<br><b style='color:#fadc3c'>★ 选中列</b>" : ""), ev);
        });
        hit.addEventListener("mouseleave", hideTip);
      })(c);
    }
    // legend
    var lg = [["#4a90d9", "先验 P(a)"], ["#d9b64a", "访问 N(a)"], ["q", "价值 Q(a) 顶色"]];
    var lx = ml;
    lg.forEach(function (it) {
      svg("rect", { x: lx, y: 8, width: 10, height: 10, fill: it[0] === "q" ? qColor((lo + hi) / 2, lo, hi) : it[0], rx: 2 }, root);
      svg("text", { x: lx + 14, y: 17, fill: "#9aa3b2", "font-size": "10" }, root).textContent = it[1];
      lx += 14 + it[1].length * 7 + 18;
    });
    var v = r.value;
    gauge.innerHTML = "局面价值 <b>v=" + v.toFixed(3) + "</b>（归一化 return-to-go，×" + 2000 + " ≈ <b>" + Math.round(v * 2000) + "</b> 分预期剩余）· 选中列 <b style='color:#fadc3c'>col " + chosen + "</b> · 访问 " + r.total_visits + " 次";
  }

  // ---- DQN Q panel (min-max normalized) ----
  function renderQPanel(dec, root, gauge) {
    var q = dec.internals.q, K = q.length;
    var chosen = typeof dec.internals.col === "number" ? dec.internals.col : -1;
    if (chosen < 0) { chosen = 0; for (var i = 1; i < K; i++) if (q[i] > q[chosen]) chosen = i; }
    var W = 520, H = 300, ml = 56, mr = 14, mt = 30, mb = 40;
    root.setAttribute("viewBox", "0 0 " + W + " " + H);
    var iw = W - ml - mr, ih = H - mt - mb;
    var qmin = Math.min.apply(null, q), qmax = Math.max.apply(null, q);
    var span = (qmax - qmin) || 1;
    var slot = iw / K, bw = Math.max(1.5, slot * 0.6);
    svg("line", { x1: ml, y1: mt, x2: ml + iw, y2: mt, stroke: "#39414f" }, root);
    svg("line", { x1: ml, y1: mt + ih, x2: ml + iw, y2: mt + ih, stroke: "#39414f" }, root);
    svg("text", { x: ml - 6, y: mt + 3, fill: "#7d8697", "font-size": "9", "text-anchor": "end" }, root).textContent = qmax.toFixed(1);
    svg("text", { x: ml - 6, y: mt + ih + 3, fill: "#7d8697", "font-size": "9", "text-anchor": "end" }, root).textContent = qmin.toFixed(1);
    for (var c = 0; c < K; c++) {
      (function (cc) {
        var cx = ml + slot * (cc + 0.5);
        var t = (q[cc] - qmin) / span;
        var h = Math.max(2, ih * t);
        if (cc === chosen) svg("rect", { x: ml + slot * cc, y: mt, width: slot, height: ih, fill: "rgba(250,220,60,0.10)" }, root);
        svg("rect", { x: cx - bw / 2, y: mt + ih - h, width: bw, height: h,
                      fill: cc === chosen ? "#fadc3c" : "rgba(199,125,255,0.75)", rx: 1 }, root);
        var hit = svg("rect", { x: ml + slot * cc, y: mt, width: slot, height: ih, fill: "transparent" }, root);
        hit.style.cursor = "pointer";
        hit.addEventListener("mousemove", function (ev) {
          showTip("<b>列 col " + cc + "</b><br><span class='tt-k'>Q 原始值:</span> " + q[cc].toFixed(2) +
            "<br><span class='tt-k'>归一化:</span> " + t.toFixed(3) + "（min-max）" +
            (cc === chosen ? "<br><b style='color:#fadc3c'>★ 选中列（argmax Q）</b>" : ""), ev);
        });
        hit.addEventListener("mouseleave", hideTip);
      })(c);
    }
    for (var c2 = 0; c2 < K; c2 += 16)
      svg("text", { x: ml + slot * (c2 + 0.5), y: mt + ih + 14, fill: "#7d8697", "font-size": "9", "text-anchor": "middle" }, root).textContent = c2;
    var lg = [["rgba(199,125,255,0.75)", "Q(a) · min-max 归一化显示"], ["#fadc3c", "选中列"]];
    var lx = ml;
    lg.forEach(function (it) {
      svg("rect", { x: lx, y: 8, width: 10, height: 10, fill: it[0], rx: 2 }, root);
      svg("text", { x: lx + 14, y: 17, fill: "#9aa3b2", "font-size": "10" }, root).textContent = it[1];
      lx += 14 + it[1].length * 7 + 18;
    });
    gauge.innerHTML = "DQN Q 值 · 当前步 <b>min-max 归一化</b>显示（原始范围 [" + qmin.toFixed(1) + ", " + qmax.toFixed(1) +
      "]，跨度 " + span.toFixed(1) + "）· 选中列 <b style='color:#fadc3c'>col " + chosen + "</b>（argmax Q）· 悬停查看原始值";
  }

  // ---- heuristic breakdown ----
  function renderHeur(dec) {
    var root = $("#heurSvg"); clear(root);
    var stats = $("#heurStats");
    if (!dec || !dec.internals || !dec.internals.candidates || !dec.internals.candidates.length) {
      stats.innerHTML = "该决策无启发式候选分解（非启发式 agent）。"; return;
    }
    var cands = dec.internals.candidates, w = dec.internals.weights || {};
    var W = 520, H = 440, ml = 40, mr = 12, mt = 20, mb = 40;
    root.setAttribute("viewBox", "0 0 " + W + " " + H);
    var iw = W - ml - mr, ih = H - mt - mb;
    var n = cands.length, slot = iw / n, bw = Math.min(30, slot * 0.62);
    // split positive / negative magnitudes (exclude gameover from scaling)
    var maxPos = 1, maxNeg = 1, anyGo = false;
    cands.forEach(function (c) {
      if (c.terms.gameover < 0) { anyGo = true; }
      var pos = Math.max(0, c.terms.merge) + Math.max(0, c.terms.safety) + Math.max(0, c.terms.potential);
      var neg = Math.max(0, -c.terms.fruits) + Math.max(0, -c.terms.merge) + Math.max(0, -c.terms.safety) + Math.max(0, -c.terms.potential);
      maxPos = Math.max(maxPos, pos); maxNeg = Math.max(maxNeg, neg);
    });
    var baseY = mt + ih * (maxPos / (maxPos + maxNeg));
    var upScale = (baseY - mt) / maxPos, dnScale = (mt + ih - baseY) / maxNeg;
    svg("line", { x1: ml, y1: baseY, x2: ml + iw, y2: baseY, stroke: "#39414f" }, root);
    // pick best (max total among non-gameover)
    var bestI = -1, bestV = -Infinity;
    cands.forEach(function (c, i) { if (c.terms.gameover >= 0 && c.total > bestV) { bestV = c.total; bestI = i; } });
    var termColors = { merge: "#5ac878", safety: "#4a90d9", potential: "#a97ad9", fruits: "#e08a5a" };
    cands.forEach(function (c, i) {
      var cx = ml + slot * (i + 0.5);
      var isGo = c.terms.gameover < 0;
      if (i === bestI) svg("rect", { x: ml + slot * i, y: mt, width: slot, height: ih, fill: "rgba(250,220,60,0.08)" }, root);
      if (isGo) {
        svg("rect", { x: cx - bw / 2, y: mt, width: bw, height: ih, fill: "rgba(224,82,77,0.18)", stroke: "rgba(224,82,77,0.6)", "stroke-dasharray": "4 3", rx: 3 }, root);
        svg("text", { x: cx, y: baseY, fill: "#e0524d", "font-size": "18", "text-anchor": "middle" }, root).textContent = "☠";
      } else {
        // stack positives upward
        var y = baseY, order = ["potential", "safety", "merge"];
        order.forEach(function (tk) {
          var val = c.terms[tk]; if (val <= 0) return;
          var h = val * upScale; y -= h;
          svg("rect", { x: cx - bw / 2, y: y, width: bw, height: h, fill: termColors[tk], "fill-opacity": 0.9 }, root);
        });
        // negatives downward
        var y2 = baseY;
        ["fruits"].forEach(function (tk) {
          var val = c.terms[tk]; if (val >= 0) return;
          var h = (-val) * dnScale;
          svg("rect", { x: cx - bw / 2, y: y2, width: bw, height: h, fill: termColors[tk], "fill-opacity": 0.9 }, root);
          y2 += h;
        });
        // total marker
        var ty = baseY - c.total * upScale;
        svg("line", { x1: cx - bw / 2 - 2, y1: ty, x2: cx + bw / 2 + 2, y2: ty, stroke: i === bestI ? "#fadc3c" : "#e7eaf0", "stroke-width": i === bestI ? 2.5 : 1.2 }, root);
      }
      svg("text", { x: cx, y: mt + ih + 14, fill: i === bestI ? "#fadc3c" : "#7d8697", "font-size": "9", "text-anchor": "middle" }, root).textContent = "c" + c.col;
      // hover
      (function (cc) {
        var hit = svg("rect", { x: ml + slot * cc.i, y: mt, width: slot, height: ih, fill: "transparent" }, root);
        hit.style.cursor = "pointer";
        hit.addEventListener("mousemove", function (ev) {
          var t = cc.c.terms, s = cc.c.sim;
          showTip("<b>候选 col " + cc.c.col + "</b> (x=" + cc.c.x + ")<br>" +
            "<span class='tt-k'>总分:</span> <b>" + cc.c.total.toFixed(1) + "</b>" + (cc.i === bestI ? " ★" : "") + "<br>" +
            "<span class='tt-k'>合并:</span> " + t.merge.toFixed(1) + "  <span class='tt-k'>安全:</span> " + t.safety.toFixed(1) + "<br>" +
            "<span class='tt-k'>潜力:</span> " + t.potential.toFixed(1) + "  <span class='tt-k'>−数量:</span> " + t.fruits.toFixed(1) + "<br>" +
            (t.gameover < 0 ? "<b style='color:#e0524d'>☠ 该落点导致游戏结束</b><br>" : "") +
            "<span class='tt-k'>前瞻:</span> 得分+" + s.score_gain + ", 堆顶y=" + (s.top_y == null ? "—" : s.top_y.toFixed(0)) + ", 余" + s.fruit_count + "果", ev);
        });
        hit.addEventListener("mouseleave", hideTip);
      })({ c: c, i: i });
    });
    // legend
    var lx = ml, items = [["merge", "合并"], ["safety", "安全(堆顶)"], ["potential", "潜力"], ["fruits", "−数量"]];
    items.forEach(function (it) {
      svg("rect", { x: lx, y: 6, width: 9, height: 9, fill: termColors[it[0]], rx: 2 }, root);
      svg("text", { x: lx + 13, y: 14, fill: "#9aa3b2", "font-size": "9.5" }, root).textContent = it[1];
      lx += 13 + it[1].length * 10 + 14;
    });
    stats.innerHTML = "候选列 <b>" + n + "</b> · 权重 merge=" + w.merge + " safety=" + w.safety + " potential=" + w.potential + " fruits=" + w.fruits + " gameover=" + w.gameover +
      (anyGo ? " · <span style='color:#e0524d'>含 ☠ 致命落点</span>" : "") + " · 选中 <b style='color:#fadc3c'>col " + (bestI >= 0 ? cands[bestI].col : "?") + "</b>";
  }

  // ------------------------------------------------------------------ curve
  function renderCurve() {
    var s = $("#curve"); clear(s);
    var decs = TRACE.decisions; if (!decs.length) return;
    var W = 560, H = 130, ml = 38, mr = 40, mt = 14, mb = 22;
    s.setAttribute("viewBox", "0 0 " + W + " " + H);
    var iw = W - ml - mr, ih = H - mt - mb;
    var maxScore = Math.max.apply(null, decs.map(function (d) { return d.score; })) || 1;
    var maxH = Math.max.apply(null, decs.map(function (d) { return d.max_height; })) || 1;
    function X(i) { return ml + (decs.length > 1 ? i / (decs.length - 1) : 0) * iw; }
    function Yscore(v) { return mt + ih - v / maxScore * ih; }
    function Yh(v) { return mt + ih - v / maxH * ih; }
    // grid
    svg("line", { x1: ml, y1: mt + ih, x2: ml + iw, y2: mt + ih, stroke: "#39414f" }, s);
    // score line
    var dp = decs.map(function (d, i) { return (i ? "L" : "M") + X(i).toFixed(1) + "," + Yscore(d.score).toFixed(1); }).join(" ");
    svg("path", { d: dp, fill: "none", stroke: "#e0524d", "stroke-width": 2 }, s);
    // height line
    var hp = decs.map(function (d, i) { return (i ? "L" : "M") + X(i).toFixed(1) + "," + Yh(d.max_height).toFixed(1); }).join(" ");
    svg("path", { d: hp, fill: "none", stroke: "#4a90d9", "stroke-width": 1.4, "stroke-dasharray": "4 3" }, s);
    // merge markers
    decs.forEach(function (d, i) { if (d.reward > 0) svg("circle", { cx: X(i), cy: Yscore(d.score), r: 2.4, fill: "#fadc3c" }, s); });
    // labels
    svg("text", { x: ml - 6, y: mt + 8, fill: "#e0524d", "font-size": "9", "text-anchor": "end" }, s).textContent = maxScore;
    svg("text", { x: ml - 6, y: mt + ih, fill: "#e0524d", "font-size": "9", "text-anchor": "end" }, s).textContent = "0";
    svg("text", { x: ml + iw + 6, y: mt + 8, fill: "#4a90d9", "font-size": "9" }, s).textContent = maxH.toFixed(0);
    svg("text", { x: 6, y: 12, fill: "#9aa3b2", "font-size": "9.5" }, s).textContent = "分数(红) / 堆顶高(蓝) / 合成(金点)";
    // current-step marker
    var ci = currentDecisionIdx();
    if (ci >= 0) {
      svg("line", { id: "curveMark", x1: X(ci), y1: mt, x2: X(ci), y2: mt + ih, stroke: "#5adcf0", "stroke-width": 1.4 }, s);
      svg("circle", { cx: X(ci), cy: Yscore(decs[ci].score), r: 3.4, fill: "#5adcf0", stroke: "#0e1015" }, s);
    }
    // click to jump
    var hit = svg("rect", { x: ml, y: mt, width: iw, height: ih, fill: "transparent" }, s);
    hit.style.cursor = "pointer";
    hit.addEventListener("click", function (ev) {
      var rect = s.getBoundingClientRect();
      var px = (ev.clientX - rect.left) / rect.width * W;
      var idx = Math.round(clamp((px - ml) / iw, 0, 1) * (decs.length - 1));
      jumpToDecision(idx);
    });
  }
  function updateCurveMarker() {
    var s = $("#curve"); var old = s.querySelector("#curveMark");
    var ci = currentDecisionIdx(); if (ci < 0) { if (old) old.remove(); return; }
    var decs = TRACE.decisions, W = 560, ml = 38, mr = 40, iw = W - ml - mr;
    var x = ml + (decs.length > 1 ? ci / (decs.length - 1) : 0) * iw;
    if (old) { old.setAttribute("x1", x); old.setAttribute("x2", x); }
    else renderCurve();
  }

  // ------------------------------------------------------------------ arena
  var AGENT_COLORS = { random: "#8b93a3", heuristic: "#5adcf0", alphazero: "#fadc3c" };
  function agentColor(a) {
    if (a.indexOf("mattjacobs") === 0) return "#c77dff";
    if (a.indexOf("moonfloof") === 0) return "#ff9e4a";
    if (a.indexOf("w3b_mlp") >= 0 || a.indexOf("mlp_deep") >= 0) return "#66e0b0";
    if (a.indexOf("w3b_tf") >= 0 || a.indexOf("tf_deep") >= 0) return "#ff7ab8";
    return AGENT_COLORS[a] || "#aaaaaa";
  }
  function currentName() { var sel = $("#gameSelect"); return sel ? sel.value : null; }
  function renderArena() {
    var s = $("#arena"); clear(s);
    var traces = (window.SUIKA_MANIFEST && window.SUIKA_MANIFEST.traces) || [];
    var withSeries = traces.filter(function (t) { return t.score_series && t.score_series.length; });
    if (!withSeries.length) return;
    var W = 560, H = 180, ml = 40, mr = 196, mt = 16, mb = 24;
    s.setAttribute("viewBox", "0 0 " + W + " " + H);
    var iw = W - ml - mr, ih = H - mt - mb;
    var maxSteps = Math.max.apply(null, withSeries.map(function (t) { return t.score_series.length; }));
    var maxScore = Math.max.apply(null, withSeries.map(function (t) { return Math.max.apply(null, t.score_series); })) || 1;
    function X(i) { return ml + (maxSteps > 1 ? i / (maxSteps - 1) : 0) * iw; }
    function Y(v) { return mt + ih - v / maxScore * ih; }
    svg("line", { x1: ml, y1: mt + ih, x2: ml + iw, y2: mt + ih, stroke: "#39414f" }, s);
    svg("line", { x1: ml, y1: mt, x2: ml, y2: mt + ih, stroke: "#39414f" }, s);
    svg("text", { x: ml - 6, y: mt + 8, fill: "#7d8697", "font-size": "9", "text-anchor": "end" }, s).textContent = maxScore;
    svg("text", { x: ml - 6, y: mt + ih, fill: "#7d8697", "font-size": "9", "text-anchor": "end" }, s).textContent = "0";
    var cur = currentName(), ly = mt + 6;
    withSeries.forEach(function (t) {
      var col = agentColor(t.agent);
      var isCur = (t.name === cur);
      var dp = t.score_series.map(function (v, i) { return (i ? "L" : "M") + X(i).toFixed(1) + "," + Y(v).toFixed(1); }).join(" ");
      var attrs = { d: dp, fill: "none", stroke: col, "stroke-width": isCur ? 2.8 : 1.5, "stroke-opacity": isCur ? 1 : 0.75 };
      if (t.rule_mode !== "loop") attrs["stroke-dasharray"] = "5 3";
      svg("path", attrs, s);
      svg("rect", { x: ml + iw + 12, y: ly - 8, width: 12, height: 3, fill: col }, s);
      svg("text", { x: ml + iw + 28, y: ly - 3, fill: isCur ? "#ffffff" : "#9aa3b2", "font-size": "9" }, s)
        .textContent = t.agent + "·s" + t.seed + (t.rule_mode !== "loop" ? "·" + t.rule_mode : "") + " =" + t.final_score + " " + t.max_fruit_name;
      ly += 17;
    });
  }

  // ------------------------------------------------------------------ render orchestration
  function renderAll() {
    drawBoard(); renderHUD();
    var di = currentDecisionIdx();
    var dec = di >= 0 ? TRACE.decisions[di] : null;
    if (di !== state.lastDec) {
      state.lastDec = di;
      renderAgentCard(dec); updateTabs(dec); renderPanels(dec);
      var ss = $("#stepSelect"); if (di >= 0) ss.value = di;
      $("#decisionHint").textContent = di >= 0 ? ("正在查看第 " + dec.step + " 步的决策") : "尚未落子";
      renderCurve();
    } else { updateCurveMarker(); }
  }

  function jumpToDecision(i) {
    var d = TRACE.decisions[i]; if (!d) return;
    pause();
    state.frameIdx = clamp(d.frame_start, 0, TRACE.frames.length - 1);
    state.lastDec = -2; renderAll();
  }

  // ------------------------------------------------------------------ playback
  var rafId = null, acc = 0, lastT = 0;
  function tick(ts) {
    if (!state.playing) return;
    if (!lastT) lastT = ts;
    var dt = ts - lastT; lastT = ts;
    // advance at the trace's STORED frame rate (fps / stride) so 1x plays the
    // complete captured physics back at true real-time speed.
    var fpsStored = (META.fps || 60) / (META.frame_stride || 1);
    acc += (dt / 1000) * fpsStored * state.speed;
    while (acc >= 1) {
      acc -= 1;
      if (state.frameIdx < TRACE.frames.length - 1) state.frameIdx++;
      else { pause(); break; }
    }
    renderAll();
    if (state.playing) rafId = requestAnimationFrame(tick);
  }
  function play() {
    if (!TRACE) return;
    if (state.frameIdx >= TRACE.frames.length - 1) state.frameIdx = 0;
    state.playing = true; lastT = 0; acc = 0;
    $("#playBtn").textContent = "⏸ 暂停";
    rafId = requestAnimationFrame(tick);
  }
  function pause() { state.playing = false; $("#playBtn").textContent = "▶ 播放"; if (rafId) cancelAnimationFrame(rafId); }
  function togglePlay() { state.playing ? pause() : play(); }

  // ------------------------------------------------------------------ human play (live engine)
  function syncSelects(v) {
    var a = $("#gameSelect"), b = $("#playGameSelect");
    if (a) a.value = v;
    if (b) b.value = v;
  }
  // The bar stays visible while this page is opened as the play page
  // (?game=__play), so a match can be picked for full-trajectory preview and
  // the user can come back to playing without hunting in the top bar.
  function playBar(on) {
    var b = $("#playBar"); if (!b) return;
    b.style.display = (on || state.pinned) ? "flex" : "none";
    document.querySelectorAll("#playBar .liveonly").forEach(function (el) {
      el.style.display = on ? "flex" : "none";
    });
    if (!on && state.pinned) {
      $("#playHint").textContent = "回放中 · 在「对局」里选任意模型轨迹看完整一局，或选 🎮 人类对局继续亲手玩";
    }
  }
  function exitPlay() {
    state.play.active = false; state.play.animating = false;
    state.play.current = null;
    playBar(false);
  }

  function enterPlay() {
    pause();
    state.play.active = true; state.play.animating = false; state.play.done = false;
    state.play.hoverX = null;
    syncSelects("__play__");
    playBar(true);
    $("#playHint").textContent = "连接实时引擎…";
    fetch("/api/new?rule=" + state.play.rule).then(function (r) { return r.json(); }).then(function (d) {
      if (!d.ok) throw new Error("api");
      setupLiveTrace(d);
      $("#playHint").textContent = "点击棋盘任意位置落子（真实 pymunk 物理逐帧回放）";
    }).catch(function (e) {
      $("#playHint").innerHTML = "实时引擎未连接：请在服务端运行 <code>python -m viz.play_server --port 8801</code>（与本页同端口）后刷新";
      state.play.active = false; playBar(true);
    });
  }

  function setupLiveTrace(d) {
    var meta = d.meta;
    meta.agent = "human"; meta.steps = 0; meta.final_score = 0;
    meta.max_fruit_type = 0; meta.max_fruit_name = meta.fruit_table[0].name;
    meta.game_over = false; meta.num_frames = 1;
    TRACE = { meta: meta, frames: [d.frame], decisions: [] };
    META = meta;
    decisionAtFrame = [-1];
    WORLD = meta.world; T = fitTransform();
    $("#engineBadge").textContent = "引擎: live-pymunk（实时）";
    $("#ruleBadge").textContent = ruleLabel(meta);
    state.frameIdx = 0; state.lastDec = -2; state.play.current = d.current;
    state.play.next = d.next;
    state.play.done = false; state.play.animating = false;
    var sl = $("#frameSlider"); sl.max = 0; sl.value = 0;
    var ss = $("#stepSelect"); clear(ss);
    renderCurve(); renderAll();
  }

  function canvasToWorldX(clientX) {
    var cv = $("#board"), rect = cv.getBoundingClientRect();
    var px = (clientX - rect.left) * (CW / rect.width);
    return (px - T.X(0)) / T.s;
  }

  function playDrop(wx) {
    if (!state.play.active || state.play.animating || state.play.done) return;
    var cur = state.play.current; if (!cur) return;
    var pa = META.play_area;
    wx = clamp(wx, pa.left + cur.radius, pa.right - cur.radius);
    state.play.animating = true; state.play.hoverX = null;
    $("#playHint").textContent = "物理结算中…";
    fetch("/api/drop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ x: wx }) })
      .then(function (r) { return r.json(); }).then(function (d) {
        if (!d.ok) throw new Error("drop");
        appendLiveDrop(d);
      }).catch(function () {
        state.play.animating = false;
        $("#playHint").textContent = "落子失败（引擎连接中断）";
      });
  }

  function appendLiveDrop(d) {
    var f0 = TRACE.frames.length;
    d.frames.forEach(function (f) { f.i = TRACE.frames.length; TRACE.frames.push(f); });
    var f1 = TRACE.frames.length;
    var lastFr = TRACE.frames[f1 - 1];
    var topY = META.play_area.bot;
    lastFr.b.forEach(function (b) { topY = Math.min(topY, b.y - b.r); });
    var maxT = 0; lastFr.b.forEach(function (b) { maxT = Math.max(maxT, b.t); });
    var idx = TRACE.decisions.length;
    TRACE.decisions.push({
      step: idx + 1, kind: "human", col: null, x: Math.round(d.x * 10) / 10,
      current: d.dropped, next: d.current, reward: Math.round(d.reward),
      score: d.score, fruit_count: d.fruit_count,
      max_height: Math.round((META.play_area.bot - topY) * 10) / 10,
      game_over: d.done, frame_start: f0, frame_end: f1, internals: {}
    });
    for (var f = f0; f < f1; f++) decisionAtFrame[f] = idx;
    META.final_score = d.score; META.steps = idx + 1;
    META.max_fruit_type = maxT; META.max_fruit_name = nameOf(maxT);
    META.game_over = d.done; META.num_frames = f1;
    state.play.current = d.current;
    state.play.next = d.next;
    state.play.done = !!d.done;
    $("#frameSlider").max = f1 - 1;
    var ss = $("#stepSelect"), o = document.createElement("option");
    o.value = idx; o.textContent = "步 " + (idx + 1) + " · 人类 · x" + Math.round(d.x) + " · " + d.score + "分";
    ss.appendChild(o);
    // animate the new frames at real-time, then unlock input
    state.frameIdx = f0; state.lastDec = -2;
    var target = f1 - 1, acc2 = 0, last2 = 0;
    function stepAnim(ts) {
      if (!state.play.active) { state.play.animating = false; return; }
      if (!last2) last2 = ts;
      acc2 += (ts - last2) / 1000 * (META.fps || 60) * state.speed; last2 = ts;
      while (acc2 >= 1 && state.frameIdx < target) { acc2 -= 1; state.frameIdx++; }
      renderAll();
      if (state.frameIdx < target) requestAnimationFrame(stepAnim);
      else {
        state.play.animating = false;
        $("#playHint").textContent = state.play.done
          ? "游戏结束！点「新局」再来一盘" : "点击棋盘落子";
      }
    }
    requestAnimationFrame(stepAnim);
  }

  // ------------------------------------------------------------------ info drawer
  function benchHTML() {
    var B = window.SUIKA_BENCH;
    if (!B) return "";
    var rows = (B.controlled || []).slice().sort(function (a, b) { return (b.mean || 0) - (a.mean || 0); });
    var tr = rows.map(function (r, i) {
      return "<tr><td>" + (i + 1) + "</td><td><b>" + r.label + "</b></td><td>" + r.cluster + "</td><td>" +
        r.train + "</td><td>" + r.seeds + "</td><td><b>" + fmt1(r.mean) + "</b></td><td>" +
        (r.median == null ? "—" : r.median) + "</td><td>" + (r.max == null ? "—" : r.max) + "</td><td>" +
        (r.min == null ? "—" : r.min) + "</td><td>" + pct(r.p2000) + "</td></tr>";
    }).join("");
    var comm = (B.community || []).map(function (c) {
      return "<tr><td><b>" + c.label + "</b></td><td>" + (c.native_mean == null ? "—" : fmt1(c.native_mean)) +
        (c.native_seeds ? " (" + c.native_seeds + " 种子)" : "—") + "</td><td>" +
        (c.ours_mean == null ? "—" : fmt1(c.ours_mean)) + "</td><td>" +
        (c.ours_max == null ? "—" : c.ours_max) + "</td><td>" + c.note + "</td></tr>";
    }).join("");
    return "<h3>基线榜（同环境受控 · greedy / deterministic）</h3>" +
      "<p class='benchnote'>全部在本项目引擎（" + B.engine + "）内独立复评，<b>只有这张表与下方曲线同环境可比</b>；" +
      "两个 cluster policy 由 A100 集群训练、在本环境重新跑 16 个种子。数据生成于 " + B.generated + "。</p>" +
      "<table><tr><th>#</th><th>智能体</th><th>集群</th><th>训练量 / 配方</th><th>种子</th><th>mean</th><th>median</th><th>max</th><th>min</th><th>≥2000</th></tr>" +
      tr + "</table>" +
      "<h3>社区 baseline（原生环境 vs 迁入本环境）</h3>" +
      "<table><tr><th>项目</th><th>原生 mean</th><th>本环境 mean</th><th>本环境 max</th><th>备注</th></tr>" + comm + "</table>";
  }

  function buildInfo() {
    var ft = (META && META.fruit_table) || [];
    var rows = ft.map(function (f) {
      return "<tr><td><span class='swatch' style='background:" + rgb(f.color) + "'></span>" + f.name + "</td><td>" + f.type + "</td><td>" + f.radius + "</td><td>" + f.points + "</td></tr>";
    }).join("");
    $("#infoContent").innerHTML =
      "<h3>引擎还原</h3><p>物理内核来自开源 <code>Ole-Batting/suika</code>（pygame + <code>pymunk 6.11.1</code>，Chipmunk 绑定）。" +
      "本项目用 <code>suika/part2/suika_env.py</code> 封装成 headless 的 <code>reset/step/get_state/render</code> 接口；可视化采集层 <code>viz/capture.py</code> 进一步逐物理帧抓取每个刚体的位置/速度/角度与合成事件，<b>不改动任何引擎逻辑</b>。</p>" +
      "<h3>训练策略与早期对照</h3>" +
      "<p><b>Set Transformer + Dueling DQN</b>：55.9M 参数，从当前水果、下一个水果、场上水果与棋盘尺寸预测 128 个落点的 Q 值，部署时直接取 argmax；不使用未来随机种子或搜索。新纪录来自已保存动作的确定性回放，旧纪录保留完整 Q128。</p>" +
      "<p><b>启发式 + 1 步真物理前瞻</b>（<code>part2/ai_agent.py</code>）：对每个候选列，从局面快照重建一个独立 pymunk 空间、真的把水果投下去让物理稳定，再按 <code>合并/安全(堆顶)/潜力(同级相邻)/−数量</code> 打分，选最高分列。「启发式分解」面板展示每个候选的逐项打分。</p>" +
      "<p><b>AlphaZero</b>（<code>rl/mcts.py</code> + <code>rl/net.py</code>）：PUCT 蒙特卡洛树搜索 + Object-Transformer 策略/价值网（批量叶子 + 虚拟损失）。每步跑数百次模拟，用网络先验 P 引导展开、用价值 v 评估叶子（不随机 rollout 到终局），按访问次数 N 选列。「MCTS 搜索树」面板中边粗细∝访问 N、颜色∝动作价值 Q；「策略/价值」面板对比先验 P 与访问分布 N。</p>" +
      "<h3>水果等级表</h3><table><tr><th>名称</th><th>type</th><th>半径</th><th>分值</th></tr>" + rows + "</table>" +
      benchHTML() +
      "<h3>本局规则与评测</h3><p>" + ruleLabel(META || {}) + "。Wave6 采用 pymunk settle 步进；720 指死亡线到地板的距离，550 与 448 是不同棋盘宽度。精选高分局用于观看实际行为，不能代替多种子平均成绩。训练曲线使用反复评测的固定开发种子。</p>" +
      "<h3>操作</h3><p>顶部选择对局；左侧播放/拖动帧滑块/选择决策步；右侧切换 MCTS 树 / 策略价值 / 启发式分解三个面板；画布右上切换叠加层（速度矢量/合成事件/落点瞄准/候选打分/等级数字/死亡线）；底部曲线可点击跳转。</p>";
  }

  // ------------------------------------------------------------------ events
  function bind() {
    $("#playBtn").onclick = togglePlay;
    $("#stepFwd").onclick = function () { pause(); if (state.frameIdx < TRACE.frames.length - 1) state.frameIdx++; state.lastDec = -2; renderAll(); };
    $("#stepBack").onclick = function () { pause(); if (state.frameIdx > 0) state.frameIdx--; state.lastDec = -2; renderAll(); };
    $("#frameSlider").oninput = function (e) { pause(); state.frameIdx = parseInt(e.target.value, 10); renderAll(); };
    $("#speedSel").onchange = function (e) { state.speed = parseFloat(e.target.value); };
    var ov = { tgVelocity: "velocity", tgMerges: "merges", tgAim: "aim", tgCand: "cand", tgLabels: "labels", tgDead: "dead" };
    Object.keys(ov).forEach(function (id) {
      var el = $("#" + id); el.checked = state.overlays[ov[id]];
      el.onchange = function () { state.overlays[ov[id]] = el.checked; drawBoard(); };
    });
    document.querySelectorAll(".tab").forEach(function (t) { t.onclick = function () { if (isTabEnabled(t.dataset.tab)) switchTab(t.dataset.tab); }; });
    // human play: click to drop, hover to aim
    var cv = $("#board");
    cv.addEventListener("click", function (ev) {
      if (!state.play.active) return;
      playDrop(canvasToWorldX(ev.clientX));
    });
    cv.addEventListener("mousemove", function (ev) {
      if (!state.play.active || state.play.animating || state.play.done) return;
      var wx = canvasToWorldX(ev.clientX);
      var pa = META.play_area, cur = state.play.current;
      if (cur) wx = clamp(wx, pa.left + cur.radius, pa.right - cur.radius);
      if (state.play.hoverX !== wx) { state.play.hoverX = wx; drawBoard(); }
    });
    cv.addEventListener("mouseleave", function () {
      if (state.play.hoverX != null) { state.play.hoverX = null; drawBoard(); }
    });
    var pr = $("#playRule");
    if (pr) pr.onchange = function () { state.play.rule = pr.value; enterPlay(); };
    var pn = $("#playNew");
    if (pn) pn.onclick = function () { enterPlay(); };
    $("#infoBtn").onclick = function () { buildInfo(); $("#infoDrawer").classList.add("open"); };
    $("#infoClose").onclick = function () { $("#infoDrawer").classList.remove("open"); };
    $("#infoDrawer").onclick = function (e) { if (e.target.id === "infoDrawer") $("#infoDrawer").classList.remove("open"); };
    document.addEventListener("keydown", function (e) {
      if (e.code === "Space") { e.preventDefault(); togglePlay(); }
      else if (e.code === "ArrowRight") { $("#stepFwd").onclick(); }
      else if (e.code === "ArrowLeft") { $("#stepBack").onclick(); }
    });
    window.addEventListener("resize", function () { /* canvas is fixed logical size; nothing to do */ });
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    setupCanvas(); bind(); loadManifest();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
