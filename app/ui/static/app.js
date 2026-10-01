// Polymom web UI. The API key lives in localStorage and is sent as X-API-Key on
// every HTMX request and fetch, so the UI uses exactly the same auth as the API.
(() => {
  "use strict";
  const KEY = "polymom.apiKey";
  const THEME = "polymom.theme";
  const API = "/api/v1/meetings";
  const store = {
    get: (k) => { try { return localStorage.getItem(k) || ""; } catch { return ""; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const $ = (sel, root = document) => root.querySelector(sel);

  function notify(message, isError = false) {
    const el = $("#notice");
    el.textContent = message;
    el.classList.toggle("err", isError);
    el.hidden = !message;
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const key = store.get(KEY);
    if (key) headers.set("X-API-Key", key);
    const res = await fetch(path, { ...options, headers });
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try { const body = await res.json(); msg = body.error?.message || msg; } catch { /* not JSON */ }
      if (res.status === 401) msg += " Enter your API key at the top of the page.";
      throw new Error(msg);
    }
    return res;
  }

  // --- API key and theme ---------------------------------------------------
  function initHeader() {
    const input = $("#api-key");
    input.value = store.get(KEY);
    $("#key-form").addEventListener("submit", (e) => {
      e.preventDefault();
      store.set(KEY, input.value.trim());
      notify("API key saved.");
      document.body.dispatchEvent(new Event("keychange"));
    });
    const toggle = $("#theme-toggle");
    const apply = (t) => {
      if (t) document.documentElement.dataset.theme = t;
      toggle.setAttribute("aria-pressed", String(document.documentElement.dataset.theme === "dark"));
    };
    apply(store.get(THEME));
    toggle.addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme === "dark" ||
        (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark";
      store.set(THEME, next);
      apply(next);
    });
  }

  document.addEventListener("htmx:configRequest", (e) => {
    const key = store.get(KEY);
    if (key) e.detail.headers["X-API-Key"] = key;
  });
  document.addEventListener("htmx:responseError", (e) => {
    const xhr = e.detail.xhr;
    let msg = `Request failed (${xhr.status}).`;
    try { msg = JSON.parse(xhr.responseText).error.message; } catch { /* not JSON */ }
    if (xhr.status === 401) msg += " Enter your API key at the top of the page.";
    notify(msg, true);
  });

  // --- Upload ----------------------------------------------------------------
  function initUpload() {
    const form = $("#upload-form");
    if (!form) return;
    const input = $("#file");
    const zone = $("#dropzone");
    const show = () => { $("#file-name").textContent = input.files[0]?.name || ""; };
    input.addEventListener("change", show);
    zone.tabIndex = 0;
    zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, () => zone.classList.remove("over")));
    zone.addEventListener("drop", (e) => { e.preventDefault(); input.files = e.dataTransfer.files; show(); });

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!input.files.length) { notify("Choose a file first.", true); return; }
      const data = new FormData();
      data.append("file", input.files[0]);
      const title = $("#title").value.trim();
      if (title) data.append("title", title);
      const speakers = $("#expected_speakers").value;
      if (speakers) data.append("expected_speakers", speakers);
      form.querySelectorAll("input[name=languages]:checked").forEach((c) => data.append("languages", c.value));
      uploadWithProgress(data);
    });
  }

  function uploadWithProgress(data) {
    // XHR, not fetch: fetch has no upload progress events.
    const btn = $("#upload-btn");
    const bar = $("#upload-progress");
    btn.disabled = true; bar.hidden = false; bar.value = 0;
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API);
    const key = store.get(KEY);
    if (key) xhr.setRequestHeader("X-API-Key", key);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.value = (100 * e.loaded) / e.total; };
    xhr.onloadend = async () => {
      btn.disabled = false; bar.hidden = true;
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* network error */ }
      if (xhr.status !== 200 && xhr.status !== 202) {
        notify(body.error?.message || `Upload failed (${xhr.status || "network error"}).`, true);
        return;
      }
      try {
        if (!body.duplicate) await api(`${API}/${body.meeting_id}/process`, { method: "POST" });
        location.href = `/meetings/${body.meeting_id}`;
      } catch (err) { notify(err.message, true); }
    };
    xhr.send(data);
  }

  // --- Meeting detail ---------------------------------------------------------
  const detail = () => $("#detail");
  const meetingId = () => detail()?.dataset.meetingId;
  const refresh = () => htmx.trigger(detail(), "refresh");

  async function loadAudio() {
    const player = $("#player");
    if (!player || player.dataset.loaded) return;
    player.dataset.loaded = "1";
    try {
      const res = await api(`${API}/${meetingId()}/audio`);
      player.src = URL.createObjectURL(await res.blob());
    } catch (err) { player.replaceWith(Object.assign(document.createElement("p"), { className: "muted", textContent: `Audio unavailable: ${err.message}` })); }
  }

  function seek(seconds) {
    const player = $("#player");
    if (!player) return;
    player.currentTime = Number(seconds) || 0;
    player.play().catch(() => { /* autoplay blocked until the user interacts */ });
  }

  function highlight() {
    const player = $("#player");
    const t = player.currentTime;
    let active = null;
    document.querySelectorAll("#transcript li").forEach((li) => {
      const on = t >= Number(li.dataset.start) && t < Number(li.dataset.end);
      li.classList.toggle("active", on);
      if (on && !active) active = li;
    });
    if (active && !player.paused) active.scrollIntoView({ block: "nearest" });
  }

  async function download(format) {
    const res = await api(`${API}/${meetingId()}/export?format=${format}`);
    const name = /filename="([^"]+)"/.exec(res.headers.get("Content-Disposition") || "")?.[1] || `minutes.${format}`;
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(await res.blob()), download: name });
    document.body.append(a); a.click(); a.remove();
  }

  const actions = {
    export: (el) => download(el.dataset.format),
    reprocess: async () => { await api(`${API}/${meetingId()}/process?force=true`, { method: "POST" }); refresh(); },
    cancel: async (el) => { await api(`${API}/${el.dataset.meeting}/cancel`, { method: "POST" }); notify("Cancellation requested; the run stops after the current stage."); },
    delete: async () => {
      if (!confirm("Delete this meeting, its audio and all results?")) return;
      await api(`${API}/${meetingId()}`, { method: "DELETE" });
      location.href = "/";
    },
  };

  document.addEventListener("click", async (e) => {
    const seekEl = e.target.closest("[data-seek]");
    if (seekEl) { seek(seekEl.dataset.seek); return; }
    const actionEl = e.target.closest("[data-action]");
    if (!actionEl) return;
    actionEl.disabled = true;
    try { await actions[actionEl.dataset.action](actionEl); } catch (err) { notify(err.message, true); } finally { actionEl.disabled = false; }
  });

  document.addEventListener("submit", async (e) => {
    if (e.target.id !== "rename-form") return;
    e.preventDefault();
    const names = {};
    new FormData(e.target).forEach((v, k) => { names[k] = String(v).trim() || null; });
    try {
      await api(`${API}/${meetingId()}/speakers`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ names }),
      });
      notify("Speaker names saved.");
      refresh();
    } catch (err) { notify(err.message, true); }
  });

  document.addEventListener("htmx:afterSwap", () => {
    loadAudio();
    const player = $("#player");
    if (player && !player.dataset.bound) { player.dataset.bound = "1"; player.addEventListener("timeupdate", highlight); }
  });

  document.addEventListener("DOMContentLoaded", () => { initHeader(); initUpload(); });
})();
