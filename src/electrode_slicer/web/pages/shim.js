// Static build only: answer the page's /api/... requests from a Web Worker
// running the slicer in Pyodide, rather than from a server.  Everything else
// the page fetches goes to the network as usual.
(() => {
  const worker = new Worker("worker.js");
  const pending = new Map();
  let next = 0;

  const banner = document.createElement("div");
  banner.setAttribute("role", "status");
  // The privacy note and download size live in a hover tip, not the banner text.
  banner.title = "Runs entirely in your browser; your STL is never uploaded. " +
    "The first visit downloads about 35 MB.";
  banner.style.cssText = "position:fixed;left:50%;bottom:16px;transform:translateX(-50%);" +
    "z-index:100;padding:7px 14px;border-radius:999px;font:12.5px 'IBM Plex Sans',system-ui,sans-serif;" +
    "background:var(--fg);color:var(--bg);box-shadow:0 2px 10px rgba(0,0,0,.25);max-width:90vw";
  const show = (text) => {
    banner.textContent = text;
    if (!banner.isConnected) document.body.append(banner);
  };
  document.addEventListener("DOMContentLoaded", () =>
    show("Loading slicer…"));

  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === "status") show(msg.text);
    else if (msg.type === "ready") banner.remove();
    else if (msg.type === "failed") {
      show(`The slicer could not start: ${msg.text}`);
      banner.style.background = "var(--infill)";
    } else if (msg.type === "reply") {
      const resolve = pending.get(msg.id);
      pending.delete(msg.id);
      resolve(new Response(msg.body, {
        status: msg.status, headers: { "Content-Type": msg.kind, ...msg.headers },
      }));
    }
  };

  const network = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = typeof input === "string" ? input : input.url;
    if (!url.startsWith("/api/")) return network(input, init);
    return new Promise((resolve) => {
      const id = ++next;
      pending.set(id, resolve);
      worker.postMessage({ id, method: (init.method || "GET").toUpperCase(), route: url,
                           body: typeof init.body === "string" ? init.body : "" });
    });
  };
})();
