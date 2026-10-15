// The slicer, in the browser: Pyodide runs the same Python the local server
// does, and the page's requests arrive here instead of over HTTP.
importScripts("https://cdn.jsdelivr.net/pyodide/v{{PYODIDE}}/full/pyodide.js");

const say = (text) => self.postMessage({ type: "status", text });

const ready = (async () => {
  say("Loading Python…");
  const py = await loadPyodide();
  say("Loading numpy, scipy and shapely…");
  await py.loadPackage(["numpy", "scipy", "shapely", "contourpy", "charset-normalizer",
                        "micropip"]);
  say("Loading trimesh and networkx…");
  const micropip = py.pyimport("micropip");
  // Pyodide's networkx lists matplotlib (for its drawing functions, which
  // nothing here calls) -- a few megabytes the first visit would download
  // for nothing.  networkx itself is pure Python and needs none of it.
  await micropip.install(py.toPy(["networkx", "decorator"]), { deps: false });
  await micropip.install("trimesh=={{TRIMESH}}");
  say("Loading the slicer…");
  const zip = await (await fetch("electrode_slicer.zip", { cache: "no-cache" })).arrayBuffer();
  py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide" });
  const handle = py.pyimport("electrode_slicer.web.pages").handle;
  self.postMessage({ type: "ready" });
  return handle;
})();
ready.catch((err) => self.postMessage({ type: "failed", text: String(err) }));

self.onmessage = async (event) => {
  const { id, method, route, body } = event.data;
  try {
    const handle = await ready;
    const reply = handle(method, route, body);
    const [status, data, kind, headers] = reply.toJs({ dict_converter: Object.fromEntries });
    reply.destroy();
    self.postMessage({ type: "reply", id, status, body: data, kind, headers }, [data.buffer]);
  } catch (err) {
    const text = JSON.stringify({ error: `the slicer failed in the browser: ${err}` });
    self.postMessage({ type: "reply", id, status: 500,
                       body: new TextEncoder().encode(text), kind: "application/json", headers: {} });
  }
};
