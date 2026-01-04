(() => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("image-input");
  const results = document.getElementById("results");
  const labelEl = document.getElementById("prediction-label");
  const scoreList = document.getElementById("score-list");
  const preview = document.getElementById("preview");
  const apiBaseEl = document.getElementById("api-base");
  const apiBaseInput = document.getElementById("api-base-input");
  const apiStatus = document.getElementById("api-status");
  const errorMsg = document.getElementById("error-msg");

  const defaultBase = "http://localhost:8000";
  let apiBase = localStorage.getItem("apiBase") || defaultBase;
  applyApiBase(apiBase);
  healthCheck(apiBase);

  apiBaseInput?.addEventListener("change", () => {
    apiBase = normalizeUrl(apiBaseInput.value) || defaultBase;
    applyApiBase(apiBase);
    localStorage.setItem("apiBase", apiBase);
    healthCheck(apiBase);
  });

  const setLoading = (isLoading) => {
    form.querySelector(".cta").textContent = isLoading ? "Predicting..." : "Predict";
    form.querySelector(".cta").disabled = isLoading;
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!fileInput.files?.length) return;

    const file = fileInput.files[0];
    
    try {
      setLoading(true);
      clearError();
      apiBase = normalizeUrl(apiBaseInput?.value || apiBase) || defaultBase;
      applyApiBase(apiBase);
      localStorage.setItem("apiBase", apiBase);

      // Show image preview
      showImage(file);

      // Send image to backend
      const data = new FormData();
      data.append("file", file);
      const res = await fetch(`${apiBase}/predict_image`, {
        method: "POST",
        body: data,
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `Request failed (${res.status})`);
      }
      const payload = await res.json();
      renderPrediction(payload);
      results.hidden = false;
    } catch (err) {
      showError(err.message || "Failed to process image");
    } finally {
      setLoading(false);
    }
  });

  function showImage(file) {
    const url = URL.createObjectURL(file);
    const img = document.createElement("img");
    img.alt = "Uploaded image";
    img.src = url;
    img.style.maxWidth = "100%";
    preview.innerHTML = "";
    preview.appendChild(img);
  }

  function renderPrediction({ label, score, scores, annotated_image_base64: encoded, note }) {
    labelEl.textContent = `${label} (${(score * 100).toFixed(1)}%)`;
    scoreList.innerHTML = "";
    Object.entries(scores || {}).forEach(([name, value]) => {
      const row = document.createElement("div");
      row.className = "score-row";
      row.innerHTML = `<span>${name}</span><span>${(value * 100).toFixed(1)}%</span>`;
      scoreList.appendChild(row);
    });

    if (encoded) {
      const img = document.createElement("img");
      img.alt = label;
      img.src = `data:image/png;base64,${encoded}`;
      preview.innerHTML = "";
      preview.appendChild(img);
    }

    if (note) {
      const noteEl = document.createElement("p");
      noteEl.className = "muted";
      noteEl.textContent = note;
      scoreList.appendChild(noteEl);
    }
  }

  function applyApiBase(value) {
    if (apiBaseEl) apiBaseEl.textContent = value;
    if (apiBaseInput) apiBaseInput.value = value;
  }

  function normalizeUrl(value) {
    if (!value) return null;
    return value.replace(/\/$/, "");
  }

  async function healthCheck(base) {
    if (!apiStatus) return;
    apiStatus.textContent = "…";
    apiStatus.className = "status status--checking";
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(`${base}/health`, { signal: controller.signal });
      clearTimeout(timer);
      if (!res.ok) throw new Error();
      apiStatus.textContent = "OK";
      apiStatus.className = "status status--ok";
    } catch (_) {
      apiStatus.textContent = "X";
      apiStatus.className = "status status--fail";
      showError("API not reachable. Check that the backend is running and the URL is correct.");
    }
  }

  function showError(message) {
    if (!errorMsg) {
      alert(message);
      return;
    }
    errorMsg.textContent = message;
    errorMsg.hidden = false;
  }

  function clearError() {
    if (!errorMsg) return;
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }
})();
