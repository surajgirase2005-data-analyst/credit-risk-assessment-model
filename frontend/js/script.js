(() => {
  "use strict";

  const API_URL = "/predict";
  const HEALTH_URL = "/health";
  const TIMEOUT_MS = 15000;

  // Field specs: type drives parsing, min/max drive validation.
  const FIELDS = {
    person_age:                 { type: "int",    min: 18, max: 100, label: "Age" },
    person_income:              { type: "float",  min: 0.01,         label: "Annual income" },
    person_home_ownership:      { type: "string",                    label: "Home ownership" },
    person_emp_length:          { type: "float",  min: 0,  max: 60,  label: "Employment length" },
    loan_intent:                { type: "string",                    label: "Loan intent" },
    loan_grade:                 { type: "string",                    label: "Loan grade" },
    loan_amnt:                  { type: "float",  min: 1,            label: "Loan amount" },
    loan_int_rate:              { type: "float",  min: 0,  max: 100, label: "Interest rate" },
    loan_percent_income:        { type: "float",  min: 0,  max: 1,   label: "Loan as % of income",
                                  requiredMsg: "Enter income and loan amount to calculate" },
    cb_person_default_on_file:  { type: "string",                    label: "Prior default on file" },
    cb_person_cred_hist_length: { type: "int",    min: 0,  max: 60,  label: "Credit history length" },
  };

  // DOM references
  const form = document.getElementById("loan-form");
  const submitBtn = document.getElementById("submit-btn");
  const resetBtn = document.getElementById("reset-btn");
  const sampleBtn = document.getElementById("sample-btn");
  const errorBox = document.getElementById("error-box");

  const resultCard = document.getElementById("result-card");
  const resultEmpty = document.getElementById("result-empty");
  const resultBody = document.getElementById("result-body");
  const resultBadge = document.getElementById("result-badge");
  const resultSummary = document.getElementById("result-summary");
  const probValue = document.getElementById("prob-value");
  const probFill = document.getElementById("prob-fill");
  const thresholdMarker = document.getElementById("threshold-marker");
  const thresholdValue = document.getElementById("threshold-value");

  const apiStatus = document.getElementById("api-status");
  const yearEl = document.getElementById("year");

  const incomeEl = form.elements["person_income"];
  const amountEl = form.elements["loan_amnt"];
  const percentEl = form.elements["loan_percent_income"];

  const SAMPLE = {
    person_age: 34,
    person_income: 65000,
    person_home_ownership: "RENT",
    person_emp_length: 5,
    loan_intent: "EDUCATION",
    loan_grade: "B",
    loan_amnt: 12000,
    loan_int_rate: 11.5,
    cb_person_default_on_file: "N",
    cb_person_cred_hist_length: 8,
  };

  /* ---------- Validation ---------- */

  function validateField(name, rawValue) {
    const spec = FIELDS[name];
    const value = String(rawValue ?? "").trim();

    if (value === "") {
      return spec.requiredMsg || `${spec.label} is required`;
    }
    if (spec.type === "string") return null;

    const num = Number(value);
    if (!Number.isFinite(num)) return "Enter a valid number";
    if (spec.type === "int" && !Number.isInteger(num)) return "Must be a whole number";
    if (spec.min !== undefined && num < spec.min) return `Must be at least ${spec.min}`;
    if (spec.max !== undefined && num > spec.max) return `Must be at most ${spec.max}`;
    return null;
  }

  function setFieldError(name, message) {
    const wrapper = form.querySelector(`[data-field="${name}"]`);
    const errorEl = wrapper.querySelector(".error");
    wrapper.classList.toggle("invalid", Boolean(message));
    errorEl.textContent = message || "";
    form.elements[name].setAttribute("aria-invalid", message ? "true" : "false");
  }

  function validateAll() {
    let valid = true;
    for (const name of Object.keys(FIELDS)) {
      const message = validateField(name, form.elements[name].value);
      setFieldError(name, message);
      if (message) valid = false;
    }
    return valid;
  }

  /* ---------- Derived field ---------- */

  function updatePercentIncome() {
    const income = parseFloat(incomeEl.value);
    const amount = parseFloat(amountEl.value);
    percentEl.value =
      income > 0 && amount > 0 ? (amount / income).toFixed(4) : "";
  }

  /* ---------- Payload & API ---------- */

  function buildPayload() {
    const payload = {};
    for (const [name, spec] of Object.entries(FIELDS)) {
      const raw = form.elements[name].value.trim();
      if (spec.type === "int") payload[name] = parseInt(raw, 10);
      else if (spec.type === "float") payload[name] = parseFloat(raw);
      else payload[name] = raw;
    }
    return payload;
  }

  async function requestPrediction(payload) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(await extractErrorMessage(res));
      }
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  async function extractErrorMessage(res) {
    try {
      const body = await res.json();
      if (Array.isArray(body.detail)) {
        return body.detail
          .map((d) => `${d.loc?.slice(-1)[0] ?? "field"}: ${d.msg}`)
          .join(" · ");
      }
      if (typeof body.detail === "string") return body.detail;
    } catch {
      /* fall through */
    }
    return `Server error (${res.status}). Please try again.`;
  }

  async function checkApiHealth() {
    try {
      const res = await fetch(HEALTH_URL, { cache: "no-store" });
      const ok = res.ok && (await res.json()).model_loaded;
      setStatus(ok ? "online" : "offline", ok ? "Model online" : "Model unavailable");
    } catch {
      setStatus("offline", "API unreachable");
    }
  }

  /* ---------- UI state ---------- */

  function setStatus(state, text) {
    apiStatus.classList.remove("online", "offline");
    apiStatus.classList.add(state);
    apiStatus.querySelector(".status-text").textContent = text;
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("loading", isLoading);
    submitBtn.querySelector(".btn-text").textContent = isLoading ? "Analyzing…" : "Assess risk";
    resultCard.setAttribute("aria-busy", String(isLoading));
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function animateNumber(el, to, decimals = 1, duration = 900) {
    const start = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = `${(to * eased).toFixed(decimals)}%`;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function renderResult({ default_probability, default_prediction, threshold, Result }) {
    const pct = default_probability * 100;
    const thresholdPct = Math.min(Math.max(threshold * 100, 0), 100);
    const isHigh = default_prediction === 1;

    resultEmpty.hidden = true;
    resultBody.hidden = false;

    resultBadge.textContent = Result;
    resultBadge.className = `badge ${isHigh ? "high" : "low"}`;

    resultSummary.textContent = isHigh
      ? "The estimated default probability is at or above the model's risk threshold, indicating elevated risk."
      : "The estimated default probability is below the model's risk threshold, indicating lower risk.";

    thresholdMarker.style.left = `${thresholdPct}%`;
    thresholdValue.textContent = `${thresholdPct.toFixed(1)}%`;

    // Animate the bar from zero on the next frame
    probFill.style.width = "0%";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        probFill.style.width = `${Math.min(pct, 100)}%`;
      });
    });
    animateNumber(probValue, pct);

    resultCard.classList.remove("pop");
    void resultCard.offsetWidth; // restart CSS animation
    resultCard.classList.add("pop");
  }

  function resetResult() {
    resultBody.hidden = true;
    resultEmpty.hidden = false;
    probFill.style.width = "0%";
    probValue.textContent = "0.0%";
    resultCard.classList.remove("pop");
  }

  /* ---------- Event handlers ---------- */

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideError();

    if (!validateAll()) {
      form.querySelector(".invalid [name]")?.focus();
      return;
    }

    setLoading(true);
    try {
      const data = await requestPrediction(buildPayload());
      renderResult(data);
    } catch (err) {
      const message =
        err.name === "AbortError"
          ? "The request timed out. The server may be waking up, so please try again."
          : err.message || "Something went wrong. Please try again.";
      showError(message);
    } finally {
      setLoading(false);
    }
  });

  // Live validation: validate on blur, clear error while typing
  for (const name of Object.keys(FIELDS)) {
    const el = form.elements[name];
    if (el.readOnly) continue;

    el.addEventListener("blur", () => {
      setFieldError(name, validateField(name, el.value));
    });
    el.addEventListener("input", () => {
      if (form.querySelector(`[data-field="${name}"]`).classList.contains("invalid")) {
        setFieldError(name, validateField(name, el.value));
      }
    });
  }

  incomeEl.addEventListener("input", updatePercentIncome);
  amountEl.addEventListener("input", updatePercentIncome);

  resetBtn.addEventListener("click", () => {
    form.reset();
    for (const name of Object.keys(FIELDS)) setFieldError(name, "");
    updatePercentIncome();
    hideError();
    resetResult();
  });

  sampleBtn.addEventListener("click", () => {
    for (const [name, value] of Object.entries(SAMPLE)) {
      form.elements[name].value = value;
      setFieldError(name, "");
    }
    updatePercentIncome();
    hideError();
  });

  /* ---------- Init ---------- */

  yearEl.textContent = new Date().getFullYear();
  checkApiHealth();
  updatePercentIncome();
})();
