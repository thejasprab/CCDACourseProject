/* global window, document */

const ArxivUI = {
  clearForm() {
    const form = document.querySelector("form[action$='/']");
    if (!form) return;
    const title = form.querySelector("#title");
    const abstract = form.querySelector("#abstract");
    if (title) title.value = "";
    if (abstract) abstract.value = "";
  },

  submitOnModeChange(selectEl) {
    // On the search page we want the dataset change to persist, but not trigger
    // an accidental POST body, so we re-submit as GET.
    const form = selectEl.form;
    if (!form) return;

    const originalMethod = form.method;
    form.method = "GET";
    form.submit();
    form.method = originalMethod;
  },

  applyTheme(theme) {
    const body = document.body;
    const next = theme === "dark" ? "dark" : "light";

    body.dataset.theme = next;
    try {
      window.localStorage.setItem("sparxiv-theme", next);
    } catch (e) {
      // storage not critical, ignore
    }

    const toggle = document.querySelector("[data-theme-toggle]");
    if (toggle) {
      toggle.textContent = next === "dark" ? "Light theme" : "Dark theme";
    }
  },

  toggleTheme() {
    const current = document.body.dataset.theme === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    ArxivUI.applyTheme(next);
  },

  initTheme() {
    let theme = "light";

    try {
      const stored = window.localStorage.getItem("sparxiv-theme");
      if (stored === "light" || stored === "dark") {
        theme = stored;
      } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        theme = "dark";
      }
    } catch (e) {
      // fall back to light if storage fails
    }

    ArxivUI.applyTheme(theme);
  },
};

window.ArxivUI = ArxivUI;

document.addEventListener("DOMContentLoaded", () => {
  ArxivUI.initTheme();
});
