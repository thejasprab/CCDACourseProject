// app/static/app.js
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
    // an accidental POST body; so we re-submit as GET.
    const form = selectEl.form;
    if (!form) return;

    // If the form method is POST, temporarily switch to GET for mode change.
    const originalMethod = form.method;
    form.method = "GET";
    form.submit();
    form.method = originalMethod;
  },
};

window.ArxivUI = ArxivUI;
