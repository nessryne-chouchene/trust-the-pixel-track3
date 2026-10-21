/* TSYP14 Security Assessment - project guide page.
 * Small progressive enhancements only; the page is fully readable without JS. */
(function () {
  "use strict";

  /* ---------- Mobile navigation ---------- */
  var toggle = document.querySelector(".nav-toggle");
  var links = document.getElementById("nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    links.addEventListener("click", function (e) {
      if (e.target.closest("a")) {
        links.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  /* ---------- Highlight the section in view ---------- */
  var navAnchors = Array.prototype.slice.call(document.querySelectorAll(".nav-links a[href^='#']"));
  if ("IntersectionObserver" in window && navAnchors.length) {
    var byId = {};
    navAnchors.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        navAnchors.forEach(function (a) { a.classList.remove("active"); a.removeAttribute("aria-current"); });
        var a = byId[entry.target.id];
        if (a) { a.classList.add("active"); a.setAttribute("aria-current", "true"); }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(byId).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  }

  /* ---------- Operating-system tabs ---------- */
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab[role='tab']"));
  function selectTab(tab, focus) {
    tabs.forEach(function (t) {
      var selected = t === tab;
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !selected;
    });
    if (focus) tab.focus();
  }
  tabs.forEach(function (tab, i) {
    tab.addEventListener("click", function () { selectTab(tab, false); });
    tab.addEventListener("keydown", function (e) {
      var next = null;
      if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
      else if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
      else if (e.key === "Home") next = tabs[0];
      else if (e.key === "End") next = tabs[tabs.length - 1];
      if (next) { e.preventDefault(); selectTab(next, true); }
    });
  });
  // Preselect the Linux/macOS tab for visitors on those systems.
  var platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "";
  if (tabs.length > 1 && /mac|linux|x11/i.test(platform) && !/win/i.test(platform)) selectTab(tabs[1], false);

  /* ---------- Copy buttons for commands ---------- */
  function fallbackCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    document.body.removeChild(area);
    return ok;
  }
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).then(function () { return true; }, function () { return fallbackCopy(text); });
    }
    return Promise.resolve(fallbackCopy(text));
  }
  document.querySelectorAll(".cmd pre:not(.output) > code").forEach(function (code) {
    var box = code.closest(".cmd");
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy command: " + code.textContent.trim().slice(0, 60));
    btn.addEventListener("click", function () {
      copyText(code.textContent.trim()).then(function (ok) {
        btn.textContent = ok ? "Copied ✓" : "Select & copy manually";
        btn.classList.toggle("copied", ok);
        btn.classList.toggle("failed", !ok);
        setTimeout(function () {
          btn.textContent = "Copy";
          btn.classList.remove("copied", "failed");
        }, 1800);
      });
    });
    box.appendChild(btn);
  });

  /* ---------- Label table cells so they can stack on phones ---------- */
  function labelCells(table) {
    var heads = Array.prototype.map.call(table.querySelectorAll("thead th"), function (th) { return th.textContent.trim(); });
    table.querySelectorAll("tbody tr").forEach(function (tr) {
      Array.prototype.forEach.call(tr.children, function (td, i) {
        if (!heads[i]) return;
        td.setAttribute("data-label", heads[i]);
        // One wrapper per cell so its content sits beside the label as a single block.
        var cell = document.createElement("span");
        cell.className = "cell";
        while (td.firstChild) cell.appendChild(td.firstChild);
        td.appendChild(cell);
      });
    });
  }
  document.querySelectorAll(".stack-mobile table").forEach(labelCells);

  /* ---------- Load the latest test report ---------- */
  var loadBtn = document.getElementById("load-results");
  var statusEl = document.getElementById("live-status");
  var outEl = document.getElementById("live-results");

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }
  function badge(status) {
    if (!status) return el("span", { "class": "badge neutral", text: "n/a" });
    return el("span", { "class": "badge " + String(status).toLowerCase(), text: status });
  }
  function codesCell(codes) {
    var td = el("td", { "class": "codes" });
    if (!codes || !codes.length) { td.textContent = "–"; return td; }
    codes.forEach(function (c, i) {
      if (i) td.appendChild(document.createTextNode(" "));
      td.appendChild(el("code", { text: c }));
    });
    return td;
  }

  function render(report) {
    var s = report.summary || {};
    var results = Array.isArray(report.results) ? report.results : [];
    outEl.innerHTML = "";

    var stats = el("div", { "class": "stat-row" }, [
      el("div", { "class": "stat " + (s.failed ? "bad" : "ok") }, [el("b", { text: s.passed + " / " + s.total }), el("span", { text: "tests passed (" + s.success_rate_percent + "%)" })]),
      el("div", { "class": "stat " + (s.failed ? "bad" : "ok") }, [el("b", { text: String(s.failed) }), el("span", { text: "tests failed" })]),
      el("div", { "class": "stat" }, [el("b", { text: s.python || "?" }), el("span", { text: "Python version" })]),
      el("div", { "class": "stat" }, [el("b", { text: (s.run_at_utc || "?").replace("T", " ").replace("Z", "") }), el("span", { text: "run time (UTC)" })])
    ]);
    outEl.appendChild(stats);

    var tbody = el("tbody");
    results.forEach(function (r) {
      var exp = r.expected || {};
      var act = r.actual || {};
      var outcome = el("td", { "class": r.passed ? "outcome-pass" : "outcome-fail", text: r.passed ? "✓ Pass" : "✗ Fail" });
      var descCell = el("td", { text: r.description || "" });
      if (r.mismatches && r.mismatches.length) {
        var ul = el("ul", { "class": "t-bad", style: "margin:6px 0 0" });
        r.mismatches.forEach(function (m) { ul.appendChild(el("li", { text: m })); });
        descCell.appendChild(ul);
      }
      tbody.appendChild(el("tr", {}, [
        el("td", {}, [el("code", { text: r.case_id })]),
        el("td", { text: r.group }),
        descCell,
        el("td", {}, [r.group === "invariants" ? el("span", { "class": "t-muted", text: "–" }) : badge(exp.security_status)]),
        el("td", {}, [r.group === "invariants" ? el("span", { "class": "t-muted", text: "–" }) : badge(act.security_status)]),
        r.group === "invariants" ? el("td", { text: "–" }) : codesCell(act.reason_codes),
        outcome
      ]));
    });
    var table = el("table", {}, [
      el("caption", { text: "All tests from test_results/results.json (" + results.length + " rows)" }),
      el("thead", {}, [el("tr", {}, ["Test case", "Group", "Description", "Expected", "Actual", "Actual reason codes", "Test outcome"].map(function (h) { return el("th", { text: h }); }))]),
      tbody
    ]);
    labelCells(table);
    outEl.appendChild(el("div", { "class": "table-wrap stack-mobile" }, [table]));
  }

  if (loadBtn) {
    loadBtn.addEventListener("click", function () {
      loadBtn.disabled = true;
      statusEl.className = "live-status";
      statusEl.textContent = "Loading test_results/results.json …";
      fetch("../test_results/results.json", { cache: "no-store" })
        .then(function (res) {
          if (!res.ok) throw new Error("HTTP " + res.status);
          return res.json();
        })
        .then(function (report) {
          if (!report || !report.summary || !Array.isArray(report.results)) throw new Error("unexpected file format");
          render(report);
          statusEl.className = "live-status " + (report.summary.failed ? "t-bad" : "t-ok");
          statusEl.textContent = "Loaded the report of the run on " + report.summary.run_at_utc + ".";
        })
        .catch(function (err) {
          outEl.innerHTML = "";
          statusEl.className = "live-status t-warn";
          if (location.protocol === "file:") {
            statusEl.textContent = "Your browser does not allow this page to read local files when it is opened by double-click. " +
              "Use the report links above, or serve the folder with the command under “View live results in this page” in How to run.";
          } else {
            statusEl.textContent = "Could not read test_results/results.json (" + err.message + "). " +
              "Generate the data and run the tests first, then try again.";
          }
        })
        .then(function () { loadBtn.disabled = false; });
    });
  }
})();
