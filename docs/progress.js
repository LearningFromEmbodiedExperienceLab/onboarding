(function () {
  "use strict";

  var STORAGE_KEY = "onboarding-book-progress:v1";

  function readStore() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return { sections: {}, pages: {} };
      }
      var data = JSON.parse(raw);
      if (!data.sections) {
        data.sections = {};
      }
      if (!data.pages) {
        data.pages = {};
      }
      return data;
    } catch (_err) {
      return { sections: {}, pages: {} };
    }
  }

  function writeStore(data) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (_err) {
      /* private mode or quota — ignore */
    }
  }

  function pageBasename() {
    var file = window.location.pathname.split("/").pop() || "index.html";
    return file.replace(/\.html$/, "") || "index";
  }

  function sectionKey(page, headingId) {
    return page + "#" + headingId;
  }

  function isDone(store, key) {
    return store.sections[key] === true;
  }

  function setDone(store, key, done) {
    if (done) {
      store.sections[key] = true;
    } else {
      delete store.sections[key];
    }
  }

  function discoverSections() {
    var page = pageBasename();
    var nodes = document.querySelectorAll("main.content section.level2[id]");
    var sections = [];
    nodes.forEach(function (section) {
      var id = section.id;
      if (!id) {
        return;
      }
      sections.push({ page: page, id: id, section: section });
    });
    return sections;
  }

  function registerPageSections(store, page, sectionIds) {
    var existing = store.pages[page] || [];
    var merged = existing.slice();
    sectionIds.forEach(function (id) {
      if (merged.indexOf(id) === -1) {
        merged.push(id);
      }
    });
    store.pages[page] = merged;
  }

  function chapterProgress(store, page) {
    var ids = store.pages[page] || [];
    if (ids.length === 0) {
      return { total: 0, done: 0, state: "unknown" };
    }
    var done = 0;
    ids.forEach(function (id) {
      if (isDone(store, sectionKey(page, id))) {
        done += 1;
      }
    });
    var state = "unread";
    if (done === ids.length) {
      state = "done";
    } else if (done > 0) {
      state = "partial";
    }
    return { total: ids.length, done: done, state: state };
  }

  function tocHeadingId(link) {
    var target =
      link.getAttribute("data-scroll-target") || link.getAttribute("href") || "";
    var hashIndex = target.indexOf("#");
    if (hashIndex === -1) {
      return "";
    }
    var id = target.slice(hashIndex + 1);
    try {
      return decodeURIComponent(id);
    } catch (_err) {
      return id;
    }
  }

  function applyTocStyles(store, page) {
    document.querySelectorAll("#TOC a.nav-link").forEach(function (link) {
      var id = tocHeadingId(link);
      if (!id) {
        return;
      }
      link.classList.remove("progress-done", "progress-unread");
      if (isDone(store, sectionKey(page, id))) {
        link.classList.add("progress-done");
      } else {
        link.classList.add("progress-unread");
      }
    });
  }

  function applySidebarStyles(store) {
    document
      .querySelectorAll("#quarto-sidebar a.sidebar-link[href$='.html']")
      .forEach(function (link) {
        var href = link.getAttribute("href") || "";
        var match = href.match(/([^/]+)\.html$/);
        if (!match) {
          return;
        }
        var page = match[1];
        var progress = chapterProgress(store, page);
        link.classList.remove(
          "progress-chapter-unread",
          "progress-chapter-partial",
          "progress-chapter-done"
        );
        if (progress.state === "done") {
          link.classList.add("progress-chapter-done");
        } else if (progress.state === "partial") {
          link.classList.add("progress-chapter-partial");
        } else {
          link.classList.add("progress-chapter-unread");
        }
      });
  }

  function updateSummary(store, page, doneCount, total) {
    var toc = document.getElementById("TOC");
    if (!toc || total === 0) {
      return;
    }
    var summary = document.getElementById("progress-summary");
    if (!summary) {
      summary = document.createElement("p");
      summary.id = "progress-summary";
      summary.className = "progress-summary";
      var title = toc.querySelector("#toc-title");
      if (title) {
        title.insertAdjacentElement("afterend", summary);
      } else {
        toc.prepend(summary);
      }

      var count = document.createElement("span");
      count.className = "progress-summary-count";
      summary.appendChild(count);

      var reset = document.createElement("button");
      reset.type = "button";
      reset.className = "progress-reset";
      reset.textContent = "Clear this page";
      reset.title = "Clear progress for this page";
      reset.addEventListener("click", function () {
        var data = readStore();
        (data.pages[page] || []).forEach(function (id) {
          delete data.sections[sectionKey(page, id)];
        });
        writeStore(data);
        refresh(data);
      });
      summary.appendChild(reset);
    }
    summary.querySelector(".progress-summary-count").textContent =
      doneCount + " of " + total + " sections marked done";
  }

  function attachHeadingControls(store, sections) {
    sections.forEach(function (item) {
      var h2 = item.section.querySelector("h2");
      if (!h2 || item.section.querySelector(".progress-mark")) {
        return;
      }
      var key = sectionKey(item.page, item.id);
      var label = document.createElement("label");
      label.className = "progress-mark";
      label.title = "Mark section done (saved in this browser only)";

      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = isDone(store, key);
      box.addEventListener("change", function () {
        var data = readStore();
        setDone(data, key, box.checked);
        registerPageSections(
          data,
          item.page,
          sections.map(function (s) {
            return s.id;
          })
        );
        writeStore(data);
        refresh(data);
      });

      label.appendChild(box);
      label.appendChild(document.createTextNode("Done"));
      h2.appendChild(label);
    });
  }

  function refresh(currentStore) {
    var store = currentStore || readStore();
    var sections = discoverSections();
    var page = pageBasename();
    var ids = sections.map(function (s) {
      return s.id;
    });
    registerPageSections(store, page, ids);
    writeStore(store);

    applyTocStyles(store, page);
    applySidebarStyles(store);

    var doneOnPage = 0;
    ids.forEach(function (id) {
      if (isDone(store, sectionKey(page, id))) {
        doneOnPage += 1;
      }
    });
    updateSummary(store, page, doneOnPage, ids.length);

    document.querySelectorAll(".progress-mark input").forEach(function (box) {
      var label = box.closest(".progress-mark");
      var section = label && label.closest("section.level2");
      if (!section) {
        return;
      }
      box.checked = isDone(store, sectionKey(page, section.id));
    });
  }

  function init() {
    if (!document.querySelector("main.content")) {
      return;
    }
    var store = readStore();
    var sections = discoverSections();
    registerPageSections(
      store,
      pageBasename(),
      sections.map(function (s) {
        return s.id;
      })
    );
    writeStore(store);
    attachHeadingControls(store, sections);
    refresh();

    window.addEventListener("storage", function (event) {
      if (event.key === STORAGE_KEY) {
        refresh();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
