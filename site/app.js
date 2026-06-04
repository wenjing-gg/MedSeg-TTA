(async function bootstrap() {
  const state = {
    data: null,
    view: "modalities",
    modality: "OCT",
    sortBy: "dice",
    region: "WT"
  };

  const paradigmOrder = [
    "input_level_transformation",
    "feature_level_alignment",
    "output_level_regularization",
    "prior_estimation"
  ];
  const modalityOrder = ["OCT", "PATH", "DER", "CXR", "MRI", "CT", "US"];
  const viewIds = new Set(["overview", "modalities"]);
  const modalityMetricLabels = {
    dice: "Dice ↑",
    hd95: "HD95 ↓",
    ji: "JI ↑",
    sen: "Sen ↑",
    ppv: "PPV ↑"
  };
  const localMethodRoutes = {
    "AIF-SFDA": {
      two_d: "input_level_transformation/AIF-SFDA/two_d"
    },
    STDR: {
      three_d: "input_level_transformation/STDR/three_d"
    },
    RSA: {
      three_d: "input_level_transformation/RSA/three_d"
    },
    "SFDA-FSM": {
      two_d: "input_level_transformation/SFDA-FSM/two_d",
      three_d: "input_level_transformation/SFDA-FSM/three_d"
    },
    "DL-TTA": {
      two_d: "input_level_transformation/DL-TTA/two_d",
      three_d: "input_level_transformation/DL-TTA/three_d"
    },
    "UPL-SFDA": {
      two_d: "output_level_regularization/UPL-SFDA/two_d",
      three_d: "output_level_regularization/UPL-SFDA/three_d"
    },
    GraTa: {
      two_d: "feature_level_alignment/GraTa/two_d",
      three_d: "feature_level_alignment/GraTa/three_d"
    },
    "UDA-MIMA": {
      two_d: "feature_level_alignment/UDA-MIMA/two_d",
      three_d: "feature_level_alignment/UDA-MIMA/three_d"
    },
    DANN: {
      two_d: "feature_level_alignment/DANN/two_d",
      three_d: "feature_level_alignment/DANN/three_d"
    },
    TestFit: {
      two_d: "feature_level_alignment/Testfit/two_d",
      three_d: "feature_level_alignment/Testfit/three_d"
    },
    "DG-TTA": {
      two_d: "output_level_regularization/DG-TTA/two_d",
      three_d: "output_level_regularization/DG-TTA/three_d"
    },
    SaTTCA: {
      two_d: "output_level_regularization/SaTTCA/two_d",
      three_d: "output_level_regularization/SaTTCA/three_d"
    },
    TENT: {
      two_d: "output_level_regularization/tent/two_d",
      three_d: "output_level_regularization/tent/three_d"
    },
    ProSFDA: {
      two_d: "prior_estimation/ProSFDA/two_d",
      three_d: "prior_estimation/ProSFDA/three_d"
    },
    ExploringTTA: {
      three_d: "prior_estimation/ExploringTTA/three_d"
    },
    PASS: {
      two_d: "prior_estimation/PASS/two_d",
      three_d: "prior_estimation/PASS/three_d"
    },
    AdaMI: {
      two_d: "prior_estimation/AdaMI/two_d",
      three_d: "prior_estimation/AdaMI/three_d"
    }
  };
  const modalityProfiles = {
    OCT: {
      source: "RIGA+ (MES)",
      target: "RIGA+ (MB)",
      task: "Fundus optic disc and cup",
      dimension: "2D"
    },
    PATH: {
      source: "CRAG",
      target: "Glas",
      task: "Histopathology gland",
      dimension: "2D"
    },
    DER: {
      source: "ISIC-2017",
      target: "PH²",
      task: "Skin lesion",
      dimension: "2D"
    },
    CXR: {
      source: "SZ-CXR",
      target: "Montgomery",
      task: "Lung field",
      dimension: "2D"
    },
    MRI: {
      source: "BraTS-GLI2024",
      target: "BraTS-SSA",
      task: "Brain tumor regions",
      dimension: "3D"
    },
    CT: {
      source: "LiTS",
      target: "3D-IRCADB",
      task: "Liver",
      dimension: "3D"
    },
    US: {
      source: "TN3K",
      target: "DDTI",
      task: "Thyroid nodule",
      dimension: "2D"
    }
  };

  try {
    const response = await fetch("./data/leaderboard.json");
    if (!response.ok) {
      throw new Error("Unable to load leaderboard.json");
    }

    state.data = await response.json();
    decorateMethods(state.data);
    initializeStaticContent(state);
    bindViewTabs(state);
    bindJumpLinks(state);
    syncViewWithHash(state);
    renderAll(state);
  } catch (error) {
    renderError(error);
  }

  function renderError(error) {
    const main = document.querySelector(".page");
    main.innerHTML = `
      <section class="band">
        <h1 class="hero__title">The MVP page could not read its benchmark data.</h1>
        <p class="hero__lede">${escapeHtml(String(error.message || error))}</p>
      </section>
    `;
  }

  function decorateMethods(data) {
    Object.entries(localMethodRoutes).forEach(([methodName, routes]) => {
      if (!data.methods[methodName]) {
        return;
      }
      data.methods[methodName].routes = routes;
      data.methods[methodName].dimensions = Object.keys(routes);
      data.methods[methodName].local = true;
      data.methods[methodName].codePath = getRootPath(routes);
    });
  }

  function initializeStaticContent(stateRef) {
    const { data } = stateRef;
    document.title = data.meta.title;
    document.getElementById("hero-lede").textContent = data.meta.subtitle;
    document.getElementById("footer-updated").textContent = `Updated ${data.meta.updated} from benchmark tables.`;

    const stats = [
      { value: Object.keys(data.modalityLeaderboards).length, label: "Modalities" },
      { value: paradigmOrder.length, label: "Paradigms" },
      { value: Object.keys(data.methods).length, label: "Methods" }
    ];

    document.getElementById("hero-stats").innerHTML = stats
      .map(
        (stat) => `
          <div class="stat-chip">
            <span class="stat-chip__value mono">${stat.value}</span>
            <span class="stat-chip__label">${stat.label}</span>
          </div>
        `
      )
      .join("");

  }

  function bindViewTabs(stateRef) {
    document.querySelectorAll(".view-tabs__button").forEach((button) => {
      button.addEventListener("click", () => {
        stateRef.view = button.dataset.view;
        updateViewTabs(stateRef);
        updateHash(stateRef.view);
      });
    });
    updateViewTabs(stateRef);
  }

  function bindJumpLinks(stateRef) {
    document.querySelectorAll("[data-view-link]").forEach((link) => {
      link.addEventListener("click", () => {
        const targetView = link.dataset.viewLink;
        if (!viewIds.has(targetView)) {
          return;
        }
        stateRef.view = targetView;
        updateViewTabs(stateRef);
      });
    });

    window.addEventListener("hashchange", () => {
      syncViewWithHash(stateRef);
    });
  }

  function syncViewWithHash(stateRef) {
    const hash = window.location.hash.replace(/^#/, "");
    if (!viewIds.has(hash)) {
      return;
    }
    stateRef.view = hash;
    updateViewTabs(stateRef);
  }

  function updateHash(view) {
    if (window.location.hash.replace(/^#/, "") === view) {
      return;
    }
    window.history.replaceState(null, "", `#${view}`);
  }

  function updateViewTabs(stateRef) {
    document.querySelectorAll(".view-tabs__button").forEach((button) => {
      const active = button.dataset.view === stateRef.view;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });

    document.querySelectorAll(".view-section").forEach((section) => {
      section.classList.toggle("is-hidden", section.id !== stateRef.view);
    });
  }

  function renderAll(stateRef) {
    renderOverview(stateRef);
    renderModalityExplorer(stateRef);
  }

  function renderOverview(stateRef) {
    const container = document.getElementById("overview");
    const grouped = groupBy(stateRef.data.paradigmOverview, "domainShift");

    container.innerHTML = `
      <div class="section-head">
        <div>
          <h2 class="section-title">Paradigm-level overview across domain shift regimes</h2>
          <p class="section-text">
            Cross-domain cells keep the paper's intra-domain to target-domain transition, while the four paradigm blocks expose Dice and HD95 behavior side by side. The top three paradigm scores for each row are highlighted directly inside the table.
          </p>
        </div>
      </div>
      <div class="overview-grid">
        ${buildOverviewTable("Strong", grouped.Strong || [], stateRef.data)}
        ${buildOverviewTable("Mild", grouped.Mild || [], stateRef.data)}
      </div>
    `;
  }

  function buildOverviewTable(groupTitle, rows, data) {
    return `
      <div class="table-block">
        <h3 class="table-block__title">${groupTitle} domain shift</h3>
        <p class="table-block__subtitle">
          Higher Dice and lower HD95 indicate better target-domain segmentation. The cross-domain columns show how far source-domain performance drops before TTA is applied.
        </p>
        <div class="table-shell">
          <table class="paper-table overview-table">
            <thead>
              <tr>
                <th rowspan="2">Modality</th>
                <th colspan="2">Cross-domain</th>
                ${paradigmOrder
                  .map((key) => `<th colspan="2">${data.paradigms[key].symbol} ${data.paradigms[key].shortLabel}</th>`)
                  .join("")}
              </tr>
              <tr>
                <th>Dice ↑</th>
                <th>HD95 ↓</th>
                ${paradigmOrder.map(() => "<th>Dice ↑</th><th>HD95 ↓</th>").join("")}
              </tr>
            </thead>
            <tbody>
              ${rows
                .map((row) => {
                  const diceRanks = rankMap(
                    paradigmOrder.map((key) => ({ id: key, value: row.scores[key].dice })),
                    "value",
                    true,
                    "id"
                  );
                  const hd95Ranks = rankMap(
                    paradigmOrder.map((key) => ({ id: key, value: row.scores[key].hd95 })),
                    "value",
                    false,
                    "id"
                  );

                  return `
                    <tr>
                      <th>${row.modality}</th>
                      <td class="metric-cell">${stackTransition(row.crossDomain.dice.intra, row.crossDomain.dice.target, "dice")}</td>
                      <td class="metric-cell">${stackTransition(row.crossDomain.hd95.intra, row.crossDomain.hd95.target, "hd95")}</td>
                      ${paradigmOrder
                        .map((key) => {
                          const score = row.scores[key];
                          return `
                            <td class="metric-cell mono ${rankClass(diceRanks[key])}">${formatMetric(score.dice, "dice")}</td>
                            <td class="metric-cell mono ${rankClass(hd95Ranks[key])}">${formatMetric(score.hd95, "hd95")}</td>
                          `;
                        })
                        .join("")}
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  function getMetricValue(metrics, metric, modality, region) {
    const value = metrics[metric];
    if (value === undefined) {
      return undefined;
    }
    return modality.regional ? value[region] : value;
  }

  function getMetricStd(metrics, metric, modality, region) {
    const std = metrics.std && metrics.std[metric];
    if (std === undefined) {
      return undefined;
    }
    return modality.regional ? std[region] : std;
  }

  function metricExistsForModality(modality, metric, region) {
    if (modality.regional && !region) {
      return modality.regions.some((regionKey) => metricExistsForModality(modality, metric, regionKey));
    }
    if (getMetricValue(modality.baseline.intra, metric, modality, region) !== undefined) {
      return true;
    }
    return Object.values(modality.methods).some((metrics) => getMetricValue(metrics, metric, modality, region) !== undefined);
  }

  function metricCell(metrics, metric, modality, region, className = "") {
    const value = getMetricValue(metrics, metric, modality, region);
    if (value === undefined) {
      return `<td class="metric-cell mono ${className}">—</td>`;
    }
    const std = getMetricStd(metrics, metric, modality, region);
    const formatted = formatMetric(value, metric);
    const text = std === undefined ? formatted : `${formatted}<span class="metric-std">±${formatMetric(std, metric)}</span>`;
    return `<td class="metric-cell mono ${className}">${text}</td>`;
  }

  function getRegionalAverage(metrics, metric, modality) {
    if (!modality.regional) {
      return getMetricValue(metrics, metric, modality, "WT");
    }
    const values = modality.regions
      .map((region) => getMetricValue(metrics, metric, modality, region))
      .filter((value) => value !== undefined);
    if (!values.length) {
      return undefined;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function getSortValue(metrics, metric, modality) {
    return modality.regional ? getRegionalAverage(metrics, metric, modality) : getMetricValue(metrics, metric, modality, "WT");
  }

  function renderModalityExplorer(stateRef) {
    const { data } = stateRef;
    const modalityKeys = modalityOrder.filter((key) => data.modalityLeaderboards[key]);
    const datasetSelector = document.getElementById("dataset-selector");
    const sortTabs = document.getElementById("sort-tabs");

    datasetSelector.innerHTML = modalityKeys
      .map((key) => {
        return `
          <button type="button" class="dataset-card ${stateRef.modality === key ? "is-active" : ""}" data-modality="${key}" aria-pressed="${stateRef.modality === key ? "true" : "false"}">
            <span class="dataset-card__modality">${key}</span>
          </button>
        `;
      })
      .join("");

    datasetSelector.querySelectorAll("[data-modality]").forEach((button) => {
      button.addEventListener("click", () => {
        stateRef.modality = button.dataset.modality;
        renderModalityExplorer(stateRef);
      });
    });

    sortTabs.innerHTML = `
      <button type="button" class="segmented__button ${stateRef.sortBy === "dice" ? "is-active" : ""}" data-sort="dice">Sort by Dice</button>
      <button type="button" class="segmented__button ${stateRef.sortBy === "hd95" ? "is-active" : ""}" data-sort="hd95">Sort by HD95</button>
    `;
    sortTabs.querySelectorAll("[data-sort]").forEach((button) => {
      button.addEventListener("click", () => {
        stateRef.sortBy = button.dataset.sort;
        renderModalityExplorer(stateRef);
      });
    });

    const container = document.getElementById("modalities");
    if (!container.querySelector(".section-head")) {
      container.insertAdjacentHTML(
        "afterbegin",
        `
          <div class="section-head">
            <div>
              <h2 class="section-title">Dataset drilldown</h2>
            </div>
          </div>
        `
      );
    }

    const selectedKey = data.modalityLeaderboards[stateRef.modality] ? stateRef.modality : modalityKeys[0];
    stateRef.modality = selectedKey;
    document.getElementById("dataset-detail").innerHTML = renderDatasetSection(stateRef, selectedKey, data.modalityLeaderboards[selectedKey]);
  }

  function renderDatasetSection(stateRef, modalityKey, modality) {
    if (modality.regional) {
      return renderRegionalDatasetSection(stateRef, modalityKey, modality);
    }
    const data = stateRef.data;
    const profile = modalityProfiles[modalityKey] || {};
    const region = "WT";
    const rows = Object.entries(modality.methods).map(([method, metrics]) => ({ method, metrics }));
    rows.sort((left, right) => {
      const leftValue = getSortValue(left.metrics, stateRef.sortBy, modality);
      const rightValue = getSortValue(right.metrics, stateRef.sortBy, modality);
      if (leftValue === undefined || rightValue === undefined) {
        return left.method.localeCompare(right.method);
      }
      return stateRef.sortBy === "dice" ? rightValue - leftValue : leftValue - rightValue;
    });

    const metricKeys = ["dice", "hd95", "ji", "sen", "ppv"].filter((metric) => metricExistsForModality(modality, metric, region));
    const rankMaps = Object.fromEntries(
      metricKeys.map((metric) => [
        metric,
        rankMap(
          rows
            .map((row) => ({ method: row.method, value: getMetricValue(row.metrics, metric, modality, region) }))
            .filter((row) => row.value !== undefined),
          "value",
          metric !== "hd95"
        )
      ])
    );
    const targetDice = getMetricValue(modality.baseline.target, "dice", modality, region);
    const contextChips = [
      { label: "Task", value: profile.task || "—" },
      { label: "Dataset", value: profile.source && profile.target ? `${profile.source} → ${profile.target}` : "—", wide: true },
      { label: "Runtime", value: profile.dimension || "Mixed" },
      { label: "Shift", value: modality.domainShift }
    ];

    return `
      <section class="dataset-section" id="dataset-${modalityKey}">
        <div class="dataset-section__head">
          <div>
            <h3 class="table-block__title">${modalityKey} dataset performance</h3>
            <p class="section-text">${escapeHtml(profile.source || "Source")} → ${escapeHtml(profile.target || "Target")} · ${rows.length} methods · ${metricKeys.map((metric) => modalityMetricLabels[metric].replace(/[↑↓]/g, "").trim()).join(", ")}</p>
          </div>
        </div>
        <div class="context-strip dataset-context">
          ${contextChips
            .map(
              (chip) => `
                <div class="context-chip ${chip.wide ? "context-chip--wide" : ""}">
                  <span class="context-chip__label">${escapeHtml(chip.label)}</span>
                  <span class="context-chip__value">${escapeHtml(chip.value)}</span>
                </div>
              `
            )
            .join("")}
        </div>
        <div class="table-shell">
          <table class="paper-table paper-table--metrics">
            <thead>
              <tr>
                <th>Method</th>
                <th>Paradigm</th>
                ${metricKeys.map((metric) => `<th>${modalityMetricLabels[metric]}</th>`).join("")}
              </tr>
            </thead>
            <tbody>
              <tr class="group-divider">
                <th>Intra-domain</th>
                <td>Source baseline</td>
                ${metricKeys.map((metric) => metricCell(modality.baseline.intra, metric, modality, region)).join("")}
              </tr>
              <tr>
                <th>Target-domain (w/o TTA)</th>
                <td>Target baseline</td>
                ${metricKeys.map((metric) => metricCell(modality.baseline.target, metric, modality, region)).join("")}
              </tr>
              ${rows
                .map((row) => {
                  const metadata = data.methods[row.method];
                  const dice = getMetricValue(row.metrics, "dice", modality, region);
                  const belowTarget = targetDice !== undefined && dice !== undefined && dice < targetDice;
                  const paradigm = data.paradigms[metadata.paradigm];
                  return `
                    <tr class="${belowTarget ? "below-target" : ""}">
                      <td>${methodLabel(row.method, metadata)}</td>
                      <td>${paradigm.symbol} ${paradigm.label}</td>
                      ${metricKeys.map((metric) => metricCell(row.metrics, metric, modality, region, rankClass(rankMaps[metric][row.method]))).join("")}
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      </section>
    `;
  }

  function renderRegionalDatasetSection(stateRef, modalityKey, modality) {
    const data = stateRef.data;
    const profile = modalityProfiles[modalityKey] || {};
    const rows = Object.entries(modality.methods).map(([method, metrics]) => ({ method, metrics }));
    rows.sort((left, right) => {
      const leftValue = getSortValue(left.metrics, stateRef.sortBy, modality);
      const rightValue = getSortValue(right.metrics, stateRef.sortBy, modality);
      if (leftValue === undefined || rightValue === undefined) {
        return left.method.localeCompare(right.method);
      }
      return stateRef.sortBy === "dice" ? rightValue - leftValue : leftValue - rightValue;
    });

    const metricKeys = ["dice", "hd95", "sen", "ppv"].filter((metric) => metricExistsForModality(modality, metric));
    const rankMaps = {};
    metricKeys.forEach((metric) => {
      modality.regions.forEach((region) => {
        rankMaps[`${metric}-${region}`] = rankMap(
          rows
            .map((row) => ({ method: row.method, value: getMetricValue(row.metrics, metric, modality, region) }))
            .filter((row) => row.value !== undefined),
          "value",
          metric !== "hd95"
        );
      });
    });

    const targetDice = getRegionalAverage(modality.baseline.target, "dice", modality);
    const contextChips = [
      { label: "Task", value: profile.task || "—" },
      { label: "Dataset", value: profile.source && profile.target ? `${profile.source} → ${profile.target}` : "—", wide: true },
      { label: "Runtime", value: profile.dimension || "Mixed" },
      { label: "Regions", value: modality.regions.join(" / ") },
      { label: "Shift", value: modality.domainShift }
    ];

    return `
      <section class="dataset-section" id="dataset-${modalityKey}">
        <div class="dataset-section__head">
          <div>
            <h3 class="table-block__title">${modalityKey} dataset performance</h3>
            <p class="section-text">${escapeHtml(profile.source || "Source")} → ${escapeHtml(profile.target || "Target")} · ${rows.length} methods · ${modality.regions.join(", ")} regions</p>
          </div>
        </div>
        <div class="context-strip dataset-context">
          ${contextChips
            .map(
              (chip) => `
                <div class="context-chip ${chip.wide ? "context-chip--wide" : ""}">
                  <span class="context-chip__label">${escapeHtml(chip.label)}</span>
                  <span class="context-chip__value">${escapeHtml(chip.value)}</span>
                </div>
              `
            )
            .join("")}
        </div>
        <div class="table-shell">
          <table class="paper-table paper-table--metrics paper-table--regional">
            <thead>
              <tr>
                <th rowspan="2">Method</th>
                <th rowspan="2">Paradigm</th>
                ${modality.regions.map((region) => `<th colspan="${metricKeys.length}">${region}</th>`).join("")}
              </tr>
              <tr>
                ${modality.regions
                  .map((region) => metricKeys.map((metric) => `<th>${region} ${modalityMetricLabels[metric]}</th>`).join(""))
                  .join("")}
              </tr>
            </thead>
            <tbody>
              <tr class="group-divider">
                <th>Intra-domain</th>
                <td>Source baseline</td>
                ${modality.regions.map((region) => metricKeys.map((metric) => metricCell(modality.baseline.intra, metric, modality, region)).join("")).join("")}
              </tr>
              <tr>
                <th>Target-domain (w/o TTA)</th>
                <td>Target baseline</td>
                ${modality.regions.map((region) => metricKeys.map((metric) => metricCell(modality.baseline.target, metric, modality, region)).join("")).join("")}
              </tr>
              ${rows
                .map((row) => {
                  const metadata = data.methods[row.method];
                  const dice = getRegionalAverage(row.metrics, "dice", modality);
                  const belowTarget = targetDice !== undefined && dice !== undefined && dice < targetDice;
                  const paradigm = data.paradigms[metadata.paradigm];
                  return `
                    <tr class="${belowTarget ? "below-target" : ""}">
                      <td>${methodLabel(row.method, metadata)}</td>
                      <td>${paradigm.symbol} ${paradigm.label}</td>
                      ${modality.regions
                        .map((region) =>
                          metricKeys
                            .map((metric) => metricCell(row.metrics, metric, modality, region, rankClass(rankMaps[`${metric}-${region}`][row.method])))
                            .join("")
                        )
                        .join("")}
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      </section>
    `;
  }

  function renderLocalCodeSection(stateRef) {
    const { data } = stateRef;
    const summaryByMethod = Object.fromEntries(data.methodSummary.map((row) => [row.method, row]));
    const container = document.getElementById("local-grid");

    container.innerHTML = paradigmOrder
      .map((paradigmKey) => {
        const paradigm = data.paradigms[paradigmKey];
        const methods = Object.entries(data.methods)
          .filter(([, metadata]) => metadata.paradigm === paradigmKey)
          .sort((left, right) => summaryByMethod[right[0]].meanDice - summaryByMethod[left[0]].meanDice);
        const localCount = methods.filter(([, metadata]) => metadata.local).length;

        if (methods.length === 0) {
          return "";
        }

        return `
          <section class="local-group">
            <div class="local-group__header">
              <div>
                <h3 class="local-group__title">${paradigm.label}</h3>
              </div>
              <p class="local-group__meta">${methods.length} methods, ${localCount} with code</p>
            </div>
            <div class="local-group__methods">
              ${methods
                .map(([methodName, metadata]) => {
                  const summary = summaryByMethod[methodName];
                  return `
                    <article class="local-method">
                      <div class="local-method__header">
                        <div>
                          <h4 class="local-method__name">${escapeHtml(methodName)}</h4>
                          <p class="local-method__meta">${escapeHtml(metadata.venue)} ${escapeHtml(metadata.ref)}</p>
                        </div>
                        <div class="local-method__metrics mono">
                          <span>Dice ${formatMetric(summary.meanDice, "dice")}</span>
                          <span>HD95 ${formatMetric(summary.meanHd95, "hd95")}</span>
                        </div>
                      </div>
                      <div class="method-label__links">
                        ${buildRoutePills(metadata, data.meta.repositoryUrl, { includeRoot: true, className: "inline-pill" })}
                      </div>
                    </article>
                  `;
                })
                .join("")}
            </div>
          </section>
        `;
      })
      .join("");
  }

  function stackTransition(sourceValue, targetValue, metric) {
    return `
      <div class="metric-stack">
        <span class="metric-stack__line mono">${formatMetric(sourceValue, metric)}</span>
        <span class="metric-stack__line"><span class="arrow">↓</span></span>
        <span class="metric-stack__line mono">${formatMetric(targetValue, metric)}</span>
      </div>
    `;
  }

  function buildRoutePills(metadata, repositoryUrl, options) {
    if (!metadata.local || !metadata.codePath) {
      return "";
    }

    const settings = {
      includeRoot: true,
      className: "inline-pill",
      ...options
    };
    const links = [];
    if (settings.includeRoot) {
      links.push({ label: "Root", path: metadata.codePath });
    }
    Object.entries(metadata.routes || {}).forEach(([dimension, path]) => {
      links.push({ label: formatDimensionLabel(dimension), path });
    });

    return links
      .map(
        (link) => `
          <a class="${settings.className}" href="${repositoryUrl}/tree/main/${link.path}" target="_blank" rel="noreferrer">
            ${link.label}
          </a>
        `
      )
      .join("");
  }

  function getRootPath(routes) {
    const routeValues = Object.values(routes);
    if (routeValues.length === 0) {
      return "";
    }
    return routeValues[0].replace(/\/(?:two_d|three_d)$/, "");
  }

  function availabilityCell(metadata, repositoryUrl) {
    if (!metadata.local || !metadata.codePath) {
      return `<span class="inline-pill">Paper only</span>`;
    }
    return `
      <div class="method-label__links">
        <span class="inline-pill inline-pill--local">In repo</span>
        ${buildRoutePills(metadata, repositoryUrl, { includeRoot: true, className: "inline-pill" })}
      </div>
    `;
  }

  function methodLabel(methodName, metadata) {
    const paradigm = state.data.paradigms[metadata.paradigm];
    return `
      <div class="method-label">
        <span class="method-label__symbol" style="color:${paradigm.color}">${paradigm.symbol}</span>
        <div class="method-label__text">
          <span class="method-label__name">${escapeHtml(methodName)}</span>
          <span class="method-label__meta">${escapeHtml(metadata.venue)} ${escapeHtml(metadata.ref)}</span>
        </div>
      </div>
    `;
  }

  function rankMap(rows, key, higherIsBetter, idKey) {
    const resolvedIdKey = idKey || "method";
    const sorted = [...rows].sort((left, right) => {
      const a = left[key];
      const b = right[key];
      return higherIsBetter ? b - a : a - b;
    });
    const map = {};
    sorted.forEach((row, index) => {
      map[row[resolvedIdKey]] = index + 1;
    });
    return map;
  }

  function rankClass(rank) {
    if (rank === 1) return "rank-1";
    if (rank === 2) return "rank-2";
    if (rank === 3) return "rank-3";
    return "";
  }

  function groupBy(rows, key) {
    return rows.reduce((accumulator, row) => {
      const bucket = row[key];
      if (!accumulator[bucket]) {
        accumulator[bucket] = [];
      }
      accumulator[bucket].push(row);
      return accumulator;
    }, {});
  }

  function average(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
  }

  function formatDimensionLabel(dimension) {
    if (dimension === "two_d") {
      return "2D";
    }
    if (dimension === "three_d") {
      return "3D";
    }
    return dimension;
  }

  function formatMetric(value, metric) {
    if (metric === "dice") {
      return Number(value).toFixed(4);
    }
    return Number(value).toFixed(2);
  }

  function escapeHtml(value) {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }
})();
