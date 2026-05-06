(async function bootstrap() {
  const state = {
    data: null,
    view: "overview",
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
  const viewIds = new Set(["overview", "summary", "modalities"]);
  const localMethodRoutes = {
    "SFDA-FSM": {
      two_d: "input_level_transformation/SFDA-FSM/two_d",
      three_d: "input_level_transformation/SFDA-FSM/three_d"
    },
    GraTa: {
      two_d: "feature_level_alignment/GraTa/two_d",
      three_d: "feature_level_alignment/GraTa/three_d"
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
        <p class="section-kicker">Leaderboard failed to load</p>
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
    });
  }

  function initializeStaticContent(stateRef) {
    const { data } = stateRef;
    const localCount = Object.values(data.methods).filter((method) => method.local).length;
    const jumpCount = Object.values(data.methods).reduce(
      (total, method) => total + Object.keys(method.routes || {}).length,
      0
    );

    document.title = data.meta.title;
    document.getElementById("hero-lede").textContent = data.meta.subtitle;
    document.getElementById("scope-note").textContent = data.meta.mvpScope;
    document.getElementById("footer-updated").textContent = `Updated ${data.meta.updated} from benchmark tables.`;

    const stats = [
      { value: Object.keys(data.modalityLeaderboards).length, label: "Modalities" },
      { value: paradigmOrder.length, label: "Paradigms" },
      { value: Object.keys(data.methods).length, label: "Methods" },
      { value: jumpCount, label: "Code jumps" },
      { value: localCount, label: "Local methods" }
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

    document.getElementById("paradigm-legend").innerHTML = paradigmOrder
      .map((key) => {
        const paradigm = data.paradigms[key];
        return `
          <span class="paradigm-pill">
            <span class="paradigm-pill__symbol" style="color:${paradigm.color}">${paradigm.symbol}</span>
            <span>${paradigm.label}</span>
          </span>
        `;
      })
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
    renderSpotlights(stateRef);
    renderOverview(stateRef);
    renderSummary(stateRef);
    renderModalityExplorer(stateRef);
    renderLocalCodeSection(stateRef);
  }

  function renderSpotlights(stateRef) {
    const { data } = stateRef;
    const summaryRows = [...data.methodSummary].sort((left, right) => right.meanDice - left.meanDice);
    const localRows = summaryRows.filter((row) => data.methods[row.method].local);
    const winners = computeShiftWinners(data);
    const sourceList = data.meta.sources
      .map((source) => `<li class="compact-list__item">${escapeHtml(source)}</li>`)
      .join("");

    document.getElementById("spotlight-grid").innerHTML = `
      <article class="spotlight-panel">
        <p class="spotlight-panel__kicker">Overall podium</p>
        <h3 class="spotlight-panel__title">Top methods by mean Dice</h3>
        <ol class="podium-list">
          ${summaryRows
            .slice(0, 3)
            .map((row, index) => renderPodiumRow(row, index + 1, data.meta.repositoryUrl))
            .join("")}
        </ol>
      </article>
      <article class="spotlight-panel">
        <p class="spotlight-panel__kicker">Code-ready slice</p>
        <h3 class="spotlight-panel__title">Best local implementations</h3>
        <ol class="podium-list">
          ${localRows
            .slice(0, 3)
            .map((row, index) => renderPodiumRow(row, index + 1, data.meta.repositoryUrl))
            .join("")}
        </ol>
      </article>
      <article class="spotlight-panel">
        <p class="spotlight-panel__kicker">Shift leaders</p>
        <h3 class="spotlight-panel__title">Which paradigm leads by regime</h3>
        <div class="trend-list">
          ${Object.entries(winners)
            .map(([shift, winner]) => {
              const paradigm = data.paradigms[winner.key];
              return `
                <div class="trend-list__row">
                  <div>
                    <span class="trend-list__label">${shift} shift</span>
                    <strong class="trend-list__name">${paradigm.symbol} ${paradigm.shortLabel}</strong>
                  </div>
                  <div class="trend-list__metrics mono">
                    <span>Dice ${formatMetric(winner.meanDice, "dice")}</span>
                    <span>HD95 ${formatMetric(winner.meanHd95, "hd95")}</span>
                  </div>
                </div>
              `;
            })
            .join("")}
        </div>
      </article>
      <article class="spotlight-panel">
        <p class="spotlight-panel__kicker">Source tables</p>
        <h3 class="spotlight-panel__title">MVP coverage</h3>
        <ul class="compact-list">
          <li class="compact-list__item">7 modality views with Dice and HD95 retained</li>
          <li class="compact-list__item">4 paradigm blocks with paper-style highlight rules</li>
          <li class="compact-list__item">Direct GitHub jumps for every local method folder</li>
          ${sourceList}
        </ul>
      </article>
    `;
  }

  function renderPodiumRow(row, rank, repositoryUrl) {
    const metadata = state.data.methods[row.method];
    const pills = buildRoutePills(metadata, repositoryUrl, {
      includeRoot: false,
      className: "inline-pill inline-pill--route"
    });

    return `
      <li class="podium-list__item">
        <div class="podium-list__rank rank-${Math.min(rank, 3)}">${rank}</div>
        <div class="podium-list__body">
          <div class="podium-list__head">
            <strong>${escapeHtml(row.method)}</strong>
            <span class="podium-list__meta">${escapeHtml(metadata.venue)} ${escapeHtml(metadata.ref)}</span>
          </div>
          <div class="podium-list__metrics mono">
            <span>Dice ${formatMetric(row.meanDice, "dice")}</span>
            <span>HD95 ${formatMetric(row.meanHd95, "hd95")}</span>
          </div>
          ${pills ? `<div class="method-label__links">${pills}</div>` : ""}
        </div>
      </li>
    `;
  }

  function computeShiftWinners(data) {
    const grouped = groupBy(data.paradigmOverview, "domainShift");
    return Object.fromEntries(
      Object.entries(grouped).map(([shift, rows]) => {
        const entries = paradigmOrder.map((paradigmKey) => {
          const meanDice = average(rows.map((row) => row.scores[paradigmKey].dice));
          const meanHd95 = average(rows.map((row) => row.scores[paradigmKey].hd95));
          return {
            key: paradigmKey,
            meanDice,
            meanHd95
          };
        });
        entries.sort((left, right) => {
          if (right.meanDice !== left.meanDice) {
            return right.meanDice - left.meanDice;
          }
          return left.meanHd95 - right.meanHd95;
        });
        return [shift, entries[0]];
      })
    );
  }

  function renderOverview(stateRef) {
    const container = document.getElementById("overview");
    const grouped = groupBy(stateRef.data.paradigmOverview, "domainShift");

    container.innerHTML = `
      <div class="section-head">
        <div>
          <p class="section-kicker">Table 3 re-framed</p>
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

  function renderSummary(stateRef) {
    const { data } = stateRef;
    const container = document.getElementById("summary");
    const sections = paradigmOrder
      .map((paradigmKey) => {
        const paradigm = data.paradigms[paradigmKey];
        const rows = data.methodSummary
          .filter((row) => data.methods[row.method].paradigm === paradigmKey)
          .sort((left, right) => right.meanDice - left.meanDice);
        const diceRanks = rankMap(rows, "meanDice", true);
        const hd95Ranks = rankMap(rows, "meanHd95", false);

        return `
          <div class="table-block">
            <h3 class="table-block__title">${paradigm.symbol} ${paradigm.label}</h3>
            <div class="table-shell">
              <table class="paper-table">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Paradigm</th>
                    <th>Mean Dice ↑</th>
                    <th>Mean HD95 ↓</th>
                    <th>Availability</th>
                  </tr>
                </thead>
                <tbody>
                  ${rows
                    .map((row) => {
                      const metadata = data.methods[row.method];
                      return `
                        <tr>
                          <td>${methodLabel(row.method, metadata)}</td>
                          <td>${paradigm.symbol} ${paradigm.label}</td>
                          <td class="metric-cell mono ${rankClass(diceRanks[row.method])}">${formatMetric(row.meanDice, "dice")}</td>
                          <td class="metric-cell mono ${rankClass(hd95Ranks[row.method])}">${formatMetric(row.meanHd95, "hd95")}</td>
                          <td>${availabilityCell(metadata, data.meta.repositoryUrl)}</td>
                        </tr>
                      `;
                    })
                    .join("")}
                </tbody>
              </table>
            </div>
          </div>
        `;
      })
      .join("");

    container.innerHTML = `
      <div class="section-head">
        <div>
          <p class="section-kicker">Table 4 re-framed</p>
          <h2 class="section-title">Method summary across all seven modalities</h2>
          <p class="section-text">
            Within each paradigm, rows stay sorted by mean Dice. Top-three metric cells keep the paper's emphasis, while local methods expose direct root, 2D, and 3D jumps wherever that code is available in this repository.
          </p>
        </div>
      </div>
      <div class="summary-grid">${sections}</div>
    `;
  }

  function renderModalityExplorer(stateRef) {
    const { data } = stateRef;
    const modalityKeys = modalityOrder.filter((key) => data.modalityLeaderboards[key]);
    const modalityTabs = document.getElementById("modality-tabs");
    const sortTabs = document.getElementById("sort-tabs");
    const regionTabs = document.getElementById("region-tabs");

    modalityTabs.innerHTML = modalityKeys
      .map(
        (key) => `
          <button type="button" class="segmented__button ${stateRef.modality === key ? "is-active" : ""}" data-modality="${key}">
            ${key}
          </button>
        `
      )
      .join("");
    modalityTabs.querySelectorAll("[data-modality]").forEach((button) => {
      button.addEventListener("click", () => {
        stateRef.modality = button.dataset.modality;
        if (stateRef.modality !== "MRI") {
          stateRef.region = "WT";
        }
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

    const modality = data.modalityLeaderboards[stateRef.modality];
    if (modality.regional) {
      regionTabs.innerHTML = modality.regions
        .map(
          (region) => `
            <button type="button" class="segmented__button ${stateRef.region === region ? "is-active" : ""}" data-region="${region}">
              ${region}
            </button>
          `
        )
        .join("");
      regionTabs.parentElement.style.display = "";
      regionTabs.querySelectorAll("[data-region]").forEach((button) => {
        button.addEventListener("click", () => {
          stateRef.region = button.dataset.region;
          renderModalityExplorer(stateRef);
        });
      });
    } else {
      regionTabs.innerHTML = "";
      regionTabs.parentElement.style.display = "none";
    }

    const container = document.getElementById("modalities");
    if (!container.querySelector(".section-head")) {
      container.insertAdjacentHTML(
        "afterbegin",
        `
          <div class="section-head">
            <div>
              <p class="section-kicker">Tables 7-14 re-framed</p>
              <h2 class="section-title">Modality drilldown</h2>
              <p class="section-text">
                Each modality keeps its own baseline context, dataset pair, and ranking logic. MRI remains region-aware, while the code column continues to point back into the repository for locally available methods.
              </p>
            </div>
          </div>
        `
      );
    }

    renderModalityContext(stateRef, modality);
    renderModalityTable(stateRef, modality);
  }

  function renderModalityContext(stateRef, modality) {
    const profile = modalityProfiles[stateRef.modality] || {};
    const context = document.getElementById("modality-context");
    const baselineDice = modality.regional
      ? modality.baseline.target.dice[stateRef.region]
      : modality.baseline.target.dice;
    const baselineHd95 = modality.regional
      ? modality.baseline.target.hd95[stateRef.region]
      : modality.baseline.target.hd95;

    const chips = [
      { label: "Modality", value: stateRef.modality },
      { label: "Task", value: profile.task || "—" },
      { label: "Dataset pair", value: profile.source && profile.target ? `${profile.source} -> ${profile.target}` : "—", wide: true },
      { label: "Runtime", value: profile.dimension || "Mixed" },
      { label: "Domain shift", value: modality.domainShift }
    ];
    if (modality.regional) {
      chips.push({ label: "MRI region", value: stateRef.region });
    }
    chips.push(
      { label: "Target Dice", value: formatMetric(baselineDice, "dice") },
      { label: "Target HD95", value: formatMetric(baselineHd95, "hd95") }
    );

    context.innerHTML = chips
      .map(
        (chip) => `
          <div class="context-chip ${chip.wide ? "context-chip--wide" : ""}">
            <span class="context-chip__label">${chip.label}</span>
            <span class="context-chip__value ${chip.value.includes("->") ? "" : "mono"}">${chip.value}</span>
          </div>
        `
      )
      .join("");
  }

  function renderModalityTable(stateRef, modality) {
    const data = stateRef.data;
    const rows = Object.entries(modality.methods).map(([method, metrics]) => ({
      method,
      metrics
    }));

    const getDice = (entry) => (modality.regional ? entry.metrics.dice[stateRef.region] : entry.metrics.dice);
    const getHd95 = (entry) => (modality.regional ? entry.metrics.hd95[stateRef.region] : entry.metrics.hd95);
    rows.sort((left, right) => {
      if (stateRef.sortBy === "dice") {
        return getDice(right) - getDice(left);
      }
      return getHd95(left) - getHd95(right);
    });

    const diceRanks = rankMap(
      rows.map((row) => ({ method: row.method, value: getDice(row) })),
      "value",
      true
    );
    const hd95Ranks = rankMap(
      rows.map((row) => ({ method: row.method, value: getHd95(row) })),
      "value",
      false
    );

    const intraDice = modality.regional ? modality.baseline.intra.dice[stateRef.region] : modality.baseline.intra.dice;
    const intraHd95 = modality.regional ? modality.baseline.intra.hd95[stateRef.region] : modality.baseline.intra.hd95;
    const targetDice = modality.regional ? modality.baseline.target.dice[stateRef.region] : modality.baseline.target.dice;
    const targetHd95 = modality.regional ? modality.baseline.target.hd95[stateRef.region] : modality.baseline.target.hd95;

    document.getElementById("modality-table").innerHTML = `
      <div class="table-shell">
        <table class="paper-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Paradigm</th>
              <th>Dice ↑</th>
              <th>HD95 ↓</th>
              <th>Code</th>
            </tr>
          </thead>
          <tbody>
            <tr class="group-divider">
              <th>Intra-domain</th>
              <td>Source baseline</td>
              <td class="metric-cell mono">${formatMetric(intraDice, "dice")}</td>
              <td class="metric-cell mono">${formatMetric(intraHd95, "hd95")}</td>
              <td>—</td>
            </tr>
            <tr>
              <th>Target-domain (w/o TTA)</th>
              <td>Target baseline</td>
              <td class="metric-cell mono">${formatMetric(targetDice, "dice")}</td>
              <td class="metric-cell mono">${formatMetric(targetHd95, "hd95")}</td>
              <td>—</td>
            </tr>
            ${rows
              .map((row) => {
                const metadata = data.methods[row.method];
                const dice = getDice(row);
                const hd95 = getHd95(row);
                const belowTarget = dice < targetDice;
                const paradigm = data.paradigms[metadata.paradigm];
                return `
                  <tr class="${belowTarget ? "below-target" : ""}">
                    <td>${methodLabel(row.method, metadata)}</td>
                    <td>${paradigm.symbol} ${paradigm.label}</td>
                    <td class="metric-cell mono ${rankClass(diceRanks[row.method])}">${formatMetric(dice, "dice")}</td>
                    <td class="metric-cell mono ${rankClass(hd95Ranks[row.method])}">${formatMetric(hd95, "hd95")}</td>
                    <td>${availabilityCell(metadata, data.meta.repositoryUrl)}</td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      </div>
      <p class="table-footnote">
        Rows turn gray when Dice falls below the target-domain baseline for the active modality view. MRI uses the selected region (${modality.regional ? stateRef.region : "not applicable"}) so the ranking stays aligned with the paper's regional evaluation protocol.
      </p>
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
          .filter(([, metadata]) => metadata.paradigm === paradigmKey && metadata.local)
          .sort((left, right) => summaryByMethod[right[0]].meanDice - summaryByMethod[left[0]].meanDice);

        if (methods.length === 0) {
          return "";
        }

        return `
          <section class="local-group">
            <div class="local-group__header">
              <div>
                <p class="section-kicker">${paradigm.symbol} ${paradigm.shortLabel}</p>
                <h3 class="local-group__title">${paradigm.label}</h3>
              </div>
              <p class="local-group__meta">${methods.length} local method${methods.length > 1 ? "s" : ""}</p>
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
                      <p class="local-method__path mono">${escapeHtml(metadata.codePath)}/</p>
                      <div class="method-label__links">
                        <span class="inline-pill inline-pill--local">Local code</span>
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
