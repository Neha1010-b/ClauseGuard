/* Results page logic */

let DATA = null;             // {id, filename, created_at, analysis}
let ACTIVE_TAB = "overview";
let ACTIVE_FILTER = "all";
let SEARCH_QUERY = "";

// ============================================================
// Boot
// ============================================================
async function init() {
    const user = await bootAuthedPage("results");
    if (!user) return;

    const params = new URLSearchParams(window.location.search);
    const docId = params.get("id");

    if (!docId) {
        showError("No document specified.");
        return;
    }

    try {
        DATA = await apiFetch(`/documents/${docId}`);
        renderPage();
    } catch (e) {
        showError(e.message || "Failed to load analysis");
    }
}

function showError(msg) {
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("error").classList.remove("hidden");
    document.getElementById("error-message").textContent = msg;
}

// ============================================================
// Render main
// ============================================================
function renderPage() {
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("content").classList.remove("hidden");
    document.getElementById("doc-title").textContent = DATA.filename;

    setupTabs();
    renderOverview();
    renderClauses();
    renderAnalysis();
    renderSummary();
}

// ============================================================
// Tabs
// ============================================================
function setupTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });
    switchTab("overview");
}

function switchTab(name) {
    ACTIVE_TAB = name;
    document.querySelectorAll(".tab-btn").forEach(btn => {
        const active = btn.dataset.tab === name;
        btn.classList.toggle("border-emerald-600", active);
        btn.classList.toggle("text-emerald-600", active);
        btn.classList.toggle("border-transparent", !active);
        btn.classList.toggle("text-slate-500", !active);
    });
    document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.toggle("hidden", pane.id !== `tab-${name}`);
    });
}

// ============================================================
// Overview
// ============================================================
function renderOverview() {
    const a = DATA.analysis;
    const s = a.summary;
    const doc = a.document;

    document.getElementById("doc-type").textContent = doc.format.toUpperCase();
    const substantive = s.total_clauses - s.skipped_structural;
    document.getElementById("doc-clauses").textContent = `${s.total_clauses} (${substantive} analyzed)`;
    document.getElementById("doc-pages").textContent = doc.pages;
    document.getElementById("doc-size").textContent = (doc.chars / 1000).toFixed(1) + "K chars";
    document.getElementById("doc-date").textContent = formatDate(DATA.created_at);

    const overall = s.high > 0 ? "High" : s.medium > 0 ? "Medium" : "Low";
    const overallEl = document.getElementById("doc-overall-risk");
    overallEl.textContent = overall;
    overallEl.className = "text-sm font-semibold " + (
        overall === "High" ? "text-red-600" :
        overall === "Medium" ? "text-amber-600" : "text-emerald-600"
    );

    document.getElementById("count-high").textContent = s.high;
    document.getElementById("count-medium").textContent = s.medium;
    document.getElementById("count-low").textContent = s.low;

    drawDonut(s.high, s.medium, s.low);

    // Key findings = top 5 non-low clauses
    const findings = a.clauses
        .filter(c => c.risk.level !== "low" && c.structural_role !== "SIGNATURE")
        .sort((x, y) => y.risk.score - x.risk.score)
        .slice(0, 5);

    const container = document.getElementById("key-findings");
    if (findings.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-slate-500 text-sm">
                No high or medium-risk clauses found. This is a good sign.
            </div>`;
        return;
    }

    container.innerHTML = findings.map(c => {
        const isHigh = c.risk.level === "high";
        return `
            <div class="flex items-start gap-4 p-4 rounded-lg border ${isHigh ? 'bg-red-50/50 border-red-200' : 'bg-amber-50/50 border-amber-200'}">
                <div class="w-8 h-8 rounded-full ${isHigh ? 'bg-red-100' : 'bg-amber-100'} flex items-center justify-center flex-shrink-0">
                    <svg class="w-4 h-4 ${isHigh ? 'text-red-600' : 'text-amber-600'}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                    </svg>
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                        <span class="text-sm font-semibold text-slate-900">${escapeHtml(c.classification.label || "Unclassified")}</span>
                        <span class="text-xs px-2 py-0.5 rounded-full font-medium ${isHigh ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}">
                            ${isHigh ? 'High Risk' : 'Medium'}
                        </span>
                        <span class="text-xs text-slate-400">§ ${escapeHtml(c.number || c.id)}</span>
                    </div>
                    <div class="text-sm text-slate-600 line-clamp-2">${escapeHtml(c.text.substring(0, 220))}${c.text.length > 220 ? '...' : ''}</div>
                </div>
                <button onclick="jumpToClause(${c.id})" class="text-xs text-slate-500 hover:text-slate-900 flex-shrink-0">View →</button>
            </div>
        `;
    }).join("");
}

function jumpToClause(id) {
    switchTab("clauses");
    ACTIVE_FILTER = "all";
    SEARCH_QUERY = "";
    document.getElementById("search-clauses").value = "";
    updateFilterChips();
    renderClauses();
    setTimeout(() => {
        const el = document.getElementById(`clause-${id}`);
        if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "center" });
            el.classList.add("ring-2", "ring-emerald-500");
            setTimeout(() => el.classList.remove("ring-2", "ring-emerald-500"), 2000);
        }
    }, 100);
}

// ============================================================
// Donut chart (pure canvas, no dependency)
// ============================================================
function drawDonut(high, medium, low) {
    const canvas = document.getElementById("risk-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const total = high + medium + low;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 160 * dpr;
    canvas.height = 160 * dpr;
    canvas.style.width = "160px";
    canvas.style.height = "160px";
    ctx.scale(dpr, dpr);

    const cx = 80, cy = 80, outer = 68, inner = 46;

    if (total === 0) {
        ctx.beginPath();
        ctx.arc(cx, cy, (outer + inner) / 2, 0, Math.PI * 2);
        ctx.lineWidth = outer - inner;
        ctx.strokeStyle = "#e2e8f0";
        ctx.stroke();
        return;
    }

    const segments = [
        { value: high, color: "#ef4444" },
        { value: medium, color: "#fbbf24" },
        { value: low, color: "#10b981" },
    ];

    let start = -Math.PI / 2;
    segments.forEach(seg => {
        if (seg.value === 0) return;
        const angle = (seg.value / total) * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(cx, cy, (outer + inner) / 2, start, start + angle);
        ctx.lineWidth = outer - inner;
        ctx.strokeStyle = seg.color;
        ctx.stroke();
        start += angle;
    });

    // Center text
    ctx.fillStyle = "#0f172a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 26px Inter, sans-serif";
    ctx.fillText(total, cx, cy - 6);
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.fillText("CLAUSES", cx, cy + 14);
}

// ============================================================
// Clauses tab
// ============================================================
function updateFilterChips() {
    document.querySelectorAll(".filter-chip").forEach(chip => {
        const active = chip.dataset.filter === ACTIVE_FILTER;
        chip.className = active
            ? "filter-chip px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-900 text-white"
            : "filter-chip px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50";
    });
}

function setupClauseControls() {
    document.querySelectorAll(".filter-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            ACTIVE_FILTER = chip.dataset.filter;
            updateFilterChips();
            renderClauses();
        });
    });
    const search = document.getElementById("search-clauses");
    search.addEventListener("input", () => {
        SEARCH_QUERY = search.value.toLowerCase().trim();
        renderClauses();
    });
}

function renderClauses() {
    setupClauseControls();
    const a = DATA.analysis;
    let clauses = a.clauses.filter(c => c.structural_role !== "SIGNATURE");

    if (ACTIVE_FILTER !== "all") {
        clauses = clauses.filter(c => c.risk.level === ACTIVE_FILTER);
    }
    if (SEARCH_QUERY) {
        clauses = clauses.filter(c =>
            c.text.toLowerCase().includes(SEARCH_QUERY) ||
            (c.classification.label || "").toLowerCase().includes(SEARCH_QUERY)
        );
    }

    // Sort: high first, then medium, then low; secondary by score descending
    const levelOrder = { high: 0, medium: 1, low: 2 };
    clauses.sort((x, y) => {
        const dx = levelOrder[x.risk.level] - levelOrder[y.risk.level];
        if (dx !== 0) return dx;
        return y.risk.score - x.risk.score;
    });

    const container = document.getElementById("clause-list");
    if (clauses.length === 0) {
        container.innerHTML = `
            <div class="bg-white rounded-xl border border-slate-200 p-12 text-center">
                <p class="text-sm text-slate-500">No clauses match this filter.</p>
            </div>`;
        return;
    }

    container.innerHTML = clauses.map(c => renderClauseCard(c)).join("");
}

function renderClauseCard(c) {
    const levelStyles = {
        high: { border: "border-red-500", badge: "bg-red-100 text-red-700", label: "HIGH RISK" },
        medium: { border: "border-amber-400", badge: "bg-amber-100 text-amber-700", label: "MEDIUM RISK" },
        low: { border: "border-emerald-500", badge: "bg-emerald-100 text-emerald-700", label: "LOW RISK" },
    };
    const s = levelStyles[c.risk.level];

    const hasExplanation = c.explanation && c.explanation.text;
    const hasTags = c.risk.signals && c.risk.signals.risky_language_tags && c.risk.signals.risky_language_tags.length > 0;
    const categories = c.risk.categories || [];

    return `
        <div id="clause-${c.id}" class="bg-white rounded-xl border-l-4 ${s.border} border-y border-r border-slate-200 overflow-hidden transition-all">
            <details class="group">
                <summary class="px-6 py-4 flex items-start gap-4 cursor-pointer hover:bg-slate-50/50">
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2 mb-2 flex-wrap">
                            <span class="text-xs font-mono text-slate-500">§ ${escapeHtml(c.number || String(c.id))}</span>
                            <span class="text-xs px-2 py-0.5 rounded-full font-semibold ${s.badge}">${s.label}</span>
                            <span class="text-xs text-slate-500">${escapeHtml(c.classification.label || "Unclassified")}</span>
                            <span class="text-xs text-slate-400">Score ${c.risk.score.toFixed(2)}</span>
                            ${categories.map(cat => `<span class="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">${escapeHtml(cat)}</span>`).join("")}
                        </div>
                        <div class="text-sm text-slate-700 line-clamp-3 clause-text">${escapeHtml(c.text.substring(0, 280))}${c.text.length > 280 ? '...' : ''}</div>
                    </div>
                    <svg class="w-5 h-5 text-slate-400 flex-shrink-0 mt-1 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                    </svg>
                </summary>
                <div class="px-6 pb-6 pt-2 border-t border-slate-100">
                    <!-- Full clause text -->
                    <div class="mb-5">
                        <div class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Full Clause</div>
                        <div class="bg-slate-50 rounded-lg p-4 clause-text text-slate-700">${escapeHtml(c.text)}</div>
                    </div>

                    <!-- Signals row -->
                    ${hasTags ? `
                    <div class="mb-5">
                        <div class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Risky Language Detected</div>
                        <div class="flex flex-wrap gap-2">
                            ${c.risk.signals.risky_language_tags.map(t => `
                                <span class="text-xs px-2.5 py-1 rounded-full bg-red-50 text-red-700 border border-red-100 font-medium">${escapeHtml(t)}</span>
                            `).join("")}
                        </div>
                    </div>` : ""}

                    <!-- Explanation -->
                    ${hasExplanation ? `
                    <div class="bg-emerald-50/50 border border-emerald-100 rounded-lg p-5 mb-4">
                        <div class="flex items-center gap-2 mb-3">
                            <svg class="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                            </svg>
                            <span class="text-xs font-semibold text-emerald-900 uppercase tracking-wide">AI Explanation</span>
                        </div>
                        <p class="text-sm text-slate-700 leading-relaxed">${escapeHtml(c.explanation.text)}</p>
                        ${c.explanation.suggested_action ? `
                            <div class="mt-4 pt-4 border-t border-emerald-100">
                                <div class="text-xs font-semibold text-emerald-900 uppercase tracking-wide mb-2">Suggested Action</div>
                                <p class="text-sm text-slate-700 leading-relaxed">${escapeHtml(c.explanation.suggested_action)}</p>
                            </div>
                        ` : ""}
                        ${c.explanation.model_used ? `
                            <div class="mt-3 text-xs text-slate-500">
                                Generated by ${escapeHtml(c.explanation.model_used)} in ${c.explanation.latency_ms}ms
                            </div>
                        ` : ""}
                    </div>` : `
                    <div class="bg-slate-50 border border-slate-200 rounded-lg p-4 mb-4 text-xs text-slate-500 italic">
                        ${c.risk.level === "low" ? "Low-risk clause — no AI explanation generated." :
                          "No AI explanation available for this clause."}
                    </div>`}

                    <!-- Meta -->
                    <div class="grid grid-cols-3 gap-4 pt-4 border-t border-slate-100 text-xs">
                        <div>
                            <div class="text-slate-500 mb-1">Classifier Confidence</div>
                            <div class="font-medium text-slate-900">${(c.classification.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                            <div class="text-slate-500 mb-1">Deviation from Standard</div>
                            <div class="font-medium text-slate-900">${(c.risk.signals.deviation_from_reference * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                            <div class="text-slate-500 mb-1">Char Range</div>
                            <div class="font-medium text-slate-900 font-mono">[${c.char_start}, ${c.char_end}]</div>
                        </div>
                    </div>
                </div>
            </details>
        </div>
    `;
}

// ============================================================
// Analysis tab
// ============================================================
function renderAnalysis() {
    const a = DATA.analysis;

    // Category breakdown
    const categories = { Legal: 0, Financial: 0, Operational: 0, Ambiguity: 0 };
    a.clauses.forEach(c => {
        (c.risk.categories || []).forEach(cat => {
            if (cat in categories) categories[cat]++;
        });
    });

    const maxCount = Math.max(1, ...Object.values(categories));
    document.getElementById("category-breakdown").innerHTML = Object.entries(categories).map(([name, count]) => {
        const pct = (count / maxCount) * 100;
        const colors = {
            Legal: "bg-blue-500", Financial: "bg-emerald-500",
            Operational: "bg-amber-500", Ambiguity: "bg-slate-500"
        };
        return `
            <div>
                <div class="flex items-center justify-between text-sm mb-1">
                    <span class="text-slate-700">${name}</span>
                    <span class="font-medium text-slate-900">${count}</span>
                </div>
                <div class="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div class="${colors[name]} h-full transition-all" style="width: ${pct}%"></div>
                </div>
            </div>
        `;
    }).join("");

    // Top clause types
    const typeCounts = {};
    a.clauses.forEach(c => {
        const t = c.classification.label || "Unclassified";
        typeCounts[t] = (typeCounts[t] || 0) + 1;
    });
    const sorted = Object.entries(typeCounts).sort((x, y) => y[1] - x[1]).slice(0, 10);

    document.getElementById("clause-types").innerHTML = sorted.map(([name, count]) => `
        <div class="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
            <span class="text-sm text-slate-700 truncate">${escapeHtml(name)}</span>
            <span class="text-sm font-semibold text-slate-900 ml-2">${count}</span>
        </div>
    `).join("");
}

// ============================================================
// Summary tab
// ============================================================
function renderSummary() {
    const a = DATA.analysis;
    const s = a.summary;

    const topRisks = a.clauses
        .filter(c => c.risk.level === "high" || c.risk.level === "medium")
        .sort((x, y) => y.risk.score - x.risk.score)
        .slice(0, 5);

    const overall = s.high > 0 ? "high" : s.medium > 0 ? "medium" : "low";
    const overallText = {
        high: `This document contains ${s.high} high-risk clause${s.high !== 1 ? 's' : ''} that warrant careful review before signing.`,
        medium: `This document contains ${s.medium} clause${s.medium !== 1 ? 's' : ''} of moderate concern. Review the flagged items below.`,
        low: `No high or medium-risk clauses were detected. The document appears to follow standard commercial terms.`,
    };

    const html = `
        <div class="bg-${overall === 'high' ? 'red' : overall === 'medium' ? 'amber' : 'emerald'}-50 border border-${overall === 'high' ? 'red' : overall === 'medium' ? 'amber' : 'emerald'}-200 rounded-lg p-5">
            <div class="flex items-center gap-3 mb-2">
                <span class="text-xs font-semibold uppercase tracking-wide text-${overall === 'high' ? 'red' : overall === 'medium' ? 'amber' : 'emerald'}-700">Overall Assessment</span>
            </div>
            <p class="text-sm text-slate-800">${overallText[overall]}</p>
        </div>

        ${topRisks.length > 0 ? `
        <div>
            <h3 class="text-sm font-semibold text-slate-900 mb-3">Clauses Requiring Attention</h3>
            <div class="space-y-3">
                ${topRisks.map(c => `
                    <div class="border border-slate-200 rounded-lg p-4">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="text-xs font-medium px-2 py-0.5 rounded ${c.risk.level === 'high' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}">
                                ${c.risk.level.toUpperCase()}
                            </span>
                            <span class="text-sm font-medium text-slate-900">${escapeHtml(c.classification.label || "Unclassified")}</span>
                            <span class="text-xs text-slate-400">§ ${escapeHtml(c.number || c.id)}</span>
                        </div>
                        ${c.explanation && c.explanation.text ? `
                            <p class="text-sm text-slate-600 mb-2">${escapeHtml(c.explanation.text.substring(0, 300))}${c.explanation.text.length > 300 ? '...' : ''}</p>
                        ` : `<p class="text-sm text-slate-600 mb-2">${escapeHtml(c.risk.explanation_hint)}</p>`}
                        ${c.explanation && c.explanation.suggested_action ? `
                            <div class="text-xs text-emerald-700 bg-emerald-50 rounded px-3 py-2 border border-emerald-100">
                                <span class="font-semibold">Recommendation: </span>${escapeHtml(c.explanation.suggested_action)}
                            </div>
                        ` : ""}
                    </div>
                `).join("")}
            </div>
        </div>
        ` : `
        <div class="text-center py-8 text-sm text-slate-500">
            No specific recommendations — the document is well within standard practice.
        </div>
        `}

        <div class="pt-4 border-t border-slate-100 text-xs text-slate-500">
            <strong>Disclaimer:</strong> ClauseGuard is an AI tool. This analysis is not a substitute for legal advice.
            Always consult a qualified lawyer before signing.
        </div>
    `;

    document.getElementById("summary-content").innerHTML = html;
}

// ============================================================
// Utils
// ============================================================
function printReport() {
    // Temporarily reveal all tab panes so they appear in the printout
    const panes = document.querySelectorAll(".tab-pane");
    const previous = [];
    panes.forEach(p => {
        previous.push(p.classList.contains("hidden"));
        p.classList.remove("hidden");
    });
    // Give the browser a tick to re-layout
    setTimeout(() => {
        window.print();
        // Restore after print dialog closes
        setTimeout(() => {
            panes.forEach((p, i) => {
                if (previous[i]) p.classList.add("hidden");
                else p.classList.remove("hidden");
            });
        }, 500);
    }, 100);
}

// Boot
init();